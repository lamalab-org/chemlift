from typing import List, Optional, Union
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
from gptchem.extractor import ClassificationExtractor
from gptchem.formatter import ClassificationFormatter
from gptchem.gpt_classifier import GPTClassifier
from more_itertools import chunked
from numpy.typing import ArrayLike
from tqdm import tqdm

from gptjchem.peft_transformers import load_model, train_model, complete, tokenize
from gptjchem.utils import (
    get_mode,
    try_exccept_nan,
    augment_smiles,
)
from transformers.utils import logging
from functools import partial
from peft.utils.save_and_load import set_peft_model_state_dict


class ChemLIFTClassifierFactory:
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

        if "openai" in model_name:
            self = GPTClassifier(model_name, **kwargs)
        else:
            self = PEFTClassifier(model_name, **kwargs)


class PEFTClassifier(GPTClassifier):
    def __init__(
        self,
        property_name: str,
        extractor: ClassificationExtractor = ClassificationExtractor(),
        batch_size: int = 64,
        tune_settings: Optional[dict] = None,
        inference_batch_size: int = 64,
        formatter: Optional[ClassificationFormatter] = None,
        representation_names: Optional[List[str]] = None,
        base_model: str = "EleutherAI/gpt-j-6b",
        load_in_8bit: bool = True,
        lora_kwargs: dict = {},
        tokenizer_kwargs: dict = {},
    ):
        self.property_name = property_name
        self.extractor = extractor
        self.batch_size = batch_size
        self.tune_settings = tune_settings or {}
        self.inference_batch_size = inference_batch_size

        self.formatter = (
            ClassificationFormatter(
                representation_column="repr",
                label_column="prop",
                property_name=property_name,
                num_classes=None,
            )
            if formatter is None
            else formatter
        )
        self.model, self.tokenizer = load_model(
            base_model=base_model, load_in_8bit=load_in_8bit, lora_kwargs=lora_kwargs
        )
        self.representation_names = representation_names if representation_names else []
        self.tokenizer_kwargs = tokenizer_kwargs
        if "cutoff_len" not in self.tokenizer_kwargs:
            self.tokenizer_kwargs["cutoff_len"] = 1024

        self.tune_settings["per_device_train_batch_size"] = self.batch_size

    def _prepare_df(self, X: ArrayLike, y: ArrayLike):
        rows = []
        for i in range(len(X)):
            rows.append({"repr": X[i], "prop": y[i]})
        return pd.DataFrame(rows)

    def return_embeddings(
        self,
        X: ArrayLike,
        layers: Optional[Union[int, List[int]]] = -1,
        padding: bool = True,
        truncation: bool = True,
        insert_in_template: bool = True,
    ):
        """Return embeddings for a set of molecular representations.

        Args:
            X (ArrayLike): Input data (typically array of molecular representations)
            layers (Optional[Union[int, List[int]]], optional): Layers to return embeddings from.
                Defaults to -1.
            padding (bool, optional): Whether to pad the input. Defaults to True.
            truncation (bool, optional): Whether to truncate the input. Defaults to True.
            insert_in_template (bool, optional): Whether to insert the input in the template. Defaults to True.

        Returns:
            ArrayLike: Embeddings
        """
        if insert_in_template:
            X = np.array(X)
            if X.ndim == 1 or (X.ndim == 2 and X.size == len(X)):
                df = self._prepare_df(X, [0] * len(X))
                formatted = self.formatter(df)
            elif X.ndim == 2 and X.size > len(X):
                if not len(self.representation_names) == X.shape[1]:
                    raise ValueError(
                        "Number of representation names must match number of dimensions"
                    )

                dfs = []
                for i in range(X.shape[1]):
                    formatter = deepcopy(self.formatter)
                    formatter.representation_name = self.representation_names[i]
                    df = self._prepare_df(X[:, i], [0] * len(X))
                    formatted = formatter(df)
                    dfs.append(formatted)

                formatted = pd.concat(dfs)
                prompt_text = formatted["prompt"].to_list()
        else:
            prompt_text = X

        embeddings = []

        with torch.no_grad():
            for chunk in tqdm(
                chunked(range(len(prompt_text)), self.inference_batch_size),
                total=len(prompt_text) // self.inference_batch_size,
            ):
                batch = [prompt_text[i] for i in chunk]

                tokenize_partial = partial(
                    tokenize,
                    tokenizer=self.tokenizer,
                    cutoff_len=1024,
                    return_tensors="pt",
                    padding=padding,
                    truncation=truncation,
                )
                prompt = tokenize_partial(batch)
                outs = self.model.forward(
                    prompt["input_ids"], output_hidden_states=True
                )
                if isinstance(layers, int):
                    embeddings.append(outs.hidden_states[layers].cpu().numpy())
                else:
                    embeddings.append(
                        [outs.hidden_states[i].cpu().numpy() for i in layers]
                    )
        # flatten the batch dim
        embeddings = np.concatenate(embeddings, axis=0)

        return embeddings

    def load_state_dict(self, checkpoint_path: str):
        """Load model from checkpoint.

        Args:
            checkpoint_path (str): Path to checkpoint
        """
        set_peft_model_state_dict(self.model, torch.load(checkpoint_path))

    def fit(
        self,
        X: Optional[ArrayLike] = None,
        y: Optional[ArrayLike] = None,
        formatted: Optional[pd.DataFrame] = None,
    ) -> None:
        """Fine tune a GPT-3 model on a dataset.

        Args:
            X (ArrayLike): Input data (typically array of molecular representations)
            y (ArrayLike): Target data (typically array of property values)
            formatted (pd.DataFrame): Formatted data (typically output of `formatter`)
        """
        if formatted is None:
            if X is None or y is None:
                raise ValueError("Either formatted data or X and y must be provided.")

        X = np.array(X)
        y = np.array(y)
        if formatted is None:
            if X.ndim == 1 or (X.ndim == 2 and X.size == len(X)):
                df = self._prepare_df(X, y)
                formatted = self.formatter(df)
            elif X.ndim == 2 and X.size > len(X):
                if not len(self.representation_names) == X.shape[1]:
                    raise ValueError(
                        "Number of representation names must match number of dimensions"
                    )

                dfs = []
                for i in range(X.ndim):
                    formatter = deepcopy(self.formatter)
                    formatter.representation_name = self.representation_names[i]
                    df = self._prepare_df(X[:, i], y)
                    formatted = formatter(df)
                    dfs.append(formatted)

                formatted = pd.concat(dfs)
        train_model(
            self.model,
            self.tokenizer,
            formatted[["prompt", "completion"]],
            train_kwargs=self.tune_settings,
            hub_model_name=None,
            report_to=None,
        )

    def _predict(
        self,
        X: Optional[ArrayLike] = None,
        temperature=0.0,
        do_sample=False,
        formatted: Optional[pd.DataFrame] = None,
    ) -> ArrayLike:
        """Predict property values for a set of molecular representations.

        Args:
            X (ArrayLike): Input data (typically array of molecular representations)
            temperature (float, optional): Temperature for sampling. Defaults to 0.7.
            do_sample (bool, optional): Whether to sample or not. Defaults to False.
            formatted (pd.DataFrame, optional): Formatted data (typically output of `formatter`).
                Defaults to None. If None, X must be provided.

        Returns:
            ArrayLike: Predicted property values
        """

        if formatted is None:
            if X is None:
                raise ValueError("Either formatted data or X must be provided.")

        if formatted is None:
            if X.ndim == 1 or (X.ndim == 2 and X.size == len(X)):
                # if pandas df or series is passed, convert to numpy array
                if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
                    X = X.to_numpy()
                df = self._prepare_df(X, [0] * len(X))
                formatted = self.formatter(df)
                dfs = [formatted]
            elif X.ndim == 2 and X.size > len(X):
                if not len(self.representation_names) == X.shape[1]:
                    raise ValueError(
                        "Number of representation names must match number of dimensions"
                    )

                dfs = []
                for i in range(X.shape[1]):
                    formatter = deepcopy(self.formatter)
                    formatter.representation_name = self.representation_names[i]
                    df = self._prepare_df(X[:, i], [0] * len(X))
                    formatted = formatter(df)
                    dfs.append(formatted)

        else:
            dfs = [formatted]

        predictions = []
        for df in dfs:
            predictions.append(
                self._query(df, temperature=temperature, do_sample=do_sample)
            )

        return predictions

    def predict(
        self,
        X: Optional[ArrayLike] = None,
        temperature=0.7,
        do_sample=False,
        formatted: Optional[pd.DataFrame] = None,
        return_std: bool = True,
    ):
        predictions = self._predict(
            X=X, temperature=temperature, do_sample=do_sample, formatted=formatted
        )

        predictions = np.array(predictions).T

        # nan values make issues here
        predictions_mode = np.array(
            [try_exccept_nan(get_mode, pred) for pred in predictions.astype(int)]
        )

        if return_std:
            predictions_std = np.array(
                [np.std(pred) for pred in predictions.astype(int)]
            )
            return predictions_mode, predictions_std
        return predictions_mode

    def _query(self, formatted_df, temperature, do_sample):
        if temperature > 0 and not do_sample:
            logger = logging.get_logger("transformers")
            logger.warning(
                "Temperature > 0 but do_sample is False. This will result in deterministic predictions. Set do_sample=True to sample from the distribution."
            )
        completions = complete(
            self.model,
            self.tokenizer,
            prompt_text=formatted_df["prompt"].to_list(),
            max_length=self.tokenizer_kwargs["cutoff_len"],
            do_sample=do_sample,
            temperature=temperature,
            batch_size=self.inference_batch_size,
        )

        completions = [c["decoded"] for c in completions]

        extracted = [
            self.extractor.extract(completions[i].split("###")[1])
            for i in range(
                len(completions)
            )  # ToDo: Make it possible to use other splitters than ###
        ]

        filtered = [v if v is not None else np.nan for v in extracted]

        return filtered


class SMILESAugmentedPEFTClassifier(PEFTClassifier):
    def fit(
        self,
        X: Optional[ArrayLike] = None,
        y: Optional[ArrayLike] = None,
        augmentation_rounds: int = 10,
        deduplicate: bool = True,
        include_original: bool = True,
    ) -> None:
        """Fine tune a GPT-3 model on a dataset.

        Args:
            X (ArrayLike): Input data (typically array of molecular representations)
            y (ArrayLike): Target data (typically array of property values)
            augmentation_rounds (int): Number of rounds of augmentation to perform
            deduplicate (bool): Whether to deduplicate the augmented data
            include_original (bool): Whether to include the original data in the training set
        """
        x_augmented = []
        y_augmented = []

        if augmentation_rounds > 1:
            for smiles, label in zip(X, y):
                augmented = augment_smiles(
                    smiles, int_aug=augmentation_rounds, deduplicate=deduplicate
                )
                y_augmented.extend([label] * len(augmented))
                x_augmented.extend(augmented)
        else:
            x_augmented = X
            y_augmented = y

        if include_original:
            x_augmented.extend(X)
            y_augmented.extend(y)

        # shuffle
        x_augmented = np.array(x_augmented)
        y_augmented = np.array(y_augmented)
        idx = np.random.permutation(len(x_augmented))
        x_augmented = x_augmented[idx]
        y_augmented = y_augmented[idx]

        super().fit(X=x_augmented, y=y_augmented)

    def _predict(
        self,
        X: Optional[ArrayLike] = None,
        temperature=0.7,
        do_sample=False,
        augmentation_rounds: int = 0,
        deduplicate: bool = True,
        include_original: bool = True,
    ):
        # we need to also keep track of canonical smiles to be able to aggregate:
        compiled_predictions = []

        for smiles in X:
            if augmentation_rounds > 1:
                augmented = augment_smiles(
                    smiles, int_aug=augmentation_rounds, deduplicate=deduplicate
                )
                if include_original:
                    augmented.append(smiles)
            else:
                augmented = [smiles]
            augmented = np.array(augmented)
            predictions = super()._predict(
                X=augmented, temperature=temperature, do_sample=do_sample
            )[0]
            compiled_predictions.append(predictions)

        return compiled_predictions

    def predict(
        self,
        X: Optional[ArrayLike] = None,
        temperature=0.7,
        do_sample=False,
        augmentation_rounds: int = 0,
        deduplicate: bool = True,
        include_original: bool = True,
    ):
        predictions = self._predict(
            X=X,
            temperature=temperature,
            do_sample=do_sample,
            augmentation_rounds=augmentation_rounds,
            deduplicate=deduplicate,
            include_original=include_original,
        )

        # nan values make issues here
        predictions_mode = np.array(
            [
                try_exccept_nan(get_mode, np.array(pred).astype(int))
                for pred in predictions
            ]
        )
        predictions_std = np.array([np.std(pred) for pred in predictions])
        return predictions_mode, predictions_std
