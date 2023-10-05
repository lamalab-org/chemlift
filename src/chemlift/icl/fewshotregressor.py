from loguru import logger
from numpy.typing import ArrayLike
from chemlift.icl.fewshotpredictor import FewShotPredictor
from typing import Union
from chemlift.icl.utils import LangChainChatModelWrapper
from langchain.llms import BaseLLM
from .fewshotpredictor import Strategy
import numpy as np


class FewShotRegressor(FewShotPredictor):
    """A few-shot regressor using in-context learning."""

    def __init__(
        self,
        llm: Union[BaseLLM, LangChainChatModelWrapper],
        property_name: str,
        n_support: int = 5,
        strategy: Strategy = Strategy.RANDOM,
        seed: int = 42,
        prefix: str = "You are an expert chemist. ",
        max_test: int = 5,
        num_digits: int = 3,
    ):
        """Initialize the few-shot predictor.

        Args:
            llm (Union[BaseLLM, LangChainChatModelWrapper]): The language model to use.
            property_name (str): The property to predict.
            n_support (int, optional): The number of examples to use as support set.
                Defaults to 5.
            strategy (Strategy, optional): The strategy to use to pick the support set.
                Defaults to Strategy.RANDOM.
            seed (int, optional): The random seed to use. Defaults to 42.
            prefix (str, optional): The prefix to use for the prompt.
                Defaults to "You are an expert chemist. ".
            max_test (int, optional): The maximum number of examples to predict at once.
                Defaults to 5.
            num_digits (int, optional): The number of digits to round to.
                Defaults to 3.

        Raises:
            ValueError: If the strategy is unknown.

        Examples:
            >>> from chemlift.icl.fewshotregressor import FewShotRegressor
            >>> from langchain.llms import OpenAI
            >>> llm = OpenAI(model_name="text-ada-001")
            >>> predictor = FewShotRegressor(llm, property_name="melting point")
            >>> predictor.fit(["water", "ethanol"], [0.1, 1.4])
            >>> predictor.predict(["methanol"])
            [0.5]
        """
        self._support_set = None
        self._llm = llm
        self._n_support = n_support
        self._strategy = strategy
        self._seed = seed
        self._property_name = property_name
        self._allowed_values = None
        self._materialclass = "molecules"
        self._max_test = max_test
        self._prefix = prefix
        self._num_digits = num_digits

    def _format_examples(self, examples, targets):
        """Format examples and targets into a string.

        Per default, it is a multiline string with
        - example: target
        """
        return "\n".join(
            [
                f"- {example}: {np.round(target, self._num_digits)}"
                for example, target in zip(examples, targets)
            ]
        )

    def _extract(self, generations, expected_len):
        generations = sum(
            [
                g[0].text.split(":")[-1].replace("Answer: ", "").strip().split(",")
                for generation in generations
                for g in generation.generations
            ],
            [],
        )
        if len(generations) != expected_len:
            logger.warning(f"Expected {expected_len} generations, got {len(generations)}")
            return [None] * expected_len
        original_length = len(generations)
        if self.intify:
            generations_ = []
            for g in generations:
                try:
                    generations_.append(float(g.strip()))
                except Exception:
                    generations_.append(None)
            generations = generations_
        assert len(generations) == original_length
        return generations

    def predict(self, X: ArrayLike, generation_kwargs: dict = {}):
        """Predict the class of a list of examples.

        Args:
            X: A list of examples.
            generation_kwargs: Keyword arguments to pass to the language model.
        """
        generations = self._predict(X, generation_kwargs)
        return self._extract(generations, expected_len=len(X))
