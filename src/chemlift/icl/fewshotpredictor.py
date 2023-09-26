import random

import numpy as np
from langchain.llms import BaseLLM
from more_itertools import chunked
from numpy.typing import ArrayLike
import enum


class Strategy(enum.Enum):
    RANDOM = "random"
    FIRST = "first"
    DIVERSE = "diverse"


class FewShotPredictor:
    template = """{prefix}What is {property_name} of {queries} given the examples below?
Answer with a comma-separated list of predictions for those {number} {materialclass}.

Examples:
{examples}

Answer:
"""

    template_single = """{prefix}What is {property_name} of {query} given the examples below?
Answer concise by only the prediction on a new line, which is one of {allowed_values}.

Examples:
{examples}

Answer:
"""

    # adding the number of molecules to the prompt is a hack to make sure that the model
    # predicts the correct number of molecules

    def __init__(
        self,
        llm: BaseLLM,
        property_name: str,
        n_support: int = 5,
        strategy: Strategy = Strategy.RANDOM,
        seed: int = 42,
        prefix: str = "You are an expert chemist. ",
    ):
        self._support_set = None
        self._llm = llm
        self._n_support = n_support
        self._strategy = strategy
        self._seed = seed
        self._property_name = property_name
        self._allowed_values = None
        self._materialclass = "molecules"
        self._max_test = 10
        self._prefix = prefix

    def _format_examples(self, examples, targets):
        """Format examples and targets into a string.

        Per default, it is a multiline string with
        - example: target
        """
        return "\n".join([f"- {example}: {target}" for example, target in zip(examples, targets)])

    def _pick_support_indices(self, examples):
        """Pick a support set from a list of examples.

        Args:
            examples: A list of examples.
            n: The number of examples to pick.
            strategy: The strategy to use to pick the support set.
            seed: The random seed to use.

        Returns:
            A list of indices of the support set.
        """
        random.seed(self._seed)
        if self._strategy == Strategy.RANDOM:
            return random.sample(range(len(examples)), self._n_support)
        elif self._strategy == Strategy.FIRST:
            return list(range(self._n_support))
        elif self._strategy == Strategy.DIVERSE:
            from apricot import FeatureBasedSelection
            from sklearn.feature_extraction.text import TfidfVectorizer

            m = TfidfVectorizer().fit_transform(examples)
            selector = FeatureBasedSelection(
                self._n_support, concave_func="sqrt", optimizer="two-stage", verbose=False
            )
            selector.fit(m)
            return selector.idxs
        else:
            raise ValueError(f"Unknown strategy {self._strategy}")

    def _format_queries(self, queries):
        """Format queries into a string.

        Per default, it is a comma-separated list of queries.
        """
        return ", ".join(queries)

    def _predict(self, X: ArrayLike, generation_kwargs: dict = {}):
        """Predict the class of a list of examples.

        Args:
            X: A list of examples.
            generation_kwargs: Keyword arguments to pass to the language model.

        Returns:
            A list of predictions.
        """
        if self._support_set is None:
            raise ValueError("Support set not initialized. Call fit first.")

        predictions = []
        for chunk in chunked(X, self._max_test):
            support_examples, support_targets = self._support_set
            if len(chunk) > 1:
                examples = self._format_examples(support_examples, support_targets)
                queries = self._format_queries(chunk)
                prompt = self.template.format(
                    property_name=self._property_name,
                    queries=queries,
                    examples=examples,
                    number=len(chunk),
                    materialclass=self._materialclass,
                    prefix=self._prefix,
                )
            else:
                examples = self._format_examples(support_examples, support_targets)
                queries = chunk[0]
                allowed_values = ", ".join(map(str, list(self._allowed_values)))
                prompt = self.template_single.format(
                    property_name=self._property_name,
                    query=queries,
                    examples=examples,
                    allowed_values=allowed_values,
                    prefix=self._prefix,
                )

            pred = self._llm.generate([prompt], **generation_kwargs)
            predictions.append(pred)
        return predictions

    def predict(self, X: ArrayLike, generation_kwargs: dict = {}):
        raise NotImplementedError

    def fit(self, X: ArrayLike, y: ArrayLike):
        """Fit the model to a support set.

        Args:
            X: A list of examples.
            y: A list of targets.
        """
        support_indices = self._pick_support_indices(X)
        self._support_set = (np.array(X)[support_indices], np.array(y)[support_indices])
        self._allowed_values = set(y)
