from loguru import logger
from numpy.typing import ArrayLike
from typing import Union
from chemlift.icl.fewshotpredictor import FewShotPredictor
from chemlift.icl.utils import LangChainChatModelWrapper


class FewShotClassifier(FewShotPredictor):
    """A few-shot classifier using in-context learning."""

    intify = True

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
                    generations_.append(int(g.strip()))
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
