from loguru import logger
from numpy.typing import ArrayLike

from chemlift.icl.fewshotpredictor import FewShotPredictor


class FewShotClassifier(FewShotPredictor):
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
        print(generations, len(generations))
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
        generations = self._predict(X, generation_kwargs)
        return self._extract(generations, expected_len=len(X))
