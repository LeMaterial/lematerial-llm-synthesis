import random


class DummyMetric:
    def __init__(self):
        pass

    def __call__(self, preds: str, refs: str) -> float:
        return random.random()
