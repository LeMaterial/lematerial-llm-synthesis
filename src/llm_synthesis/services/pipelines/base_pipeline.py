from abc import ABC, abstractmethod


class BasePipeline(ABC):
    @abstractmethod
    def run(self, *args, **kwargs):
        """Run the pipeline with the provided arguments."""
        pass
