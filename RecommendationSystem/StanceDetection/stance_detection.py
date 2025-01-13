from abc import ABC, abstractmethod

from RecommendationSystem.StanceDetection.Stance import Stance


class StanceDetection(ABC):
    @abstractmethod
    def compute_stance(self, topic: str, comment: str) -> Stance:
        ...
