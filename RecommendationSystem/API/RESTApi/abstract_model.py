from abc import ABC, abstractmethod
from typing import List


class AbstractModel(ABC):

    @abstractmethod
    def get_recommendations(self, comment_data: dict) -> List:
        """
        Abstract interface method for the REST API view
        :param comment_data:
        :return:
        """
        pass

