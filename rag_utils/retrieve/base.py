from abc import ABC, abstractmethod


class Vectorstore(ABC):
    @abstractmethod
    def transform(self): ...

    @abstractmethod
    def load(self): ...
