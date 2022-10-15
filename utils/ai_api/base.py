import abc
import typing
from abc import ABC



class ImportExport(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def import_model(*args, **kwargs): ...

    @abc.abstractmethod
    def export_model(self, *args, **kwargs): ...


class Trainable(abc.ABC):
    @abc.abstractmethod
    def fit_model(self, input_: typing.List[str], output: typing.List[str], **kwargs): ...


class Predict(abc.ABC):
    def predict(self, input_: typing.List[str], **kwargs):
        return list(map(self.predict_one, input_))

    @abc.abstractmethod
    def predict_one(self, input_: str, **kwargs):
        ...


class AI(
         ImportExport,
         Trainable, Predict, ABC):
    ...
