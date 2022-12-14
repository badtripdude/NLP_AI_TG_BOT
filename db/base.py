from abc import abstractmethod
from typing import List, Type, TypeVar, Union

T = TypeVar("T")


class RawConnection:
    @staticmethod
    def __make_request(
            sql: str,
            params: Union[tuple, List[tuple]] = None,
            fetch: bool = False,
            mult: bool = False
    ):
        """
        You have to override this method for all synchronous databases (e.g., Sqlite).
        :param sql:
        :param params:
        :param fetch:
        :param mult:
        :return:
        """
        raise NotImplementedError

    @staticmethod
    def _make_request(
            sql: str,
            params: Union[tuple, List[tuple]] = None,
            fetch: bool = False,
            mult: bool = False,
            model_type: Type[T] = None
    ):
        """
        You have to override this method for all synchronous databases (e.g., Sqlite).
        :param sql:
        :param params:
        :param fetch:
        :param mult:
        :param model_type:
        :return:
        """
        raise NotImplementedError


class UsersTableAbstract(RawConnection):  # Interface
    @staticmethod
    @abstractmethod
    def user_exists(user_id) -> bool: ...

    @staticmethod
    @abstractmethod
    def add_user(user_id, exists_checker=True) -> None: ...


class TrainersTableAbstract(RawConnection):

    @staticmethod
    @abstractmethod
    def add_new_trainer(user_id, join_date, **kwargs) -> None: ...

    @staticmethod
    @abstractmethod
    def trainer_exists(user_id) -> bool: ...


class DatasetTableAbstract(RawConnection):
    @staticmethod
    @abstractmethod
    def add_new_translation(user_id, *, input_, output, **kwargs) -> None: ...
