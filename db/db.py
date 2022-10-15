import sqlite3
# from collections.abc import generator
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from loguru import logger

import settings

if __name__ != '__main__':
    from .base import *
else:
    from base import *

T = TypeVar("T")
db_title = settings.DB_TITLE

SCHEMA = """
CREATE TABLE Trainers (id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE NOT NULL, user_id INTEGER UNIQUE NOT NULL REFERENCES Users (user_id) ON DELETE NO ACTION ON UPDATE CASCADE, join_date DATETIME DEFAULT ((DATETIME('now'))))


CREATE TABLE Users (id INTEGER PRIMARY KEY NOT NULL UNIQUE, user_id INTEGER UNIQUE NOT NULL, join_date DATETIME DEFAULT ((DATETIME('now'))) NOT NULL, username VARCHAR, first_name VARCHAR, last_name)


CREATE TABLE Dataset (id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE NOT NULL, user_id INT REFERENCES Trainers (user_id) ON DELETE CASCADE ON UPDATE CASCADE, input STRING NOT NULL ON CONFLICT IGNORE, output STRING NOT NULL ON CONFLICT IGNORE, add_date DATETIME DEFAULT ((DATETIME('now'))));
"""
class SqliteDBConn:
    def __init__(self, db_name):
        self.db_name = db_name

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_name)
        self.conn.row_factory = sqlite3.Row
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()
        if exc_val:
            raise


class SqliteConnection(RawConnection):
    @staticmethod
    def __make_request(
            sql: str,
            params: Union[tuple, List[tuple]] = (),
            fetch: bool = False,
            mult: bool = False
    ) -> Optional[Union[List[Dict[str, Any]], Dict[str, Any]]]:
        with SqliteDBConn(db_title) as conn:
            c = conn.cursor()
            try:
                if isinstance(params, list):
                    c.executemany(sql, params)
                else:
                    c.execute(sql, params)
            except Exception as e:
                logger.error(e)
                ...
            if fetch:
                if mult:
                    r = c.fetchall()
                else:
                    r = c.fetchone()
                return r
            else:
                conn.commit()

    @staticmethod
    def _convert_to_model(data: Optional[dict], model: Type[T]) -> Optional[T]:
        if data is not None:
            return model(**data)
        else:
            return None

    @staticmethod
    def _make_request(
            sql: str,
            params: Union[tuple, List[tuple]] = (),
            fetch: bool = False,
            mult: bool = False,
            model_type: Type[T] = None
    ) -> Optional[Union[List[T], T]]:
        raw = SqliteConnection.__make_request(sql, params, fetch, mult)

        if raw is None:
            if mult:
                return []
            else:
                return None
        else:
            if mult:
                if model_type is not None:
                    return [SqliteConnection._convert_to_model(i, model_type) for i in raw]
                else:
                    return [i for i in raw]
            else:
                if model_type is not None:
                    return SqliteConnection._convert_to_model(raw, model_type)
                else:
                    return raw


class Users(UsersTableAbstract, SqliteConnection):  # High level api
    @staticmethod
    def update_user(user_id, **kwargs):
        username = kwargs.get('username')
        first_name = kwargs.get('first_name')
        last_name = kwargs.get('last_name')
        request = ''
        # build request
        if username:
            request += f'`username` = "{username}",'
        if first_name:
            request += f'`first_name` = "{first_name}",'
        if last_name:
            request += f'`last_name` = "{last_name}",'
        request = request.rstrip(',')
        # UsersDb._make_request('UPDATE `Users` SET `username` = ? WHERE `user_id` = ?;',
        #                       params=(username, user_id,))

        Users._make_request(f'UPDATE `Users` SET {request} WHERE `user_id` = ?;',
                            params=(user_id,)
                            )

    @staticmethod
    def user_exists(user_id):
        result = Users._make_request('SELECT `id` FROM `Users` WHERE `user_id` = ?;',
                                     params=(user_id,),
                                     fetch=True, mult=True)
        return bool(len(result))

    @staticmethod
    def add_user(user_id, **kwargs) -> None:
        Users._make_request('INSERT INTO `Users` (`user_id`) VALUES (?);',
                            (user_id,))
        Users.update_user(user_id, **kwargs)

    @staticmethod
    def get_all_users():
        return [tuple(el) for el in Users._make_request('SELECT * from `Users`;', fetch=True,
                                                        mult=True)]


class Trainers(TrainersTableAbstract, SqliteConnection):
    @staticmethod
    def add_new_trainer(user_id, **kwargs) -> None:
        # if not AiDb.trainer_exists(user_id):
        Trainers._make_request('INSERT OR IGNORE INTO `Trainers` (`user_id`) VALUES (?);',
                               params=(user_id,))

    @staticmethod
    def trainer_exists(user_id) -> bool:
        result = Users._make_request('SELECT `id` FROM `Trainers` WHERE `user_id` = ?;',
                                     params=(user_id,),
                                     fetch=True, mult=True)
        return bool(len(result))


class Dataset(DatasetTableAbstract, SqliteConnection):

    @staticmethod
    def add_new_translation(user_id, *, input_, output, **kwargs) -> None:
        Dataset._make_request('INSERT INTO `Dataset` (`user_id`, `input`, `output`) VALUES (?, ?, ?)',
                              params=(user_id, input_, output))

    @staticmethod
    def get(model=list):
        """return full table"""
        for el in Dataset._make_request('SELECT * from `Dataset`;',
                                        fetch=True,
                                        mult=True):
            yield el
