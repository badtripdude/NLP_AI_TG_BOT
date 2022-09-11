from aiogram import types

from .consts import InlineConstructor
from aiogram.utils.callback_data import CallbackData
from .callbacks import change_ans
import random


class Users(InlineConstructor):

    @staticmethod
    def change_answer(msg: types.Message):
        phrases = ['исправить ответ']
        schema = [1]

        actions = [
            {'text': random.choice(phrases),
             'callback_data': (d := {
                 'msg_id': msg.message_id,
                 # 'user_id': msg.from_user.id
             }, change_ans)
             }
        ]
        return Users._create_kb(actions, schema)
