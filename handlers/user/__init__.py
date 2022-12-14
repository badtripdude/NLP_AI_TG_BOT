from aiogram import types, Dispatcher
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import CommandStart

import keyboards
import states
from db import Users


async def start(message: types.Message, state: FSMContext):
    await state.finish()
    kwargs = {'username': message.from_user.username,
              'first_name': message.from_user.first_name,
              'last_name': message.from_user.last_name, }
    if not Users.user_exists(user_id=message.chat.id):
        Users.add_user(message.from_user.id, **kwargs)
    Users.update_user(message.from_user.id, **kwargs)

    await message.answer('''
/train - включить режим тренировки- ты меня учишь как правильно отвечать/реагировать на что-либо,
/start - вызывет это сообщение вновь, можешь им пользоваться для отмены каких-либо действий
                        '''.strip())


def setup(dp: Dispatcher):
    dp.register_message_handler(start, CommandStart(),
                                state='*')
    from handlers.user.communication import process_answer, train_switcher_command
    from handlers.user.communication.train \
        import train_ai_model, process_correct_answer

    dp.register_message_handler(train_switcher_command,
                                commands=['train'])
    dp.register_callback_query_handler(process_correct_answer,
                                       keyboards.inline.callbacks.change_ans.filter(),
                                       )
    dp.register_message_handler(train_ai_model,
                                state=states.TextProcessing.correct_answer
                                )

    dp.register_message_handler(process_answer,
                                content_types=types.ContentType.TEXT,
                                )
