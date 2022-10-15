import aiogram.types
import loguru

import keyboards
import states
from db import Trainers, Dataset
import aiogram.contrib.fsm_storage.memory

local_storage = aiogram.contrib.fsm_storage.memory.MemoryStorage()


def generate_answer(message: aiogram.types.Message, ai):
    # TODO:
    loguru.logger.trace('generating answer')
    return {'text': ai.predict_one(message.text)}


def update_trainers(message: aiogram.types.Message):
    if not Trainers.trainer_exists(message.chat.id):
        Trainers.add_new_trainer(message.chat.id, )


async def train_switcher_command(message: aiogram.types.Message, state: aiogram.dispatcher.FSMContext):
    data = await state.get_data({})
    is_training = data.get('is_training', False)

    if is_training:
        await state.update_data({'is_training': False})
        await message.answer('сейчас ты выключил режим тренировки.')
        return
    await state.update_data({'is_training': True})
    await message.answer('сейчас ты включил режим тренировки.')


async def generate_train_callback(message: aiogram.types.Message, state: aiogram.dispatcher.FSMContext):
    data = await state.get_data()

    rmk = None
    if data.get('is_training'):
        rmk = keyboards.inline.Users.change_answer(message)
        update_trainers(message)

    return {'reply_markup': rmk}


async def update_prev_message(message: aiogram.types.Message):
    context = await local_storage.get_data(chat=message.chat.id)
    prev_message_id = context.get('prev_message_id')
    if prev_message_id:
        try:
            await message.bot.edit_message_reply_markup(message.chat.id,
                                                        prev_message_id,
                                                        reply_markup=None)
        except (aiogram.exceptions.MessageNotModified, aiogram.exceptions.MessageCantBeEdited) as e:
            loguru.logger.trace('Cant edit message with error:'+str(e))
            context.pop('prev_message_id')
        # loguru.logger.debug(f'input = {message.text}\n output = {}')
    if message.reply_markup is not None:
        context.update({'prev_message_id': message.message_id})
    await local_storage.update_data(chat=message.chat.id,
                                    data=context)


async def process_answer(message: aiogram.types.Message, state: aiogram.dispatcher.FSMContext, ai):
    msg_kwargs = {}

    msg_kwargs.update({
        **await generate_train_callback(message, state=state),
        **generate_answer(message, ai)
    })
    message_sent = await message.answer(**msg_kwargs)
    await update_prev_message(message_sent)
