import aiogram.utils.exceptions
import loguru
from aiogram import types, Dispatcher
from aiogram.types import CallbackQuery
import states
import keyboards
from aiogram.dispatcher import FSMContext

from db import Trainers, Dataset


async def train_switcher(message: types.Message, state: FSMContext):
    Trainers.add_new_trainer(message.from_user.id, )
    data = await state.get_data()
    if data.get('train', None):
        await state.update_data({'train': False})
        await message.answer('ты выключил режим тренировки')
        return
    await message.answer('сейчас ты включил режим тренировки')
    await state.update_data({'train': True})


async def user_is_training(user: types.User, data):
    if data.get('train', False):
        reply_markup = keyboards.inline.Users.change_answer(message)
        if last_answer := data.get('lanswer'):
            if last_answer.reply_markup:
                try:
                    await message.bot.edit_message_text(last_answer.text + '.', message.chat.id,
                                                        last_answer.message_id)
                except (aiogram.utils.exceptions.MessageCantBeEdited,
                        aiogram.utils.exceptions.MessageNotModified) as e:
                    loguru.logger.error(e)


async def generate_answer(message: types.Message, ai, state: FSMContext):
    data = await state.get_data(default={})
    reply_markup = None

    # check if user is training
    # if user_is_training(message.from_user):

    # await state.reset_data()
    answer = await message.answer(''
                                  # + translator.tf_translate(tf.constant([message.text.__str__().lower()]))['text'][
                                  #     0].numpy().decode() + '.',
                                  + ai.predict(message.text.__str__().lower()),
                                  # + '*меня еще нах не обучали, так что я су ка набираю базу, гандон бля',
                                  # +'.',
                                  reply_markup=reply_markup)

    # set state_data
    await state.update_data({'lanswer': answer,
                             'user_question': message})
    # log
    loguru.logger.info(
        f'user({message.from_user.username}:{message.from_user.id}) asked `{message.text}`, bot answered `{answer.text}`')


async def process_correct_answer(query: CallbackQuery, state: FSMContext):
    msg_id_for_change = query.message.message_id
    await query.bot.edit_message_text('покажи как нужно, я постараюсь исправиться в будущем...',
                                      chat_id=query.from_user.id,
                                      message_id=msg_id_for_change)
    await states.TextProcessing.correct_answer.set()
    await state.update_data({'bots_message_to_correct': msg_id_for_change})


async def receive_correct_answer_from_user(message: types.Message, ai, state: FSMContext):
    data = await state.get_data()
    if data is None:
        return await state.finish()
    # change bot's msg to correct
    msg_id_to_correct = (await state.get_data()).get('bots_message_to_correct', None)
    if msg_id_to_correct:
        try:
            await message.bot.edit_message_text(message.text,
                                                chat_id=message.chat.id,
                                                message_id=msg_id_to_correct)
        except (aiogram.exceptions.MessageNotModified,) as e:
            # await message.answer('')
            ...
        await message.bot.delete_message(message.chat.id, message.message_id)

    # save new data
    user_msg = data.get('user_question')
    if user_msg:
        input_ = data['user_question'].text
        output = message.text
        Dataset.add_new_translation(input_=input_, output=output,
                                    user_id=message.from_user.id, )
        loguru.logger.info(
            f'added new translation ({input_}:{output}) from user {message.from_user.username}:{message.from_user.id}')

    # reset some data
    await state.update_data({
        'bots_message_to_correct': None,
        'lanswer': None,
        'user_question': None,
    })
    # reset state
    await state.reset_state(with_data=False)
