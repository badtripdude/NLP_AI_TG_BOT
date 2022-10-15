import aiogram.contrib.fsm_storage.memory
import loguru

import states
from db import Dataset
from utils.ai_api import AI

local_storage = aiogram.contrib.fsm_storage.memory.MemoryStorage()


async def process_correct_answer(query: aiogram.types.CallbackQuery, state: aiogram.dispatcher.FSMContext):
    await query.bot.edit_message_text('пришли правильный ответ',
                                      chat_id=query.from_user.id,
                                      message_id=query.message.message_id)
    await states.TextProcessing.correct_answer.set()
    await local_storage.update_data(chat=query.message.chat.id,
                                    user=query.from_user.id,
                                    data={'bots_temp_msg_id': query.message.message_id,
                                          'user_input': query.message.text,
                                          },
                                    )


async def train_ai_model(message: aiogram.types.Message, state: aiogram.dispatcher.FSMContext, ai: AI):
    data = await local_storage.get_data(chat=message.chat.id,
                                        user=message.from_user.id,
                                        )
    await message.bot.edit_message_text(message.text,
                                        chat_id=message.chat.id,
                                        message_id=data.get('bots_temp_msg_id'),
                                        )
    await message.delete()
    input_ = data.get('user_input')
    output = message.text
    loguru.logger.debug(f"input={input_},"
                        f"output={output}")
    Dataset.add_new_translation(message.from_user.id,
                                input_=data,
                                output=output)

    await state.reset_state(with_data=False)
