import aiogram
import aiogram.contrib.fsm_storage.memory
import loguru

import states

local_storage = aiogram.contrib.fsm_storage.memory.MemoryStorage()


class CorrectAnswerProcessor:
    def __init__(self, query: aiogram.types.CallbackQuery,
                 state: aiogram.dispatcher.FSMContext,
                 loop=None):
        self.query = query
        self.loop = loop
        if self.loop:
            import asyncio
            self.loop = asyncio.get_event_loop()

        self.loop.create_task(self.process())

    async def process(self, ):
        query = self.query
        await query.bot.edit_message_text('пришли правильный ответ',
                                          chat_id=query.from_user.id,
                                          message_id=query.message.message_id)
        await local_storage.update_data(chat=query.message.chat.id,
                                        user=query.message.from_user.id,
                                        data={'bots_temp_msg_id'
                                              'user_input': query.message.text,
                                              })
        await states.TextProcessing.correct_answer.set()

    async def train_ai_model(self, message: aiogram.types.Message):
        ...


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


async def train_ai_model(message: aiogram.types.Message, state: aiogram.dispatcher.FSMContext):
    data = await local_storage.get_data(chat=message.chat.id,
                                        user=message.from_user.id,
                                        )
    await message.bot.edit_message_text(message.text,
                                        chat_id=message.chat.id,
                                        message_id=data.get('bots_temp_msg_id'),
                                        )
    await message.delete()
    loguru.logger.debug(f"input={data.get('user_input')},"
                        f"output={message.text}")
    await state.reset_state(with_data=False)
