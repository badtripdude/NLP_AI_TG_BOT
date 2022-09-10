from aiogram import types, Dispatcher
from aiogram.dispatcher import FSMContext
from utils import ai_api
import states


async def save_model(message: types.Message, state: FSMContext,
                     ai):
    await states.ControlAi.process_model_action.set()
    await state.update_data({
        'action': lambda msg: ai_api.export_model(ai, msg.text)
    })
    await message.answer('name...')


async def load_model(message: types.Message,
                     state: FSMContext, ai
                     ):
    await states.ControlAi.process_model_action.set()
    await state.update_data({
        'action': lambda msg: ai_api.import_model(ai, msg.text)
    })
    await message.answer('name...')


async def process_model_action(message: types.Message, state: FSMContext):
    data = await state.get_data()
    data.get('action')(message)
    await state.reset_state(False)

