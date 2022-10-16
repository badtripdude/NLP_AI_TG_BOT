from aiogram import types
from aiogram.dispatcher import FSMContext

import settings
import states


async def save_model(message: types.Message, state: FSMContext,
                     ai):
    await states.ControlAi.process_model_action.set()
    await state.update_data({
        'action': lambda msg: ai.export_model(path=settings.AI_MODELS_DIR / msg.text)
    })
    await message.answer('name...')


async def load_model(message: types.Message,
                     state: FSMContext, ai
                     ):
    await states.ControlAi.process_model_action.set()
    await state.update_data({
        'action': lambda msg: ai.import_model(settings.AI_MODELS_DIR / msg.text)
    })
    await message.answer('name...')


async def process_model_action(message: types.Message, state: FSMContext):
    data = await state.get_data()
    data.get('action')(message)
    await state.reset_state(False)
