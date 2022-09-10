from aiogram.dispatcher.filters.state import StatesGroup, State


class TextProcessing(StatesGroup):
    correct_answer = State()


class ControlAi(StatesGroup):
    process_model_action = State()
