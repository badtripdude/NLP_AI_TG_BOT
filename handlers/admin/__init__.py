from aiogram import Dispatcher

import states
from handlers.admin.control_ai import load_model, save_model, process_model_action


def setup(dp: Dispatcher):
    dp.register_message_handler(save_model, is_super_admin=True, commands=['save_model'])
    dp.register_message_handler(load_model, is_super_admin=True, commands=['load_model'])
    dp.register_message_handler(process_model_action, is_super_admin=True,
                                state=states.ControlAi.process_model_action)
    # dp.register_message_handler(command_broadcast, commands=['broadcast'])
    # dp.register_message_handler(process_broadcast, state=states.BroadCast.process_broadcast, )
