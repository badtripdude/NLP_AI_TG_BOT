import aiogram
from .bot_blocked import blocked_by_user


def setup(dp: aiogram.Dispatcher):
    dp.register_errors_handler(blocked_by_user,
                               exception=aiogram.exceptions.BotBlocked)
