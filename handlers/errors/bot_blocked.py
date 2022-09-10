from aiogram import types, exceptions
from loguru import logger


def blocked_by_user(upd: types.Update, err: exceptions.BotBlocked):
    logger.error(f"Bot blocked by user: {upd} [{err}")
