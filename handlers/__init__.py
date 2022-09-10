from aiogram import Dispatcher


def setup(dp: Dispatcher):
    from . import admin, user, errors
    admin.setup(dp)
    user.setup(dp)
    errors.setup(dp)
