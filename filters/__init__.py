from aiogram import Dispatcher
from .is_super_admin import SuperAdminFilter


def setup(dp: Dispatcher):
    dp.bind_filter(SuperAdminFilter)
