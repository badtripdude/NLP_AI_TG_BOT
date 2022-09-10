from aiogram.dispatcher.filters import BoundFilter
from aiogram import types

import settings


class SuperAdminFilter(BoundFilter):
    key = 'is_super_admin'

    def __init__(self, is_super_admin):
        self.is_super_admin = is_super_admin

    async def check(self, message: types.Message) -> bool:
        if str(message.from_user.id) in settings.app_config['BOT']['SUPER_ADMINS']:
            return True
        return False
