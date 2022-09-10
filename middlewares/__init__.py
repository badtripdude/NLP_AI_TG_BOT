from aiogram import Dispatcher

from .throttling import ThrottlingMiddleware
from .additionalkwargsdp import AdditionalKwargsDP


def setup(dp: Dispatcher):
    dp.middleware.setup(ThrottlingMiddleware(limit=.4))
    # dp.middleware.setup(Ai(dp['ai']))
    dp.middleware.setup(AdditionalKwargsDP(ai=dp['ai'],
                                           ))
