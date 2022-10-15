import asyncio

import settings
from aiogram import executor
from aiogram import Bot, Dispatcher
from aiogram.contrib.fsm_storage.memory import MemoryStorage
import loguru


def setup_ai_model(dp_: Dispatcher):
    loguru.logger.info('Setup Ai module')
    from utils.ai_api import AiModel, AI
    from db import Trainers

    res = Trainers._make_request('select input from Dataset', fetch=True, mult=True)
    inputs = list([str(list(el)[0]).lower() for el in res])
    res = Trainers._make_request('select output from Dataset', fetch=True, mult=True)
    outputs = list([str(list(el)[0]).lower() for el in res])

    p = settings.MODEL_NAME.as_posix() + '/'
    try:  # FIXME
        dp['ai'] = AiModel.import_model(
            path=p,
            input=inputs,
            output=outputs
        )
        loguru.logger.success('Loaded AiModel({})', p)
    except Exception as e:
        loguru.logger.warning(
            f"Couldn't import model with name {settings.app_config['AI']['MODEL_NAME']}\nwith err msg: {e}")

        a: AI = AiModel(input=inputs,
                        output=outputs)
        a.fit_model(input_=inputs,
                    output=outputs,
                    epochs=23, )
        a.export_model(path=p)
        loguru.logger.success(f'Create Saved Fitted AiModel')
        dp['ai'] = a


async def on_startup(dp_: Dispatcher):
    import handlers
    import middlewares
    import filters
    setup_ai_model(dp)

    filters.setup(dp)
    middlewares.setup(dp_)
    handlers.setup(dp_)


async def on_shutdown(dp_: Dispatcher): ...


if __name__ == '__main__':
    loguru.logger.info('App started')
    try:
        bot_config = settings.app_config['BOT']
        bot = Bot(token=bot_config['API_TOKEN'])
        dp = Dispatcher(bot,
                        storage=MemoryStorage(),
                        run_tasks_by_default=False,
                        throttling_rate_limit=bot_config['THROTTLING_RATE_LIMIT'],
                        no_throttle_error=False,
                        filters_factory=None)
        loop = asyncio.get_event_loop()
        executor.start_polling(dp, on_startup=on_startup,
                               on_shutdown=on_shutdown,
                               loop=loop,
                               )
    except KeyboardInterrupt:
        ...
    loguru.logger.info('app closed!')
