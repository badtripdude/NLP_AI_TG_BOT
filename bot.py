import asyncio
import tensorflow.python.framework.errors_impl

import settings
from aiogram import executor
from aiogram import Bot, Dispatcher
from aiogram.contrib.fsm_storage.memory import MemoryStorage
import loguru


def setup_ai_model(dp_: Dispatcher):
    loguru.logger.info('Setup Ai module')
    dp['ai'] = None
    # from utils.ai_api import AI, import_model, create_train_model, export_model
    # from ai import preprocess
    # from db import Dataset, get_table
    # from ai.preprocess import build_dataset
    # d = get_table('Dataset', model=dict)
    #
    # input, output = [], []
    # for el in d:
    #     input.append(str(el['input']))
    #     output.append(str(el['output']))
    # # setup dictionary of bot
    # input_t_proc = preprocess.text_processor()
    # output_t_proc = preprocess.text_processor()
    # input_t_proc.adapt(input)
    # output_t_proc.adapt(output)
    # # create model
    # t_m = create_train_model(1024, 1024, input_text_processor=input_t_proc,
    #                          output_text_processor=output_t_proc)
    #
    # ai = AI(train_m=t_m)
    # try:
    #     import_model(ai, 'data/ai_models/train_models/best/')
    #     loguru.logger.info('Loaded best model')
    # except tensorflow.errors.OpError:
    #     loguru.logger.info('Best model not found')
    #     loguru.logger.info('Fit new model')
    #     ai.fit_model(build_dataset((input, output)),
    #                  epochs=23)
    #     export_model(ai, 'data/ai_models/train_models/best/')
    # dp_['ai'] = ai


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