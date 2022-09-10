from aiogram.dispatcher.middlewares import LifetimeControllerMiddleware


class AdditionalKwargsDP(LifetimeControllerMiddleware):
    skip_patterns = ["error.log", "update"]

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        # self.ai = ai

    async def pre_process(self, obj, data, *args):
        for k, v in self.kwargs.items():
            data[k] = v
        # data["ai"] = self.ai

    async def post_process(self, obj, data, *args):
        # del data["ai"]
        for k in self.kwargs.keys():
            del data[k]