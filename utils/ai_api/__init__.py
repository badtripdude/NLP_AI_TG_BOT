__all__ = ('AI', 'export_model', 'import_model',
           'create_train_model', 'create_module',
           )

import abc
import pathlib
import tensorflow as tf

from ai import module, train_model
from ai.callbacks import BatchLogs
from ai.preprocess import text_processor, build_dataset


def export_model(ai, path: pathlib.Path):
    return ai.train_model.save_weights(path)


def import_model(ai, path: pathlib.Path = '.'):
    t_m = ai.train_model
    t_m.load_weights(path)
    return AI(t_m, create_module(t_m))


def create_train_model(embedding_dim, units,
                       input_text_processor,
                       output_text_processor, **kwargs):
    t_m = train_model.TrainModel(embedding_dim, units,
                                 input_text_processor=input_text_processor,
                                 output_text_processor=output_text_processor)
    t_m.compile(optimizer=kwargs.get('optimizer',
                                     tf.keras.optimizers.Adam()),
                loss=kwargs.get('loss', train_model.MaskedLoss()))

    return t_m


def create_module(train_m: train_model.TrainModel):
    return module.Translator(
        encoder=train_m.encoder,
        decoder=train_m.decoder,
        input_text_processor=train_m.input_text_processor,
        output_text_processor=train_m.output_text_processor,
    )


class AIBase(abc.ABC):  # FIXME: update
    ...


class AI(AIBase):
    """Main Model"""

    def __init__(self, train_m, module_=None):
        if module_ is None:
            module_ = module.Translator(
                encoder=train_m.encoder,
                decoder=train_m.decoder,
                input_text_processor=train_m.input_text_processor,
                output_text_processor=train_m.output_text_processor,
            )
        self.module = module_
        self.train_model = train_m

    def fit_model(self,
                  dataset, *, epochs=3, callbacks=None, **kwargs):
        if callbacks is None:
            callbacks = [BatchLogs('batch_loss')]
        self.train_model.fit(dataset, epochs=epochs,
                             callbacks=callbacks)

    def predict(self, input_: str):
        return self.module.tf_translate(input_text=tf.constant([input_]))['text'][0].numpy().decode()


def __test():
    from db import Trainers

    res = Trainers._make_request('select input from Dataset', fetch=True, mult=True)
    input_ = list([str(list(el)[0]).lower() for el in res])
    res = Trainers._make_request('select output from Dataset', fetch=True, mult=True)
    output = list([str(list(el)[0]).lower() for el in res])
    # d = list(Dataset.get())
    # input_ = [str(el['input']) for el in d]
    # output = [str(el['output']) for el in d]
    in_, out_ = text_processor(), text_processor()
    in_.adapt(input_)
    out_.adapt(output)
    dataset = build_dataset((input_, output))

    t_m = create_train_model(1024, 1024, input_text_processor=in_,
                             output_text_processor=out_)
    # tf.saved_model.save(t_m, '.')
    # t_m = tf.saved_model.load('.')
    ai = AI(t_m, create_module(t_m))
    ai.fit_model(dataset, epochs=10)
    print(ai.predict('привет'))
    # export_model(ai, '.')
    # ai = import_model(create_train_model(1024, 1024, input_text_processor=in_,
    #                                      output_text_processor=out_), '.')
    # t_m = create_train_model(1023, 1024, input_text_processor=in_,
    #                          output_text_processor=out_)
    # t_m.load_weights('.')
    ai.fit_model(dataset, epochs=15)
    ai.fit_model(build_dataset((['ратмир'], ['лох']), ), epochs=15)
    # ai = AI(t_m, create_module(t_m))
    for i in range(10):
        print()
        print()
        print(ai.predict('привет'))
        print(ai.predict('пока'))
        print(ai.predict('ратмир'))
        print(ai.predict('ты кто?'))


if __name__ == '__main__':
    __test()
