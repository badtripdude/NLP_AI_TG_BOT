import tensorflow as tf

from ai import module, train_model, preprocess
from ai.callbacks import BatchLogs
from ai.preprocess import build_dataset

if __name__ == '__main__':
    from base import AI
else:
    from .base import AI


class AiModel(AI):
    def __init__(self, **kwargs):
        embedding_dim = kwargs.get('embedding_dim', 1024)
        units = kwargs.get('units', 1024)
        input_ = kwargs['input']
        output = kwargs['output']

        input_t_proc = preprocess.text_processor()
        output_t_proc = preprocess.text_processor()
        input_t_proc.adapt(input_)
        output_t_proc.adapt(output)

        self.train_model = self.create_train_model(embedding_dim,
                                                   units,
                                                   input_t_proc,
                                                   output_t_proc)
        self.create_module()

    def export_model(self, *args, **kwargs):
        self.train_model.save_weights(kwargs['path'])

    def create_module(self, **kwargs):
        self.module = module.Translator(
            encoder=self.train_model.encoder,
            decoder=self.train_model.decoder,
            input_text_processor=self.train_model.input_text_processor,
            output_text_processor=self.train_model.output_text_processor,
        )

    def create_train_model(self,
                           embedding_dim, units,
                           input_text_processor,
                           output_text_processor, **kwargs):
        t_m = train_model.TrainModel(embedding_dim, units,
                                     input_text_processor=input_text_processor,
                                     output_text_processor=output_text_processor)
        t_m.compile(optimizer=kwargs.get('optimizer',
                                         tf.keras.optimizers.Adam()),
                    loss=kwargs.get('loss', train_model.MaskedLoss()))

        return t_m

    @staticmethod
    def import_model(*args, **kwargs):
        a = AiModel(**kwargs)
        a.train_model.load_weights(kwargs['path'])
        return a

    def fit_model(self,
                  input_, output, *, epochs=3, callbacks=None, **kwargs):
        dataset = build_dataset((input_, output), batch_size=64)
        if callbacks is None:
            callbacks = [BatchLogs('batch_loss')]
        self.train_model.fit(dataset, epochs=epochs,
                             callbacks=callbacks)

    def predict_one(self, input_: str, **kwargs):
        return self.module.tf_translate(input_text=tf.constant([input_]))['text'][0].numpy().decode()


