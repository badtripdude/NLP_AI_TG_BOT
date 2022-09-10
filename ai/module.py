import numpy as np
import tensorflow as tf

from ai.utils import DecoderInput


class Translator(tf.Module):

    def __init__(self, encoder, decoder, input_text_processor,
                 output_text_processor):
        self.encoder = encoder
        self.decoder = decoder
        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor

        self.output_token_string_from_index = (
            tf.keras.layers.StringLookup(
                vocabulary=output_text_processor.get_vocabulary(),
                mask_token='',
                invert=True))

        # The output should never generate padding, unknown, or start.
        index_from_string = tf.keras.layers.StringLookup(
            vocabulary=output_text_processor.get_vocabulary(), mask_token='')
        token_mask_ids = index_from_string(['', '[UNK]', '[START]']).numpy()

        token_mask = np.zeros([index_from_string.vocabulary_size()], dtype=bool)
        token_mask[np.array(token_mask_ids)] = True
        self.token_mask = token_mask

        self.start_token = index_from_string(tf.constant('[START]'))
        self.end_token = index_from_string(tf.constant('[END]'))

    def tokens_to_text(self, result_tokens):
        result_text_tokens = self.output_token_string_from_index(result_tokens)

        result_text = tf.strings.reduce_join(result_text_tokens,
                                             axis=1, separator=' ')

        result_text = tf.strings.strip(result_text)
        return result_text

    def sample(self, logits, temperature):
        # 't' is usually 1 here.

        token_mask = self.token_mask[tf.newaxis, tf.newaxis, :]

        # Set the logits for all masked tokens to -inf, so they are never chosen.
        logits = tf.where(self.token_mask, -np.inf, logits)

        if temperature == 0.0:
            new_tokens = tf.argmax(logits, axis=-1)
        else:
            logits = tf.squeeze(logits, axis=1)
            new_tokens = tf.random.categorical(logits / temperature,
                                               num_samples=1)

        return new_tokens

    def translate(self,
                  input_text, *,
                  max_length=50,
                  return_attention=True,
                  temperature=1.0):
        if isinstance(input_text, str):  # TODO: remove this
            input_text = tf.constant([input_text])

        batch_size = tf.shape(input_text)[0]
        input_tokens = self.input_text_processor(input_text)
        enc_output, enc_state = self.encoder(input_tokens)

        dec_state = enc_state
        new_tokens = tf.fill([batch_size, 1], self.start_token)

        result_tokens = []
        attention = []
        done = tf.zeros([batch_size, 1], dtype=tf.bool)

        for _ in range(max_length):
            dec_input = DecoderInput(new_tokens=new_tokens,
                                     enc_output=enc_output,
                                     mask=(input_tokens != 0))

            dec_result, dec_state = self.decoder(dec_input, state=dec_state)

            attention.append(dec_result.attention_weights)

            new_tokens = self.sample(dec_result.logits, temperature)

            # If a sequence produces an `end_token`, set it `done`
            done = done | (new_tokens == self.end_token)
            # Once a sequence is done it only produces 0-padding.
            new_tokens = tf.where(done, tf.constant(0, dtype=tf.int64), new_tokens)

            # Collect the generated tokens
            result_tokens.append(new_tokens)

            if tf.executing_eagerly() and tf.reduce_all(done):
                break

        # Convert the list of generates token ids to a list of strings.
        result_tokens = tf.concat(result_tokens, axis=-1)
        result_text = self.tokens_to_text(result_tokens)

        if return_attention:
            attention_stack = tf.concat(attention, axis=1)
            return {'text': result_text, 'attention': attention_stack}
        else:
            return {'text': result_text}

    @tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None])])
    def tf_translate(self, input_text):
        return self.translate(input_text)


def train_and_export_model():
    # train
    from callbacks import BatchLogs
    from preprocess import text_processor, build_dataset
    from ai.train_model import TrainModel
    from ai.train_model import MaskedLoss
    def load_data(path):
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            pairs = [line.split('\t') for line in lines]
            # return pairs
            inp = [inp for targ, inp, _ in pairs]
            targ = [targ for targ, inp, _ in pairs]
        return inp, targ
        # return ['привет', 'salyam'], ['hello', 'hi']

    embedding_dim = 1024
    units = 1024

    # dataset_ = build_dataset(data_ := load_data('rus.txt'))
    # data =
    from db import Trainers
    res = Trainers._make_request('select input from Dataset', fetch=True, mult=True)
    input_ = list([str(list(el)[0]).lower() for el in res])
    res = Trainers._make_request('select output from Dataset', fetch=True, mult=True)
    output = list([str(list(el)[0]).lower() for el in res])
    # print(input_[-1])
    # print(output[-1].encode('utf-8'))
    # input_ = ['dasdf']
    # output = ['Родины!!! Сука фашист!!! ']
    # output = ['Р']
    # »
    # Иди нахуй агент сша пидораст гей транс!!!!!! Ты сгоришь в аду за предательство Родины!!! Сука фашист!!!
    data = (input_, output)
    dataset = build_dataset(data)
    # dataset = build_dataset(data := (['привет',
    #                                   # 'прив',
    #                                   # 'привеет'
    #                                   ],
    #                                  ['здарова',
    #                                   # 'салям',
    #                                   # 'ну привет'
    #                                   ]))

    # train
    # test_proc = text_processor()
    # test_proc.adapt(data_[0][:100])

    input_text_processor = text_processor()
    input_text_processor.adapt(data[0])

    output_text_processor = text_processor()
    output_text_processor.adapt(data[1])

    # dataset2 = build_dataset(data2 := (['пока',
    #                                    'пока',
    #                                    'пока'
    #                                    ],
    #                                   ['пока',
    #                                    'поки поки',
    #                                    'пока'
    #                                    ]))
    train_model = TrainModel(embedding_dim, units,
                             input_text_processor=input_text_processor,
                             output_text_processor=output_text_processor)

    train_model.compile(
        optimizer=tf.keras.optimizers.Adam(
            # learning_rate=1
        ),
        loss=MaskedLoss(),
    )
    batch_loss = BatchLogs('batch_loss')
    train_model.fit(dataset, epochs=22,
                    callbacks=[batch_loss],
                    )
    # train_model.input_text_processor.adapt(data[0])
    # train_model.output_text_processor.adapt(data[1])
    # train_model.fit(dataset2, epochs=3,
    #                 callbacks=[batch_loss],
    #                 )
    # test
    translator = Translator(
        encoder=train_model.encoder,
        decoder=train_model.decoder,
        input_text_processor=input_text_processor,
        output_text_processor=output_text_processor,
    )

    input_text = tf.constant([
        'пока',
        'привет',
        'как дела',
        'сука',
        'даун',
        'иди нах',
        'Сможешь достать языком до своего члена',
        'богато живу',
        'что делаешь'
    ])

    # while 1:
    #     input('enter')
    #     result = translator.tf_translate(
    #         input_text=input_text)
    #     for answer in result['text']:
    #         print('>>>' + answer.numpy().decode())
    #     print('---------------------------------------------')
    #     break

    # Export
    import pathlib
    p = pathlib.Path(__file__).parent.parent / 'data' / 'translator'
    tf.saved_model.save(translator, p.__str__(),
                        signatures={'serving_default': translator.tf_translate})
    # while 1:
    #     translator.tf_translate(tf.constant([input('text>>>')]))
    # import pickle
    # pickle.dump(translator, open(p, 'wb'))


def import_model():
    import pathlib
    p = pathlib.Path(__file__).parent.parent / 'data' / 'translator'
    # print(p)
    m = tf.saved_model.load(p.__str__())
    return m


if __name__ == '__main__':
    train_and_export_model()
    # import_model()
    # import pathlib
    # p = pathlib.Path(__file__).parent.parent / 'data' / 'translator'
    # print(p)
    # m = tf.saved_model.load(p)
    # input('enter')
    # result = m.tf_translate(tf.constant(['привет']))
    # print(result['text'][0].numpy().decode())
    ...
