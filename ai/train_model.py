import tensorflow as tf

from ai.layers import Encoder, Decoder, BahdanauAttention
from ai.utils import DecoderInput


class MaskedLoss(tf.keras.losses.Loss):
    def __init__(self):
        self.name = 'masked_loss'
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

    def __call__(self, y_true, y_pred):
        # Calculate the loss for each item in the batch.
        loss = self.loss(y_true, y_pred)

        # Mask off the losses on padding.
        mask = tf.cast(y_true != 0, tf.float32)
        loss *= mask

        # Return the total.
        return tf.reduce_sum(loss)


class TrainModel(tf.keras.Model):
    def __init__(self, embedding_dim, units,
                 input_text_processor,
                 output_text_processor,
                 use_tf_function=True):
        super().__init__()
        # Build the encoder and decoder
        encoder = Encoder(
            input_text_processor.vocabulary_size(),
            embedding_dim, units)
        decoder = Decoder(
            output_text_processor.vocabulary_size(),
            embedding_dim, units)

        self.encoder = encoder
        self.decoder = decoder
        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor
        self.use_tf_function = use_tf_function

    def train_step(self, inputs):
        if self.use_tf_function:
            return self._tf_train_step(inputs)
        else:
            return self._train_step(inputs)

    def _preprocess(self, input_text, target_text):
        # Convert the text to token IDs
        input_tokens = self.input_text_processor(input_text)
        target_tokens = self.output_text_processor(target_text)

        # Convert IDs to masks.
        input_mask = input_tokens != 0

        target_mask = target_tokens != 0

        return input_tokens, input_mask, target_tokens, target_mask

    def _train_step(self, inputs):
        input_text, target_text = inputs

        (input_tokens, input_mask,
         target_tokens, target_mask) = self._preprocess(input_text, target_text)

        max_target_length = tf.shape(target_tokens)[1]

        with tf.GradientTape() as tape:
            # Encode the input
            enc_output, enc_state = self.encoder(input_tokens)

            # Initialize the decoder's state to the encoder's final state.
            # This only works if the encoder and decoder have the same number of
            # units.
            dec_state = enc_state
            loss = tf.constant(0.0)

            for t in tf.range(max_target_length - 1):
                # Pass in two tokens from the target sequence:
                # 1. The current input to the decoder.
                # 2. The target for the decoder's next prediction.
                new_tokens = target_tokens[:, t:t + 2]
                step_loss, dec_state = self._loop_step(new_tokens, input_mask,
                                                       enc_output, dec_state)
                loss = loss + step_loss

            # Average the loss over all non padding tokens.
            average_loss = loss / tf.reduce_sum(tf.cast(target_mask, tf.float32))

        # Apply an optimization step
        variables = self.trainable_variables
        gradients = tape.gradient(average_loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        # Return a dict mapping metric names to current value
        return {'batch_loss': average_loss}

    @tf.function(input_signature=[[tf.TensorSpec(dtype=tf.string, shape=[None]),
                                   tf.TensorSpec(dtype=tf.string, shape=[None])]])
    def _tf_train_step(self, inputs):
        return self._train_step(inputs)

    # @tf.function(input_signature=[tf.data.Dataset])
    # def fit(self, dataset):
    #     return super().fit()

    def _loop_step(self, new_tokens, input_mask, enc_output, dec_state):
        input_token, target_token = new_tokens[:, 0:1], new_tokens[:, 1:2]

        # Run the decoder one step.
        decoder_input = DecoderInput(new_tokens=input_token,
                                     enc_output=enc_output,
                                     mask=input_mask)

        dec_result, dec_state = self.decoder(decoder_input, state=dec_state)

        # `self.loss` returns the total for non-padded tokens
        y = target_token
        y_pred = dec_result.logits
        step_loss = self.loss(y, y_pred)

        return step_loss, dec_state


def test():
    from callbacks import BatchLogs
    from preprocess import text_processor, build_dataset
    def load_data(path):
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            pairs = [line.split('\t') for line in lines]
            # return pairs
            inp = [inp for targ, inp, _ in pairs]
            targ = [targ for targ, inp, _ in pairs]
        return inp, targ
        # return ['привет', 'salyam'], ['hello', 'hi']

    # dataset = build_dataset(data := load_data('rus.txt'))
    dataset = build_dataset(data := (['gg'], ['lol']))

    # train

    embedding_dim = 1024
    units = 1024

    input_text_processor = text_processor()
    input_text_processor.adapt(data[0])

    output_text_processor = text_processor()
    output_text_processor.adapt(data[1])

    train_model = TrainModel(embedding_dim, units,
                             input_text_processor=input_text_processor,
                             output_text_processor=output_text_processor)
    train_model.compile(
        optimizer='adam',
        loss=MaskedLoss()
    )
    batch_loss = BatchLogs('batch_loss')
    train_model.fit(dataset, epochs=3,
                    callbacks=[batch_loss],
                    )


if __name__ == '__main__':
    test()
