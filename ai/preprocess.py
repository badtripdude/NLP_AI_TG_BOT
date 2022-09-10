import tensorflow as tf
import typing

from ai.utils import tf_lower_and_split_punct


def build_dataset(data: typing.Tuple[typing.List[str], typing.List[str]]
                  ,
                  batch_size=64) -> tf.data.Dataset:
    inp, targ = data
    buffer_size = len(inp)

    dataset = tf.data.Dataset.from_tensor_slices(data) \
        .shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    return dataset


def text_processor(max_tokens=5000):
    return tf.keras.layers.TextVectorization(
        standardize=tf_lower_and_split_punct,
        max_tokens=max_tokens)


if __name__ == '__main__':
    def load_data(path):
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            pairs = [line.split('\t') for line in lines]
            # return pairs
            inp = [inp for targ, inp, _ in pairs]
            targ = [targ for targ, inp, _ in pairs]
        return inp, targ
        # return ['привет', 'salyam'], ['hello', 'hi']


    dataset_ = build_dataset(load_data('rus.txt'))
    # dataset_ = build_dataset(('gg', 'gg'))

    for example_input_batch, example_target_batch in dataset_.take(1):
        print(example_input_batch[:5])
        print()
        print(example_target_batch[:5])
        break
