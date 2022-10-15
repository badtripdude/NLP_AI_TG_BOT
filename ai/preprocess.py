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



