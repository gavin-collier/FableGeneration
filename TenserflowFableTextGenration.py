import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

import numpy as np
import os
import time

text = open('Fables.txt', 'r').read()

print(f'Length of text: {len(text)} characters')

vocab = sorted(set(text))
print(f'{len(vocab)} unique characters')

ids_from_chars = preprocessing.StringLookup(vocabulary=list(vocab))

chars = tf.strings.unicode_split(text, input_encoding='UTF-8')

ids = ids_from_chars(chars)

chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True)
chars = chars_from_ids(ids)
