#code taken and adapted from
# https://github.com/keras-team/keras/edit/master/examples/lstm_text_generation.py
# and
# https://www.tensorflow.org/tutorials/text/text_generation
from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
import io
import matplotlib.pyplot as plt

path = "encodedData.txt" #encoded training data stored under

#setting up vocabulary to train model on
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()
print('corpus length:', len(text))
chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
# I tried changing this with very little impact in results generated
maxlen = 40
step = 3
sentences = []
next_chars = []
#getting number of sequences
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

#converting into vector form
print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
# I tried experimenting with a larger model and multiple LSTM layers but saw little impact on results
def build_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(tf.keras.layers.Dense(len(chars), activation='softmax'))
    return model

model=build_model()
model.compile(loss='categorical_crossentropy', optimizer="adam")
checkpoint_dir = './lstm_training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

#callback to save model checkpoints
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)
#fitting model and saving checkpoints
history = model.fit(x, y,
          batch_size=128,
          epochs=2, callbacks=[checkpoint_callback])

# tf.train.latest_checkpoint(checkpoint_dir)
# model = build_model()
# model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
# model.build(tf.TensorShape([1, None]))
# model.summary()

#plotting
plt.plot(history.history["loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()
