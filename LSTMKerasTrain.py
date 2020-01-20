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

from tensorflow_core.python.keras.callbacks import LambdaCallback

path = "encodedData.txt"
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
def build_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(tf.keras.layers.Dense(len(chars), activation='softmax'))
    return model

model=build_model()
#optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer="adam")


# def sample(preds, temperature=1.0):
#     # helper function to sample an index from a probability array
#     preds = np.asarray(preds).astype('float64')
#     preds = np.log(preds) / temperature
#     exp_preds = np.exp(preds)
#     preds = exp_preds / np.sum(exp_preds)
#     probas = np.random.multinomial(1, preds, 1)
#     return np.argmax(probas)


# def on_epoch_end(epoch, _):
#     # Function invoked at end of each epoch. Prints generated text.
#     print()
#     print('----- Generating text after Epoch: %d' % epoch)
#
#     start_index = random.randint(0, len(text) - maxlen - 1)
#     #for diversity in [0.2, 0.5, 1.0, 1.2]:
#     diversity=1.0
#     print('----- diversity:', diversity)
#     generated = ''
#     sentence = text[start_index: start_index + maxlen]
#     generated += sentence
#     print('----- Generating with seed: "' + sentence + '"')
#     sys.stdout.write(generated)
#
#     for i in range(400):
#         x_pred = np.zeros((1, maxlen, len(chars)))
#         for t, char in enumerate(sentence):
#             x_pred[0, t, char_indices[char]] = 1.
#
#         preds = model.predict(x_pred, verbose=0)[0]
#         next_index = sample(preds, diversity)
#         next_char = indices_char[next_index]
#
#         sentence = sentence[1:] + next_char
#
#         sys.stdout.write(next_char)
#         sys.stdout.flush()
#     print()

#print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
checkpoint_dir = './lstm_training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)
history = model.fit(x, y,
          batch_size=128,
          epochs=25, callbacks=[checkpoint_callback])

tf.train.latest_checkpoint(checkpoint_dir)
model = build_model()
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
model.summary()
plt.plot(history.history["loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()
