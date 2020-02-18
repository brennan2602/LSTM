#code taken and adapted from
# https://github.com/keras-team/keras/edit/master/examples/lstm_text_generation.py
# and
# https://www.tensorflow.org/tutorials/text/text_generation
from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
import io
import random


path = "encodedData.txt" #training data stored here

#defining vocabulary
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

#converting to vectors
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
model.compile(loss='categorical_crossentropy', optimizer="adam")


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate():
    # Function generates text.
    start_index = random.randint(0, len(text) - maxlen - 1) #this is used to get the seed string when generating
    diversity=1 #equivalent to temperature
    #generated = ''
    sentence = text[start_index: start_index + maxlen] #initial seed
    #generated += sentence
    genVals=""
    for i in range(15000):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]
        sentence = sentence[1:] + next_char
        genVals=genVals+next_char #updating sequence with new character
    #print(genVals)
    #print(generated)
    file1 = open("generated.txt", "a")
    file1.write(genVals)
    file1.close()

checkpoint_dir = './lstm_training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
tf.train.latest_checkpoint(checkpoint_dir)# getting latest checkpoint
print(tf.train.latest_checkpoint(checkpoint_dir))
model = build_model()
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))#loading model with latest checkpoint
model.build(tf.TensorShape([1, None]))
model.summary()
generate()

