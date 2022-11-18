import glob
import pickle
import numpy as np
import pandas as pd
import music21 as music
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, Bidirectional
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import random as rand
import os
from midi_proccessing import *




def lstm(network_input , network_output):
  model = Sequential()
  model.add(LSTM(128, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
  model.add(Dropout(0.2))
  model.add(LSTM(256, return_sequences=True))
  model.add(Dropout(0.2))
  model.add(LSTM(64, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=False))
  model.add(Dropout(0.2))
  model.add(Dense(4))
  model.add(Activation('softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
  model.fit(network_input, network_output, epochs=2)#, batch_size=64)
  return model

def predict_notes(network_input , model):
    #pick the first n notes from a random song generate the next note and add it to the prev notes to pull data from
    start_notes = np.random.randint(0, len(network_input)-1)
    #int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    start = 20
    num_prev_notes = 20
    prev_notes = network_input[start_notes]
    print("initial input for lstm predictions:  " , prev_notes)
    prediction_output = []


    for note_index in range(100): #here, we're generating 100 notes

        prediction_input = np.reshape(prev_notes[start - num_prev_notes:start], (1, 20, 4))
        prediction = model.predict(prediction_input, verbose = 0)
        prediction_output.append(prediction)
        prev_notes = np.concatenate((prev_notes , prediction) , axis = 0)
        start += 1




    #[gap from last note , note length , pitch]
    print(prediction_output)
    return prediction_output
files = get_midi_filepaths()
songs = []
for midi_path in files:
    songs.append(midi_path_to_data(midi_path))


input , output = prepare_song_data_for_model(songs , 20)
model = lstm(np.array(input) , np.array(output))
predict_notes(input , model)


