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
from genre_selector import *


def lstm(network_input , network_output):
    print('input shap: ' , network_input.shape)
    print('output shape: ' , network_output.shape)
    model = Sequential()
    model.add(LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=False
    ,kernel_regularizer=tf.keras.regularizers.L1(0.01) , activity_regularizer=tf.keras.regularizers.L2(0.01)))
    model.add(Dropout(.2))
    model.add(Dense( 4, activation ='linear'))
    model.add(Activation('softmax'))
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='rmsprop')
    model.fit(network_input, network_output, epochs=40)
    return model

def lstm2(network_input , network_output):
    #bad rn
    model = Sequential()
    model.add(LSTM(
        256,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(4 , 'linear'))
    model.add(Activation('softmax'))
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='rmsprop')
    model.fit(network_input, network_output, epochs=8 , batch_size=64)
    return model
def lstm3(network_input, network_output):
    model1 = Sequential()
    model1.add(tf.keras.layers.InputLayer((8, 4)))
    model1.add(LSTM(256 , return_sequences=False ))
    model1.add(Dense(4 , 'linear'))
    model1.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='rmsprop', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    model1.fit(network_input, network_output, epochs=40)
    #print(model1.summuary())
    return model1
def predict_notes(network_input , model , num_prev_notes):
    '''
    params:
        network_input:  list input that was sent to to the model
        model: trained lstm mode;
        num_prev_notes: int of the number of previous notes the model considered
    '''
    #pick the first n notes from a random song generate the next note and add it to the prev notes to pull data from
    start_notes = np.random.randint(0, len(network_input)-1)
    #int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    start = num_prev_notes
    prev_notes = network_input[start_notes]
    #print("initial input for lstm predictions:  " , prev_notes)
    prediction_output = []


    for note_index in range(100): #here, we're generating 100 notes

        prediction_input = np.reshape(prev_notes[start - num_prev_notes:start], (1, num_prev_notes, 4))
        prediction = model.predict(prediction_input, verbose = 0)
        prediction_output.append(prediction)
        prev_notes = np.concatenate((prev_notes , prediction) , axis = 0)
        start += 1




    #[gap from last note , note length , pitch , volume]
    #for note in prediction_output:
        #print(note)
    return prediction_output

#get midi data
files = get_midi_filepaths()
print(len(files) , 'number of songs shown to model')
songs = []
for midi_path in files:
    instrument_data = midi_path_to_data(midi_path)
    if instrument_data !=  None:
        songs += midi_path_to_data(midi_path)

#format and normalize midi data
input , output = prepare_song_data_for_model(songs , 8)
model = lstm3(np.array(input) , np.array(output))
ml_notes = predict_notes(input , model , 8)

songs = []
for midi_path in files:
    instrument_data = midi_path_to_data(midi_path)
    if instrument_data !=  None:
        songs += midi_path_to_data(midi_path)
predictions_to_music(ml_notes , songs)

