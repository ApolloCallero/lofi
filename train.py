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
import pickle
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
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dense(256))
    model.add(Dense(256))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dense(256))
    model.add(LSTM(512))
    model.add(Dense(4 , 'linear'))
    #model.add(Dense(4))
    #model.add(Activation('softmax'))
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='rmsprop', metrics=['accuracy'])
    model.fit(network_input, network_output,batch_size=64, epochs=8)
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


def lstm4(network_input , network_output):
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dense(256))
    model.add(Dense(256))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dense(256))
    model.add(LSTM(512 , return_sequences=False))
    model.add(Dense(4))
    #model.add(Dense(4))
    #model.add(Activation('softmax'))
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='rmsprop', metrics=['accuracy'])
    model.fit(network_input, network_output,batch_size=64, epochs=8)
    #pickle.dump(model, open('data/models/10_song_model.sav', 'wb'))
    return model
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

def predict_notes_seperate_models(pitch_input , pitch_model,gap_input,gap_model ,num_prev_notes):
    '''
    params:
        network_input:  list input that was sent to to the model
        model: trained lstm mode;
        num_prev_notes: int of the number of previous notes the model considered
    '''
    #pick the first n notes from a random song generate the next note and add it to the prev notes to pull data from
    start_note_index = np.random.randint(0, len(pitch_input)-1)
    #int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    start = num_prev_notes
    #prev_gaps = gap_input[start_note_index]
    prev_pitchs = pitch_input[start_note_index]
    predicted_pitchs = []

    pitch_to_num_dict ,  num_to_pitch = make_note_num_dicts(None)
    #generate 100 pitchs
    for note_index in range(100): 

        #get a num_pitch_long list with the likelyness each note is going to be played
        prediction_input = np.reshape(prev_pitchs[start - num_prev_notes:start], (1, num_prev_notes, 1))
        prediction = pitch_model.predict(prediction_input, verbose = 0)[0]

        #choose a pitch randomly from the top 3 note pitchs 
        top_likely = np.argpartition(prediction, -3)[-3:]
        index = np.random.choice(top_likely, 1)[0]
        predicted_pitchs.append(num_to_pitch[index /len(list(num_to_pitch.values()))])
        prev_pitchs = np.concatenate((prev_pitchs , [[index]]) , axis = 0)
        start += 1
    #generate 100 gaps 
    prev_gaps = gap_input[start_note_index]
    predicted_gaps = []
    start = num_prev_notes
    for note_index in range(100): #here, we're generating 100 notes
        prediction_input = np.reshape(prev_gaps[start - num_prev_notes:start], (1, num_prev_notes, 1))
        prediction = gap_model.predict(prediction_input, verbose = 0)
        predicted_gaps.append(prediction)
        prev_gaps = np.concatenate((prev_gaps , prediction) , axis = 0)
        start += 1
    print(predicted_pitchs , predicted_gaps )
    return prev_pitchs[-100:] , predicted_gaps[-100:]

def sepearate_models_train(input_data , output_data):
    '''
    params:
    input data: n by sequence_length by 4 list where each list is in the form of [gap,length,pitch,volume]
    Returns:
        gap_model: model trained on just the gaps between each note
        length_model: model trained on just the lengths of each note
        pitch_model: model trained on the pitchs of each note
    '''
    num_pitch = len(list(make_note_num_dicts(None)[0].values()))
    #get data split by each of the 3 features
    gap_input = []
    length_input = []
    pitch_input = []
    for seq in input_data:
        gap_seq = []
        length_seq = []
        pitch_seq = []
        for note in seq:
            gap_seq.append([note[0]])
            length_seq.append([note[1]])
            pitch_seq.append([note[2]])
        gap_input.append(gap_seq)
        length_input.append(length_seq)
        pitch_input.append(pitch_seq)

    gap_output = []
    length_output = []
    pitch_output = []
    for step in output_data:
        gap_output.append(step[0])
        length_output.append(step[1])
        pitch_output.append(step[2])
        

    #make the model to predict the gap between notes
    gap_input = np.array(gap_input)
    gap_output = np.array(gap_output)
    gap_model = Sequential()
    gap_model.add(LSTM(256, input_shape=(gap_input.shape[1], gap_input.shape[2])))
    gap_model.add(Dense(1))
    gap_model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='Adam' , metrics=['accuracy'])

    gap_model.fit(gap_input, gap_output, epochs=8, batch_size=64)
    
    #format then make then pitch model
    pitch_input = np.array(pitch_input)
    pitch_output = np.array(pitch_output)
    pitch_output = pitch_output * num_pitch
    pitch_output = np_utils.to_categorical(pitch_output , num_classes=num_pitch )
    pitch_model = Sequential()
    pitch_model.add(LSTM(512,return_sequences=True ))
    pitch_model.add(Dense(256))
    pitch_model.add(Dense(256))
    pitch_model.add(LSTM(512, return_sequences=True))
    pitch_model.add(Dense(256))
    pitch_model.add(LSTM(512 , return_sequences=False))
    pitch_model.add(Dense(num_pitch))
    pitch_model.add(Dense(num_pitch))
    pitch_model.add(Activation('softmax'))
    pitch_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    #gap_model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='rmsprop', metrics=['accuracy'])
    pitch_model.fit(pitch_input,pitch_output,batch_size=64, epochs=8)
    
    return pitch_input , pitch_model , gap_input , gap_model
    
    
def main():
    #get midi data
    files = get_midi_filepaths()
    songs = []
    for midi_path in files:
        instrument_data = midi_path_to_data(midi_path , split_instruments=False)
        if instrument_data !=  None:
            songs += instrument_data
            if len(songs) % 50 == 0:
                print(len(songs) , " proccessed")
            if len(songs) ==  10:
                break

    #format and normalize midi data
    input , output = prepare_song_data_for_model(songs , 10)
    print(np.array(input).shape)
    print(np.array(output).shape)
    model = lstm4(np.array(input) , np.array(output))
    ml_notes = predict_notes(input , model , 10)

    #unnormalize predictions and transform it to music
    songs = []
    for midi_path in files[0:50]:
        instrument_data = midi_path_to_data(midi_path , split_instruments=False)
        if instrument_data !=  None:
            songs += instrument_data
    predictions_to_music(ml_notes , songs)

def seperate_models_main():
    #get midi data
    files = get_midi_filepaths()
    songs = []
    for midi_path in files:
        instrument_data = midi_path_to_data(midi_path , split_instruments=False)
        if instrument_data !=  None:
            songs += instrument_data
            if len(songs) % 50 == 0:
                print(len(songs) , " proccessed")


    #format and normalize midi data
    input , output = prepare_song_data_for_model(songs , 12)
    pitch_input , pitch_model , gap_input , gap_model = sepearate_models_train(input , output)
    pred_pitchs , pred_gaps = predict_notes_seperate_models(pitch_input , pitch_model,gap_input , gap_model,12)

    #unnormalize predictions and transform it to music
    songs = []
    for midi_path in files[0:50]:
        instrument_data = midi_path_to_data(midi_path , split_instruments=False)
        if instrument_data !=  None:
            songs += instrument_data
    predictions_to_music_seperate_models(pred_pitchs , pred_gaps , songs)
#main()
seperate_models_main()