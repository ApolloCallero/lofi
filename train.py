import glob
import pickle
import numpy as np
import pandas as pd
import music21 as music
# MLP for Pima Indians Dataset Serialize to JSON and HDF5
from keras.models import Sequential , model_from_json
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
from tensorflow import keras
from keras import layers
NUM_PREV_NOTES = 10

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

#def lstsm_final():
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

def predict_notes_seperate_models(pitch_input , pitch_model,gap_input,gap_model , length_input, length_model ,num_prev_notes , songs):
    '''
    params:
        network_input:  list input that was sent to to the model
        model: trained lstm mode;
        num_prev_notes: int of the number of previous notes the model considered
    '''
    gap_to_num , num_to_gap , length_to_num , num_to_length , pitch_to_num , num_to_pitch , volume_to_num , num_to_volume = get_dicts(songs , num_prev_notes)
    #pick the first n notes from a random song generate the next note and add it to the prev notes to pull data from
    start_note_index = np.random.randint(0, len(pitch_input)-1)
    #int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    start = num_prev_notes
    #prev_gaps = gap_input[start_note_index]
    prev_pitchs = pitch_input[start_note_index]
    predicted_pitchs = []

    #get unique gaps

    #pitch_to_num_dict ,  num_to_pitch , gap_to_num , num_to_gap = make_note_num_dicts(None , unique_gaps)
    #generate 100 pitchs
    print('dict before: ' , num_to_pitch)
    for note_index in range(100): 

        #get a num_pitch_long list with the likelyness each note is going to be played
        prediction_input =  np.array([[list(prev_pitchs[0][start - num_prev_notes:start])]])
        prediction_input = tf.cast(prediction_input , tf.float32 )
        prediction_input /= tf.cast(len(list(num_to_pitch.values())) , tf.float32 )
        prediction = pitch_model.predict(prediction_input, verbose = 0)[0]
        
        #choose a pitch randomly from the top 3 note pitchs 
        top_likely = np.argpartition(prediction, -1)[-1:]
        #if [[np.argpartition(prediction, -1)[-1:]]] != 
        index = np.random.choice(top_likely, 1)[0]
        predicted_pitchs.append(num_to_pitch[int(index)])
        prev_pitchs = np.column_stack((prev_pitchs , [[index]]))
        start += 1
    #generate 100 gaps 
    prev_gaps = gap_input[start_note_index]
    predicted_gaps = []
    start = num_prev_notes
    for note_index in range(100): #here, we're generating 100 notes
        prediction_input = np.array([[list(prev_gaps[0][start - num_prev_notes:start])]])
        prediction_input = tf.cast(prediction_input , tf.float32 )
        prediction_input /= tf.cast(len(list(num_to_gap.values())) , tf.float32 )
        predictions = gap_model.predict(prediction_input, verbose = 0)[0]
        top_likely = np.argpartition(predictions, -1)[-1:]
        index = np.random.choice(top_likely, 1)[0]
        predicted_gaps.append(num_to_gap[index])
        prev_gaps= np.column_stack((prev_gaps , [[index]]))
        start += 1

    #generate 100 note lengths
    prev_lengths = length_input[start_note_index]
    predicted_lengths = []
    start = num_prev_notes
    
    for note_index in range(100): #here, we're generating 100 notes
        prediction_input = np.array([[list(prev_lengths[0][start - num_prev_notes:start])]])
        predictions = length_model.predict(prediction_input, verbose = 0)[0]
        top_predictions = np.argpartition(predictions, -1)[-1:][0]
        print(top_predictions)
        index = np.random.choice(top_likely, 1)[0]
        print(top_likely)
        print('length out:' ,top_likely)
        predicted_lengths.append(num_to_length[index])
        prev_lengths = np.column_stack((prev_lengths , [[index]]))
        start += 1
    
    print(predicted_gaps , predicted_pitchs , predicted_lengths)
    return predicted_pitchs[-100:] , predicted_gaps[-100:]

def sepearate_models_train(pitch_input,pitch_output , gap_input,gap_output , length_input , length_output , vol_input , vol_output , songs):
    '''
    params:
    input data: n by sequence_length by 4 list where each list is in the form of [gap,length,pitch,volume]
    Returns:
        gap_model: model trained on just the gaps between each note
        length_model: model trained on just the lengths of each note
        pitch_model: model trained on the pitchs of each note
    '''


    #make the model to predict the gap between notes
    gap_to_num , num_to_gap , length_to_num , num_to_length , pitch_to_num , num_to_pitch , volume_to_num , num_to_volume = get_dicts(songs , NUM_PREV_NOTES)
    print('in train: ' , num_to_pitch)
    num_unique_gaps = len(list(gap_to_num.values()))
    #pitch_to_num_dict ,  num_to_pitch , num_to_gap , gap_to_num = make_note_num_dicts(None , set([round(i,3) for i in gap_output]))
    #num_unique_gaps = len(num_to_gap.values())
    gap_input = np.array(gap_input)
    gap_input /= tf.cast(num_unique_gaps , tf.float64)
    gap_output = np.array(gap_output)
    gap_output = np_utils.to_categorical(gap_output , num_classes=num_unique_gaps )
    gap_model = Sequential()
    gap_model.add(LSTM(256, input_shape=(gap_input.shape[1], gap_input.shape[2]), return_sequences=True) )
    gap_model.add(Dense(128))
    gap_model.add(Dense(256))
    gap_model.add(LSTM(256, input_shape=(gap_input.shape[1], gap_input.shape[2]), return_sequences=False) )
    gap_model.add(Dense(num_unique_gaps))
    gap_model.add(Dense(num_unique_gaps))
    gap_model.add(Activation('softmax'))
    gap_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    gap_model.fit(gap_input, gap_output, epochs=8, batch_size=64)
    #format then make then pitch model
    pitch_input = np.array(pitch_input)
    pitch_output = np.array(pitch_output)
    num_unique_pitchs= len(list(pitch_to_num.values()))
    pitch_input = tf.cast(pitch_input , tf.float32 )
    pitch_input /= tf.cast(num_unique_pitchs , tf.float32)
    print('pitch in:',pitch_input)
    pitch_output = np_utils.to_categorical(pitch_output , num_classes=num_unique_pitchs )
    print('pitch out:',pitch_output)
    pitch_model = Sequential()
    pitch_model.add(LSTM(512,return_sequences=True ))
    pitch_model.add(Dense(128))
    pitch_model.add(Dense(256))
    pitch_model.add(LSTM(512, return_sequences=True ))
    pitch_model.add(Dense(200))
    pitch_model.add(LSTM(512 , return_sequences=False))
    pitch_model.add(Dense(num_unique_pitchs))
    pitch_model.add(Dense(num_unique_pitchs))
    pitch_model.add(Activation('softmax'))
    pitch_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    #gap_model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='rmsprop', metrics=['accuracy'])
    pitch_model.fit(pitch_input,pitch_output,batch_size=64, epochs=8)

    num_lengths = len(list(length_to_num.values()))
    length_input = np.array(length_input)
    length_input = tf.cast(length_input , tf.float64 )
    length_input /= tf.cast(num_lengths , tf.float64)
    length_output = np.array(length_output)
    #length_output = length_output* np.unique(length_output)
    length_output = np_utils.to_categorical(length_output , num_classes=num_lengths)
    length_model = Sequential()
    length_model.add(LSTM(512,return_sequences=True ))
    length_model.add(Dense(128))
    length_model.add(Dense(256))
    length_model.add(LSTM(512, return_sequences=True ))
    length_model.add(Dense(200))
    length_model.add(LSTM(512 , return_sequences=False))
    length_model.add(Dense(num_lengths))
    length_model.add(Dense(num_lengths))
    length_model.add(Activation('softmax'))
    length_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    #gap_model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='rmsprop', metrics=['accuracy'])
    length_model.fit(length_input,length_output,batch_size=64, epochs=8)

    

    return pitch_model , gap_model , length_model
    

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
        notes = [len(i['pitch']) for i in songs]
        if len(songs) == 103:
            break
    
    #format and normalize midi data
    input , output = prepare_song_data_for_model(songs , NUM_PREV_NOTES )
    gap_x , length_x , pitch_x , vol_x , gap_y , length_y , pitch_y , vol_y = split_features(input , output)
    n_pitch = len(np.unique(pitch_y))



    print('input shape: ',np.array(input).shape)
    print('output shape: ',np.array(output).shape)

    songs = []
    for midi_path in files:
        instrument_data = midi_path_to_data(midi_path , split_instruments=False)
        if instrument_data !=  None:
            songs += instrument_data
        if len(songs) == 103:
            break
    pitch_model , gap_model , length_model= sepearate_models_train(pitch_x,pitch_y ,gap_x,gap_y , length_x , length_y , vol_x , vol_y , songs)
    songs = []
    for midi_path in files:
        instrument_data = midi_path_to_data(midi_path , split_instruments=False)
        if instrument_data !=  None:
            songs += instrument_data
        if len(songs) == 103:
            break
    pred_pitchs , pred_gaps = predict_notes_seperate_models(pitch_x , pitch_model,gap_x , gap_model, length_x , length_model, NUM_PREV_NOTES , songs )

    #unnormalize predictions and transform it to music
    songs = []
    for midi_path in files:
        instrument_data = midi_path_to_data(midi_path , split_instruments=False)
        if instrument_data !=  None:
            songs += instrument_data
        if len(songs) == 103:
            break      
    predictions_to_music_seperate_models(pred_pitchs , pred_gaps , songs)
#main()
seperate_models_main()