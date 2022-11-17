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



files = get_midi_filepaths()
songs = []
for midi_path in files:
    songs.append(midi_path_to_data(midi_path))
print(len(songs))
input , output = prepare_song_data_for_model(songs , 20)