import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from midi_proccessing import *
from genre_selector import *
import seaborn as sns
def plot_song_data_over_time(song_data):
    '''
    Desc:
        plots a songs pitchs, gaps,lengths and volume in 4 sub plots
    params:
        song_data: Dictionairy in the form of {gaps:[....],lengths:[...],pitch:[...],volume:[...]}
    '''
    gap_y = []
    length_y = []
    pitch_y = []
    volume_y = []
    for index in range(0,len(song_data['pitch'])):
        gap_y.append(song_data['gaps'][index])
        length_y.append(song_data['lengths'][index])
        pitch_y.append(song_data['pitch'][index])
        volume_y.append(song_data['volume'][index])

    time = []
    for index in range(0,len(song_data['pitch'])):
        if time == []:
            time.append(song_data['gaps'][0])
        else:
            offset_from_start = time[-1] + song_data['gaps'][index]
            time.append(offset_from_start)
    #plot 1:
    plt.subplot(2, 2, 1)
    plt.plot(time , length_y)
    plt.title("Pitch length over time when note appeared in song")

    #plot 2:
    plt.subplot(2, 2, 2)
    plt.plot(time , pitch_y)
    plt.title("pitch over time when note appeared in song")

    #plot 3:
    plt.subplot(2, 2, 3)
    plt.plot(time , volume_y)
    plt.title("note voulme over time when note appeared in song")
    plt.suptitle("Song data")
    plt.show()

def correlation_matrix():
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
    df = pd.DataFrame(columns=['gaps','lengths' , 'pitchs' , 'volumes'])

    #add data to df
    songs, note_to_int , int_to_note = normalize_notes(songs)
    for song in songs:
        for i in range(0,len(song['pitch'])):
            df.loc[len(df.index)] = [song['gaps'][i],song['lengths'][i],song['pitch'][i],song['volume'][i]] 
            print(i)
    corr_matrix = df.corr()
    print(corr_matrix)
    #plot data
    sns.heatmap(corr_matrix, xticklabels=corr_matrix.columns.values,yticklabels=corr_matrix.columns.values)
def main():
    song_path = "data/hand_picked/nujabes/reflection-eternal/intro-and-verse_nokey.mid"
    music_data = midi_path_to_data(song_path , split_instruments=False)
    music_data = normalize_notes(music_data)[0][0]
    plot_song_data_over_time(music_data)
    correlation_matrix()
main()