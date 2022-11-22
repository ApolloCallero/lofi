'''
Lecture notes:
    low value in matrix profile -> there was somewhere else in the time sereis very similar to the current subsequence
    matrix profile indexs often recpricate to each other but not all the time

    motifs are low values becausse that means another sequence is similar to it, anomalys are high values because
    that means it is very far away from its closest time sequence
'''


from matrixprofile import *
import numpy as np
from midi_proccessing import *
from matplotlib import pyplot as plt
song_path = "data/hand_picked/nujabes/reflection-eternal/intro-and-verse_nokey.mid"
music_data = midi_path_to_data(song_path , split_instruments=False)
music_data = normalize_notes(music_data)[0][0]
pitchs = music_data['pitch']
gaps = music_data['gaps']

#caculate matrix profile

for m in [4,6,8,10]:
    mp = matrixProfile.stamp(pitchs,m)

    #Append np.nan to Matrix profile to enable plotting against raw data
    mp_adj = np.append(mp[0],np.zeros(m-1)+np.nan)


    
    #plot pitch data and matrix profile
    fig, (ax1, ax2) = plt.subplots(2,1,sharex=False)
    st = fig.suptitle('matrix profile of reflection-eternal by nujabes, window=' + str(m), fontsize="x-large")
    ax1.plot([i for i in range(len(pitchs))] , pitchs)
    ax1.set_ylabel('Notes', size=10)


    ax2.plot(np.arange(len(mp_adj)),mp_adj, color='red')
    ax2.set_ylabel('Matrix Profile', size=8)
    ax2.set_xlabel('Sample', size=8)
    plt.savefig('data/plots/' + 'matrix profile of reflection-eternal by nujabes, window=' + str(m))
    plt.show()
#plot pitch matrix profile

