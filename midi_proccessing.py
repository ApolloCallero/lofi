
import music21 as music
import os
def get_midi_filepaths():
    lofi_paths = ['data']
    midi_paths = []
    
    for path in lofi_paths:
      for root, dirs, files in os.walk(path):
          # select file name
          for file in files:
              # check the extension of files
              if file.endswith('.mid'):
                  # print whole path of files
                  midi_paths.append(os.path.join(root, file))
    return midi_paths



def make_note_num_dicts(seen_notes):
  '''
  Make a dictionairy in the form kinda like {A1:0 , B1:.01 , C1:.02 ....... F5:1}
  and another one with the key and 
  '''
  ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7','A8',
   'B-1', 'B-2', 'B-3', 'B-4', 'B-5', 'B-6',
    'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7','B8',
    'C#2', 'C#3', 'C#4', 'C#5', 'C#6', 'C#7', 
    'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 
    'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 
     'E-1','E-2', 'E-3', 'E-4', 'E-5', 'E-6',
     'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 
     'F#1', 'F#2', 'F#3', 'F#4', 'F#5', 'F#6', 
     'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 
     'G#2', 'G#3', 'G#4', 'G#5', 'G#6', 
     'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7']
    
  #C-C C# D D# E- E F F# G G# A- A A# B- B B#
  notes = ['C-','C', 'C#', 'D-','D','D#', 'E-', 'E','E#','F-' 'F', 'F#' ,'G','G#' ,'A-', 'A', 'A#' ,'B-' 'B', 'B#']
  
  octaves = 8
  pitch_to_num_dict = {}
  i = 0
  for octave in range(1,octaves+1):
    for note in note:
      for note_type in ["" , "-","#"]:
        if note + note_type + octave + note in seen_notes:
          continue
        else:
          pitch_to_num_dict[note + note_type + octave + note] = i
          i += 1
  #normalize the pitchs
  for key in list(pitch_to_num_dict.keys()):
    pitch_to_num_dict[key] /= len(list(pitch_to_num_dict.keys()))
  num_to_pitch = dict(zip( pitch_to_num_dict.values(), pitch_to_num_dict.keys()))
  return pitch_to_num_dict
def normalize_notes(songs):
  global n_pitch
  global n_gaps
  global n_lengths
  global n_volume
  combined_pitchs = []
  combined_gaps = []
  combined_lengths = []
  combined_volume = []
  for song in songs:
      for note in song['pitch']:
          combined_pitchs.append(note)
      for gap in song['gaps']:
        combined_gaps.append(gap)
      for length in song['lengths']:
        combined_lengths.append(length)
      for volume in song['volume']:
        combined_volume.append(volume)
  
  # maybe pitch/chords should be held as a category instead of number???
  total_notes = 0
  total_chords = 0
  all_pitch_chords = sorted(set(item for item in combined_pitchs))
  pitchs = []
  for i in all_pitch_chords:
      pitchs.append(i)
  print("# notes: " , len(pitchs))
  pitchs.sort()
  pitchnames = pitchs

  print(pitchnames)
  n_pitch = float(len(set(combined_pitchs)))
  n_gaps = float(len(set(combined_gaps)))
  n_lengths = float(len(set(combined_lengths)))
  n_volume = float(len(set(combined_volume)))

  note_to_num = dict((note, number / n_pitch) for number, note in enumerate(pitchnames))
  num_to_note = dict((number, note) for number, note in enumerate(pitchnames))

  #normalize notes HERE
  songs_normailized_notes = []# 2d list of songs and dict notes in the song
  for song in songs:
    notes_in_song = {'gaps':[],'lengths':[] , 'pitch':[],'octave':[]}
    for index in range(0,len(song['gaps'])):
      song['gaps'][index] /= n_gaps
      song['lengths'][index] /= n_lengths
      song['pitch'][index] = note_to_num[song['pitch'][index]]
      song['volume'][index] /= n_volume
  return songs , note_to_num , num_to_note
def midi_path_to_data(midi_path):
    try:
      midi = music.converter.parse(midi_path)
    except:
      print('Unable to read ' , midi_path)
      return None 
    notes_to_parse = midi.flat.notes

    
    note_gaps = []#1d list of time offsets/'gaps' between notes
    note_lengths = []
    note_pitch = []
    note_volume = []
    lastOffset = 0
    for note in midi.flat.notes:
        #add pitch for this note
        if isinstance(note, music.note.Note): #if it's a single note, we don't have to join it to any other notes in the series
            note_pitch.append(str(note.pitch))
        elif isinstance(note, music.chord.Chord): #split each note in a chord and have them be it's on 'step' in the time series just with no time between them
            for note_in_chord in note.notes:
              note_pitch.append(str(note_in_chord.pitch))
              note_gaps.append(0)
              note_lengths.append(note_in_chord.quarterLength)
              note_volume.append(note_in_chord.volume.velocity)
            continue
        
        if note_gaps == []:
            note_gaps = [0]
            note_lengths.append(float(note.quarterLength))
            note_volume.append(note.volume.velocity)
            continue
    
        #offset/gap from start
        totalOffset = note.getOffsetInHierarchy(midi)

        # add 'gap' from last note 
        offset_from_last_note = totalOffset - lastOffset
        note_gaps.append(offset_from_last_note)
        lastOffset = totalOffset

        #add 'note length'
        note_lengths.append(float(note.quarterLength))
        
        #add note volume
        note_volume.append(note.volume.velocity)



    return {'gaps':note_gaps , 'lengths':note_lengths , 'pitch':note_pitch , 'volume':note_volume}
def prepare_song_data_for_model(songs , num_prev_notes):

  global note_to_int
  global int_to_note
  songs, note_to_int , int_to_note = normalize_notes(songs)
  network_input = []
  network_output = []

  # create input sequences and the corresponding outputs
  for song in songs:
    for note_num in range(num_prev_notes, len(song['gaps']), 1):


        #sequence_in = song[note_num - sequence_length:note_num]#notes before
        prev_gaps = song['gaps'][note_num - num_prev_notes:note_num]
        prev_lengths = song['lengths'][note_num - num_prev_notes:note_num]
        prev_pitch = song['pitch'][note_num - num_prev_notes:note_num]
        prev_volume = song['volume'][note_num - num_prev_notes:note_num]

        #normalize input
        step_input = list(zip(prev_gaps , prev_lengths , prev_pitch , prev_volume))

        #output is [gap from last note , note length , pitch , velocity]
        curr_note_data = [song['gaps'][note_num], song['lengths'][note_num], song['pitch'][note_num] , song['pitch'][note_num]]


        network_input.append(step_input)
        network_output.append(curr_note_data)

  return network_input , network_output
def list_instruments(midi):
    partStream = midi.parts.stream()
    print("List of instruments found on MIDI file:")
    for p in partStream:
        aux = p
        print (p.partName)
    for n in midi.flat.notes:
        if type(n) != music.chord.Chord:
          print("Note: %s %d %0.1f" % (n.pitch.name, n.pitch.octave, n.duration.quarterLength))

