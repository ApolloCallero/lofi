
import music21 as music
import os
import numpy as np





def midi_path_to_data(midi_path , split_instruments):
  '''
  TBD: split 'instruments' or not?'
  '''
  try:
    midi = music.converter.parse(midi_path)
  except:
    print('Unable to read ' , midi_path)
    return None 
  notes_to_parse = midi.flat.notes

    


  if split_instruments == False:
    parts_dicts = []
    #for part in midi.parts:
    note_gaps = []#1d list of time offsets/'gaps' between notes
    note_lengths = []
    note_pitchs = []
    note_volume = []
    lastOffset = 0
    for note in midi.flat.notes:
        #add pitch for this note
        if isinstance(note, music.note.Note): #if it's a single note, we don't have to join it to any other notes in the series
            note_pitchs.append(str(note.pitch))
        elif isinstance(note, music.chord.Chord): #split each note in a chord and have them be it's on 'step' in the time series just with no time between them
            for note_in_chord in note.notes:
              note_pitchs.append(str(note_in_chord.pitch))
              note_gaps.append(0)
              note_lengths.append(note_in_chord.quarterLength)
              note_volume.append(note_in_chord.volume.velocityScalar)
            continue
        
        if note_gaps == []:
            note_gaps = [0]
            note_lengths.append(float(note.quarterLength))
            note_volume.append(note.volume.velocityScalar)
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
        note_volume.append(note.volume.velocityScalar)
    if len(list(set(note_volume))) != 1:
      print(len(list(set(note_volume))))
    return [{'gaps':note_gaps , 'lengths':note_lengths , 'pitch':note_pitchs , 'volume':note_volume}]
  else:
    parts_dicts = []
    #for part in midi.parts:
    note_gaps = []#1d list of time offsets/'gaps' between notes
    note_lengths = []
    note_pitchs = []
    note_volume = []
    lastOffset = 0
    for note in midi.flat.notes:
        #add pitch for this note
        if isinstance(note, music.note.Note): #if it's a single note, we don't have to join it to any other notes in the series
            note_pitchs.append(str(note.pitch))
        elif isinstance(note, music.chord.Chord): #split each note in a chord and have them be it's on 'step' in the time series just with no time between them
            for note_in_chord in note.notes:
              note_pitchs.append(str(note_in_chord.pitch))
              note_gaps.append(0)
              note_lengths.append(note_in_chord.quarterLength)
              note_volume.append(note_in_chord.volume.velocityScalar)
            continue
        
        if note_gaps == []:
            note_gaps = [0]
            note_lengths.append(float(note.quarterLength))
            note_volume.append(note.volume.velocityScalar)
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
        note_volume.append(note.volume.velocityScalar)

    #a lot of parts in non-lofi songs have one note only, drop it here if thats that case
    for note in note_pitchs:
      if note != note[0]:
        parts_dicts.append({'gaps':note_gaps , 'lengths':note_lengths , 'pitch':note_pitchs , 'volume':note_volume})
  return parts_dicts

def get_dicts(songs , num_prev_notes):
  '''
  Makes dicts that transfer each normalized feature value into the number that will be used for it it training and vice versa
  Uses i*num_prev_note first so the values match up to the one hot encoded output in training
  '''

  #get every i * num_prev_notes note in each song
  gaps = []
  lengths = []
  pitchs = []
  volumes = []
  for song in songs:
    for i in range(0, len(song['pitch']) , num_prev_notes):
      if i == 0:
        continue
      gaps.append(round(song['gaps'][i] , 3)) 
      lengths.append(round(song['lengths'][i] , 3)) 
      pitchs.append(song['pitch'][i]) 
      volumes.append(round(song['volume'][i] , 3))

  #get every unique value for each feature
  unique_gaps = list(set(gaps))
  unique_lengths = list(set(lengths))
  unique_pitchs = list(set(pitchs))
  unique_volumes= list(set(volumes))

  #do non output notes next
  for song in songs:
    for i in range(0, len(song['pitch'])):
      if round(song['gaps'][i],3) not in unique_gaps:
        unique_gaps.append(round(song['gaps'][i] , 3))
      if round(song['lengths'][i],3) not in unique_lengths:
        unique_lengths.append(round(song['lengths'][i] , 3))
      if song['pitch'][i] not in unique_pitchs:
        unique_pitchs.append(song['pitch'][i])
      if round(song['volume'][i],3) not in unique_volumes:
        unique_volumes.append(round(song['volume'][i] , 3))
  #make dict's
  gap_to_num = dict(zip(unique_gaps , [i for i in range(0,len(unique_gaps))]))
  num_to_gap = dict(zip([i for i in range(0,len(unique_gaps))] , unique_gaps ))

  length_to_num = dict(zip(unique_lengths , [i for i in range(0,len(unique_lengths))]))
  num_to_length = dict(zip([i for i in range(0,len(unique_lengths))] , unique_lengths ))

  pitch_to_num = dict(zip(unique_pitchs , [i for i in range(0,len(unique_pitchs))]))
  num_to_pitch = dict(zip([i for i in range(0,len(unique_pitchs))] , unique_pitchs ))

  volume_to_num = dict(zip(unique_volumes , [i for i in range(0,len(unique_volumes))]))
  num_to_volume = dict(zip([i for i in range(0,len(unique_volumes))] , unique_volumes ))
  return gap_to_num , num_to_gap , length_to_num , num_to_length , pitch_to_num , num_to_pitch , volume_to_num , num_to_volume

def prepare_song_data_for_model(songs , num_prev_notes):
  '''
  Parmas: 
    Songs: dict of dicts where each dict represents notes in a song
    num_prev_notes: the number of notes in the input
  
  Output: 
    network_input: list input for the model in the shape of (#seq , num_prev_notes , num_features)
    network_output: list ouput for each sequennce/time step in the shape of (#seq , num_features)
  '''
  gap_to_num , num_to_gap , length_to_num , num_to_length , pitch_to_num , num_to_pitch , volume_to_num , num_to_volume = get_dicts(songs , num_prev_notes)
  print('in prepare: ',num_to_pitch)
  #map each value in songs to a number for training
  for song in songs:
    for i in range(0, len(song['pitch'])):
      song['gaps'][i] = gap_to_num[round(song['gaps'][i] , 3)]
      song['lengths'][i] = length_to_num[round(song['lengths'][i] , 3)]
      song['pitch'][i] = pitch_to_num[song['pitch'][i]]
      song['volume'][i] = volume_to_num[round(song['volume'][i] , 3)]
  network_input = []
  network_output = []

  # create input sequences and the corresponding outputs
  count = 0
  for song in songs:
    count += 1
    for note_num in range(num_prev_notes, len(song['pitch'])):


        #sequence_in = song[note_num - sequence_length:note_num]#notes before
        prev_gaps = song['gaps'][note_num - num_prev_notes:note_num]
        prev_lengths = song['lengths'][note_num - num_prev_notes:note_num]
        prev_pitch = song['pitch'][note_num - num_prev_notes:note_num]
        prev_volume = song['volume'][note_num - num_prev_notes:note_num]

        #normalize input
        step_input = list(zip(prev_gaps , prev_lengths , prev_pitch , prev_volume))

        #output is [gap from last note , note length , pitch , velocity]
        curr_note_data = [song['gaps'][note_num], song['lengths'][note_num], song['pitch'][note_num] , song['volume'][note_num]]


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

def round_pitch(pitch_num , real_pitch_nums):
  closest = real_pitch_nums[0]
  for pitch_key in real_pitch_nums:
    if abs(pitch_num - closest) > abs(pitch_num - pitch_key):
      closest = pitch_key
  return closest
def predictions_to_music(notes , unnormalized_data ):
  '''
  params:
    notes: 2d list of notes outputted from the model, each note should be in the form [gap from last note , note length , pitch , volume]
          with eac h value being between 1 and 0
    unnormalized data: data before the normalization, used so we can get our normalized predictions back to the real output
    num_to_pitch: dict in the form of {.01: C1 ... , 1:G8}
  '''

  #'unormalize' notes
  combined_gaps = []
  combined_lengths = []
  combined_volume = []
  for song in unnormalized_data:
      for gap in song['gaps']:
        combined_gaps.append(gap)
      for length in song['lengths']:
        combined_lengths.append(length)

  max_gap = max(combined_gaps)
  max_length = max(combined_lengths)
  readable_notes = []
  print('final output: ')
  for note in notes:
    note = note[0]
    gap = note[0] * max_gap
    length = note[1] * max_length
    pitch = num_to_pitch[round_pitch(note[2] , list(num_to_pitch.keys()))]
    volume = note[3] * 127
    readable_notes.append([gap , length , pitch , volume])
  #combine the notes into a music21 stream  
  offset = 0
  output_notes = []
  for note in readable_notes:
      print(note)
      gap = note[0]
      offset += gap
      length = note[1]
      pitch = note[2]
      volume = note[3]
      new_note = music.note.Note(pitch) #storing it in the object
      new_note.offset = offset #connecting it to our offset command later on
      new_note.storedInstrument = music.instrument.Piano() #playing it with piano
      new_note.volume.velocity = volume
      new_note.quarterLength = length
      output_notes.append(new_note) #adding it to the song
  print(len(output_notes))
  print(offset)
  s = music.stream.Stream(output_notes)
  mf = s.write('midi', fp="data/testOutput.mid")
  s.show('midi')
def predictions_to_music_seperate_models(pitchs ,gaps, unnormalized_data , notes_to_generate=100):
  '''
  params:
    notes: 2d list of notes outputted from the model, each note should be in the form [gap from last note , note length , pitch , volume]
          with eac h value being between 1 and 0
    unnormalized data: data before the normalization, used so we can get our normalized predictions back to the real output
    num_to_pitch: dict in the form of {.01: C1 ... , 1:G8}
  '''
  #gap_to_num , num_to_gap , length_to_num , num_to_length , pitch_to_num , num_to_pitch , volume_to_num , num_to_volume = get_dicts(songs , num_prev_notes)
  #'unormalize' notes
  combined_gaps = []
  combined_lengths = []
  combined_volume = []
  for song in unnormalized_data:
      for gap in song['gaps']:
        combined_gaps.append(gap)
      for length in song['lengths']:
        combined_lengths.append(length)

  max_gap = max(combined_gaps)
  print('max gap:',max_gap)
  max_length = max(combined_lengths)
  readable_notes = []
  print('final output: ')
  for i in range(0,notes_to_generate):
    gap = gaps[i]
    #length = note[1] * max_length

    #print(pitchs[i])
    pitch = pitchs[i]#num_to_pitch[round_pitch(pitchs[i][0]/len(list(num_to_pitch.keys()))  , list(num_to_pitch.keys()))]
    #volume = note[3] * 127
    readable_notes.append([gap , pitch])
  #combine the notes into a music21 stream  
  offset = 0
  output_notes = []
  for note in readable_notes:
      gap = note[0]
      print(gap)
      pitch = note[1]
      new_note = music.note.Note(pitch) #storing it in the object
      offset += min(gap , 2)
      new_note.offset = offset #connecting it to our offset command later on
      new_note.storedInstrument = music.instrument.Piano() #playing it with piano
      new_note.volume.velocity = 80
      #new_note.quarterLength = .5#length
      output_notes.append(new_note) #adding it to the song
  print(len(output_notes))
  print(offset)
  s = music.stream.Stream(output_notes)
  mf = s.write('midi', fp="data/900_song_no_guess_seq-10-fixed.mid")
  s.show('midi')

def split_features(input , output):
  gap_x = []
  length_x = []
  pitch_x  = []
  vol_x = []
  gap_y = []
  length_y = []
  pitch_y  = []
  vol_y = []
  for seq in input:
    seq_gap_x = []
    seq_length_x = []
    seq_pitch_x = []
    seq_vol_x = []
    for note in seq:
      seq_gap_x.append(note[0])
      seq_length_x.append(note[1])
      seq_pitch_x.append(note[2])
      seq_vol_x.append(note[3])
    gap_x.append([seq_gap_x])
    length_x.append([seq_length_x])
    pitch_x.append([seq_pitch_x])
    vol_x.append(seq_vol_x)
  for note in output:
    gap_y.append(note[0])
    length_y.append(note[1])
    pitch_y.append(note[2])
    vol_y.append(note[3])
  x = np.array(gap_x)
  print('gap in :',type(x[0]))
  print(x[0])
  return np.array(gap_x) , np.array(length_x) , np.array(pitch_x) , np.array(vol_x) , np.array(gap_y) , np.array(length_y) , np.array(pitch_y) , np.array(vol_y)  

    