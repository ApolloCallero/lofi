
import music21 as music
import os





def make_note_num_dicts(seen_notes):
  '''
  Make a dictionairy in the form kinda like {A1:0 , B1:.01 , C1:.02 ....... F5:1}
  and another one with the key and 
  '''    

  notes = ["C" , "D" , "E" , "F" , "G" , "A","B"]
  octaves = 8
  pitch_to_num_dict = {}
  i = 0
  for octave in range(-1,octaves+1):
    for note in notes:
      for note_type in ["-" , "","#"]:
          pitch_to_num_dict[note + note_type + str(octave)] = i
          i += 1
  #normalize the pitchs
  for key in list(pitch_to_num_dict.keys()):
    pitch_to_num_dict[key] /= len(list(pitch_to_num_dict.keys()))
  num_to_pitch = dict(zip( pitch_to_num_dict.values(), pitch_to_num_dict.keys()))
  return pitch_to_num_dict ,  num_to_pitch
def normalize_notes(songs):
  '''
  params:
    songs: dict of in the form of { {gaps:[3,..5] , lengths:[.1 , ... .05] , pitch:['E4',...'A3'],volume:[40,...90]}  }
            where each item in songs is dict representing a songs data
  returns: same dict but each value is a number between 1 and 0
  '''
  global num_to_pitch
  combined_pitchs = []
  combined_gaps = []
  combined_lengths = []
  combined_volume = []
  count = 0
  for song in songs:
      count += 1
      print(count)
      for note in song['pitch']:
          combined_pitchs.append(note)
      for gap in song['gaps']:
        combined_gaps.append(gap)
      for length in song['lengths']:
        combined_lengths.append(length)
  unique_pitchs = sorted(set(item for item in combined_pitchs))
  pitch_to_num , num_to_pitch = make_note_num_dicts(seen_notes=unique_pitchs)

  #normalize notes here
  songs_normailized_notes = []# 2d list of songs and dict notes in the song
  max_gap = max(combined_gaps)
  max_length = max(combined_lengths)
  for song in songs:
    count -= 1
    print(count)
    for index in range(0,len(song['pitch'])):
      song['gaps'][index] /= max_gap
      song['lengths'][index] /= max_length
      song['pitch'][index] = pitch_to_num[song['pitch'][index]]#match the pitch lettter to a normalized number
  return songs , pitch_to_num , num_to_pitch
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
def prepare_song_data_for_model(songs , num_prev_notes):
  '''
  Parmas: 
    Songs: dict of dicts where each dict represents notes in a song
    num_prev_notes: the number of notes in the input
  
  Output: 
    network_input: list input for the model in the shape of (#seq , num_prev_notes , num_features)
    network_output: list ouput for each sequennce/time step in the shape of (#seq , num_features)
  '''
  songs, note_to_int , int_to_note = normalize_notes(songs)
  network_input = []
  network_output = []

  # create input sequences and the corresponding outputs
  count = 0
  print(songs)
  for song in songs:
    count += 1
    print(count)
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