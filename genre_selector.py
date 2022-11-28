import os
import shutil
import json
def get_midi_filepaths():
    #add ~130 lofi midi files from non-hook theory datasets
    lofi_paths = ['data/lofi midis/LOFI PIANO MIDI FILES','data/hand_picked']#'data/lofi midis/lofi_from_others']
    midi_paths = []
    for path in lofi_paths:
      for root, dirs, files in os.walk(path):
          # select file name
          for file in files:
              # check the extension of files
              if file.endswith('.mid') or file.endswith('.midi'):
                  # print whole path of files
                  midi_paths.append(os.path.join(root, file))
    return midi_paths


def make_genre_dir(new_dir_name , genres_to_include):
    '''
    Queries the hook theory dataset by adding any song that has a genre in genres_to_include
    to a new dir name called new_dir_name. 
    '''

    #make new dir
    os.makedirs(new_dir_name)
    #loop thru all files in hook theory xml data
    for root, dirs, files in os.walk('data/hook_theory/xml'):
        for file in files:

            if file == 'song_info.json':
                song_info = json.load(open(root + '/' + file))
                #if it has a genre type in genres_to_include
                for genre in genres_to_include:
                    if genre in song_info["genres"]:
                        #get midi file path
                        #file = 'xml/a/artist/song/song_info.json'
                        sub_dirs = file.split('/')
                        print(sub_dirs)
                        for index , sub_dir in enumerate(sub_dirs):
                            if sub_dir == 'xml':
                                print('copied!')
                                sub_path_to_dir_to_add = sub_dirs[index + 1:-1].join('/')#/a/artist/song
                                dir_to_add = 'data/hook_theory/piano_roll' + sub_path_to_dir_to_add

                                shutil.copyfile(file , new_dir_name + dir_to_add)
                                

make_genre_dir('data/hip-hop' , ["Hip-Hop/Rap"])