import os
def get_midi_filepaths():
    #add ~130 lofi midi files from non-hook theory datasets
    lofi_paths = ['data/hand_picked']
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

'''
def make_genre_dir(new_dir_name , genres_to_include):
    
    Queries the hook theory dataset by adding any song that has a genre in genres_to_include
    to a new dir name called new_dir_name. 
    



    #single songs can be in multiple genres!!

    #make new dir
    os.makedir(new_dir_name)
    #loop thru all files in hook theory xml data
    for root, dirs, files in os.walk(path):
        if file.endswith('song_info.json')
            song_info = eval(file)
        else:
            continue
        #if it has a genre type in genres_to_include
        for genre in genres_to_include:
            if genre in song_info["genres"]
                #get midi file path
                "piano-roll/"
                #add copy of file into new dir
                new_file_name = txt.split("/").join()
                shutil.copyfile(file , new_dir_name + new_file_name
'''