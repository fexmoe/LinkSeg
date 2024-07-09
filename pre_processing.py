# for a given file: 
    # converts audio to wav if necessary
    # calculates beats
    # downsamples to 22.05 kHz

import argparse
import madmom
import os
import numpy as np
import librosa
import shutil
from pathlib import Path


from tqdm import tqdm
#from configuration import config
from pydub import AudioSegment
from input_output import FileStruct, write_beats

       


def madmom_beats(audiofile, y_, sr):
    
    if '.mp3' in str(audiofile):
        file_struct = FileStruct(audiofile)
        song_name = audiofile.split('.mp3')[0]
        dst = os.path.join(file_struct.ds_path, song_name+'.wav')
        # convert wav to mp3                                                            
        sound = AudioSegment.from_mp3(audiofile)
        sound.export(dst, format="wav")
        os.remove(audiofile)
        audiofile = dst
        y_, sr = librosa.load(audiofile, mono=True)
    
    if sr != 44100:
        y_ = librosa.resample(y_, orig_sr=sr, target_sr=44100)
        sr_ = 44100

    audio_duration = librosa.get_duration(y_, sr=sr_)
    proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    act = madmom.features.beats.TCNBeatProcessor()(y_)
    beat_times = np.asarray(proc(act))
    if beat_times[0] > 0:
        beat_times = np.insert(beat_times, 0, 0)
    new_beats = []
    for i in range(len(beat_times)):
        if beat_times[i] < audio_duration:
            new_beats.append(beat_times[i])
    beat_times = new_beats
    beats = librosa.time_to_frames(beat_times, sr=22050, hop_length=256)
    return beats, audio_duration


def compute_beats(audio_file, feat_config, y, sr):
    file_struct = FileStruct(audio_file)
    beat_frames, duration = madmom_beats(audio_file, y, sr)
    write_beats(file_struct, feat_config, beat_frames, duration)


def process_beats(file_struct, feat_config):
    if not os.path.isfile(file_struct.beat_file):
        y, sr = librosa.load(file_struct.audio_file, mono=True)
        compute_beats(file_struct.audio_file, feat_config, y, sr)
    else:
        print('Beats already found, skipping.')
    return 0


def get_paths(ds_path, config):
    tracklist = librosa.util.find_files(os.path.join(ds_path, 'audio'), ext=config.dataset.audio_exts)
    npy_path = os.path.join(ds_path, 'audio_npy')
    if not os.path.exists(npy_path):
        os.makedirs(npy_path)
    return tracklist, npy_path

def get_npy(fn):
    x, _ = librosa.core.load(fn, sr=22050)
    return x

def process_audio(file_struct):
    if not os.path.exists(file_struct.audio_npy_file):
        try:
            x = get_npy(file_struct.audio_npy_file)
            np.save(open(file_struct.audio_npy_file, 'wb'), x)
        except:
            pass

def organize_directories(file):
    file_name_with_extension = file.split('/')[-1]
    #print('file_name_with_extension =', file_name_with_extension)
    file_name = file.split('/')[-1].split('.')[0]
    #print('file_name =', file_name)
    overall_file_path = os.path.join('predictions', file_name)
    #print('overall_file_path =', overall_file_path)
    if not os.path.exists(overall_file_path):
        os.makedirs(overall_file_path)
    new_audio_file_name = os.path.join(overall_file_path, file_name_with_extension)
    shutil.copy(file, new_audio_file_name)
    new_audio_file_name = os.path.abspath(new_audio_file_name)
    return FileStruct(new_audio_file_name)
        
def pre_process(file, config):
    file_struct = organize_directories(file)
    process_beats(file_struct, config)
    process_audio(file_struct)
    return file_struct

    
    