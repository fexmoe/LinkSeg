from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

import librosa
from tqdm import tqdm
from sklearn import preprocessing
from torch.utils import data
from sklearn.preprocessing import OneHotEncoder
import mir_eval
import pickle
from data_utils import *
import ujson


class GNN_dset(torch.utils.data.Dataset):
    def __init__(self, split, tracklist, max_len, n_embedding, hop_length):
        self.tracklist = clean_tracklist_audio(data_path=[], annotations=False, tracklist_=tracklist)
        self.max_len = max_len
        self.hop_length = hop_length
        self.n_embedding = n_embedding
        self.split == split
        self.beat_frames_list = self.build_SSMS_ref()

    def build_SSMS_ref(self):
        beat_frames_list = []
        
        for track in tqdm(self.tracklist):
            file_struct = FileStruct(track)
            beat_frames = read_beats(file_struct.beat_file)
            beat_frames = librosa.util.fix_frames(beat_frames)
            if len(beat_frames)>self.max_len:
                beat_frames = beat_frames[::2]
            beat_frames_list.append(beat_frames)

        return beat_frames_list
    

    def __getitem__(self, index):
        track = self.tracklist[index]
        file_struct =  FileStruct(track)
        beat_frames = self.beat_frames_list[index]
        
        beat_times = librosa.frames_to_time(beat_frames, sr=22050, hop_length=self.hop_length)
        beat_frames = librosa.time_to_frames(beat_times, sr=22050, hop_length=1)
        pad_width = ((self.hop_length*self.n_embedding) - 2)//2 
        waveform = np.load(file_struct.audio_npy_file)
        features_padded = np.pad(waveform, pad_width=((pad_width, pad_width)), mode='edge')
        features = np.stack([features_padded[i:i+pad_width*2] for i in beat_frames], axis=0)
        return track, features
    
    def __len__(self):
        return len(self.tracklist)
    

def ssm_dataloader_test(split, batch_size, tracklist, max_len, n_embedding, hop_length, num_workers):
    data_loader = data.DataLoader(dataset=GNN_dset(split, tracklist, max_len, n_embedding, hop_length),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=False,
                                  num_workers=num_workers)
    return data_loader