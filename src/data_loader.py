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
    def __init__(self, split, tracklist, max_len, n_embedding, hop_length, nb_section_labels):
        self.max_len = max_len
        self.hop_length = hop_length
        self.n_embedding = n_embedding
        self.split = split 
        self.nb_section_labels = nb_section_labels
        self.tracklist = clean_tracklist_audio(data_path=[], annotations=True, tracklist_=tracklist)
        self.SSMS, self.labels, self.beat_frames_list, self.tracklist = self.build_SSMS_ref()
        

    def build_SSMS_ref(self):
        SSMS_ref = []
        labels_ref = []
        beat_frames_list = []
        tracklist = []

        le = preprocessing.LabelEncoder()
        one_hot_le = OneHotEncoder()
        
        for track in tqdm(self.tracklist):
            file_struct = FileStruct(track)
            beat_frames, duration = read_beats(file_struct.beat_file)
            beat_frames = librosa.util.fix_frames(beat_frames)
            beat_frames = downsample_frames(beat_frames, max_length=self.max_len)
           
            beat_times = librosa.frames_to_time(beat_frames, sr=22050, hop_length=256)
            ref_labels, ref_times, duration_ = get_ref_labels(file_struct, 0)

            ref_inter = times_to_intervals(ref_times)
            (ref_inter, ref_labels) = mir_eval.util.adjust_intervals(ref_inter, list(ref_labels), t_min=0, t_max=duration)
            ref_times = intervals_to_times(ref_inter)

            ref_labels_segment = np.arange(len(ref_labels))
            labels_list_segment = get_labels(beat_frames, list(ref_times), list(ref_labels_segment))
            labels_list_segment = le.fit_transform(np.array(labels_list_segment))
            labels_list_oh_segment = one_hot_le.fit_transform(labels_list_segment.reshape(-1,1)).toarray()

            if self.nb_section_labels == 7:
                ref_labels_merged = merge_labels(ref_labels, indices, substrings)
                labels_list_merged = get_labels(beat_frames, list(ref_times), list(ref_labels_merged))
                labels_list_oh_merged = np.zeros((len(labels_list_merged), len(indices)))
            else:
                ref_labels_merged = merge_labels(ref_labels, indices_9classes, substrings_9classes)
                labels_list_merged = get_labels(beat_frames, list(ref_times), list(ref_labels_merged))
                labels_list_oh_merged = np.zeros((len(labels_list_merged), len(indices_9classes)))

            
            for i in range(len(labels_list_oh_merged)):
                labels_list_oh_merged[i,labels_list_merged[i]] = 1

            assert len(labels_list_merged) == len(beat_frames)
            assert len(labels_list_segment) == len(beat_frames)

            SSM_segment = labels_list_oh_segment @ labels_list_oh_segment.T
            SSM_merged = labels_list_oh_merged @ labels_list_oh_merged.T

            SSMS_ref.append((SSM_segment, SSM_merged))
            labels_ref.append(labels_list_oh_merged)
            beat_frames_list.append(beat_frames)
            tracklist.append(track)

        return SSMS_ref, labels_ref, beat_frames_list, tracklist
    

    def __getitem__(self, index):
        track = self.tracklist[index]
        file_struct =  FileStruct(track)
        beat_frames = self.beat_frames_list[index]
        
        beat_times = librosa.frames_to_time(beat_frames, sr=22050, hop_length=256)
        beat_frames = librosa.time_to_frames(beat_times, sr=22050, hop_length=1)

        (SSM_segment, SSM_merged) = self.SSMS[index]
        labels = self.labels[index]
        pad_width = ((self.hop_length*self.n_embedding) - 2)//2 
        waveform = np.load(file_struct.audio_npy_file)
        features_padded = np.pad(waveform, pad_width=((pad_width, pad_width)), mode='edge')
        features = np.stack([features_padded[i:i+pad_width*2] for i in beat_frames], axis=0)
        return track, features, (SSM_segment, SSM_merged), labels
    

    def __len__(self):
        return len(self.tracklist)
    

def ssm_dataloader(split, batch_size, tracklist, max_len, n_embedding, hop_length, nb_section_labels, num_workers):
    data_loader = data.DataLoader(dataset=GNN_dset(split, tracklist, max_len, n_embedding, hop_length, nb_section_labels),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=False,
                                  num_workers=num_workers)
    return data_loader