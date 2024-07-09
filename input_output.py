# -*- coding: utf-8 -*-
"""
A lot of functions are copied from MSAF:
https://github.com/urinieto/msaf/blob/master/msaf/input_output.py
These set of functions help the algorithms of MSAF to read and write
files of the Segmentation Dataset.
"""
import json
import os
from pathlib import Path

import ujson
import numpy as np
import jams
import mir_eval
import collections
import datetime

# Local stuff
import utils
#from configuration import config
import librosa
from tqdm import tqdm
import pandas as pd

# Put dataset config in a global var
#ds_config = config.dataset


class FileStruct:
    def __init__(self, audio_file):
        audio_file = Path(audio_file)
        self.audio_file = audio_file
        self.track_name = audio_file.stem
        self.ds_path = audio_file.parents[0]
        self.beat_file = self.ds_path.joinpath(self.track_name +'_beats_.json')
        self.audio_npy_file = self.ds_path.joinpath(self.track_name + '.npy')  
        self.predictions_file = self.ds_path.joinpath(self.track_name+ '.jams')                                            

        

    def __repr__(self):
        """Prints the file structure."""
        return "FileStruct(\n\tds_path=%s,\n\taudio_file=%s,," % (
                self.ds_path, self.audio_file)


def clean_tracklist_audio(config, split, annotations=None, tracklist_=[]):
    if split == 'train':
        tracklist = []
        for ds_path in config.listb[:-1]:
            if len(tracklist_) < 1:
                tracklist += librosa.util.find_files(os.path.join(ds_path, 'audio'), ext=config.dataset.audio_exts)
            else:
                tracklist = tracklist_
        if annotations == None:
            annotations = False
    elif split == 'valid':
        if len(tracklist_) < 1:
            tracklist = librosa.util.find_files(os.path.join(config.listb[-1], 'audio'), ext=config.dataset.audio_exts)
        else:
            tracklist = tracklist_
        annotations = True
    tracklist_clean = []
    for song in tqdm(tracklist):
        file_struct = FileStruct(song)
        if os.path.isfile(file_struct.beat_file) and os.path.isfile(file_struct.audio_npy_file):
            if annotations and os.path.isfile(file_struct.ref_file):
                tracklist_clean.append(song)
            elif not annotations:
                tracklist_clean.append(song)
            else:
                print(song)
        else:
            print(song)
    return tracklist_clean


def read_hier_references(audio_path, annotation_id=0, exclude_levels=[]):

    ds_path = os.path.dirname(os.path.dirname(audio_path))

    hier_bounds = []
    hier_labels = []
    
    namespaces = ["segment_salami_upper", "segment_salami_function",
                  "segment_open", "segment_tut", "segment_salami_lower", "multi_segment"]

    ds_path = os.path.dirname(os.path.dirname(audio_path))

    low = True
    if 'SALAMI' in ds_path:
        if low: 
            try:
                jam_path = os.path.join(ds_path, ds_config.references_dir+'_hier_low',
                                        os.path.basename(audio_path)[:-4] +
                                        ds_config.references_ext)
                
                jam = jams.load(jam_path, validate=False)
            except:
                jam_path = os.path.join(ds_path, ds_config.references_dir+'_hier_low',
                                        os.path.basename(audio_path)[:-5] +
                                        ds_config.references_ext)
        else:
            try:
                jam_path = os.path.join(ds_path, ds_config.references_dir+'_hier_up',
                                        os.path.basename(audio_path)[:-4] +
                                        ds_config.references_ext)
                
                jam = jams.load(jam_path, validate=False)
            except:
                jam_path = os.path.join(ds_path, ds_config.references_dir+'_hier_up',
                                        os.path.basename(audio_path)[:-5] +
                                        ds_config.references_ext)
    else:

        # Read references
        try:
            jam_path = os.path.join(ds_path, ds_config.references_dir+'_hier',
                                    os.path.basename(audio_path)[:-4] +
                                    ds_config.references_ext)
            
            jam = jams.load(jam_path, validate=False)
        except:
            jam_path = os.path.join(ds_path, ds_config.references_dir+'_hier',
                                    os.path.basename(audio_path)[:-5] +
                                    ds_config.references_ext)

    jam = jams.load(jam_path, validate=False)
    file_struct = FileStruct(audio_path)
    with open(file_struct.beat_file, 'r') as f:
        out_json = ujson.load(f)
    f.close()
    duration = float(out_json["globals"]["duration"])
    # Remove levels if needed
    for exclude in exclude_levels:
        if exclude in namespaces:
            namespaces.remove(exclude)

    # Build hierarchy references
    for i in jam['annotations']['multi_segment']:
        bounds_0, labels_0 = [], []
        bounds_1, labels_1 = [], []
        bounds_2, labels_2 = [], []
        for bound in i['data']:
            if bound.value['level'] == 0:
                bounds_0.append(bound.time)
                labels_0.append(bound.value['label'])
            elif bound.value['level'] == 1:
                bounds_1.append(bound.time)
                labels_1.append(bound.value['label'])
            elif bound.value['level'] == 2:
                bounds_2.append(bound.time)
                labels_2.append(bound.value['label'])
        if len(bounds_0) > 0:
            hier_bounds.append(bounds_0)
            hier_labels.append(labels_0)
        if len(bounds_1) > 0:
            hier_bounds.append(bounds_1)
            hier_labels.append(labels_1)
        if len(bounds_2) > 0:
            hier_bounds.append(bounds_2)
            hier_labels.append(labels_2)
    
    #print('Input output =', [len(i) for i in hier_bounds], [i for i in hier_labels])
    ref_inters_list = []
    for ref_int, ref_lab in zip(hier_bounds, hier_labels):
        #
        ref_int = utils.times_to_intervals(ref_int)
        (ref_int, ref_lab) = mir_eval.util.adjust_intervals(ref_int, ref_lab, t_min=0, t_max=duration)
        #ref_int, ref_lab = utils.remove_empty_segments(ref_int, ref_lab)
        ref_inters_list.append(ref_int)
    
    
    return ref_inters_list, hier_labels


def get_functional_labels_salami(file_struct, annot=-1):
    jam = jams.load(str(file_struct.ref_file), validate=False)
    try:
        annot = jam.search(namespace='segment_salami_function.*')[annot]
    except:
        print(file_struct.audio_file)
    ref_inters, ref_labels = annot.to_interval_values()
    ref_times = utils.intervals_to_times(ref_inters)
    ref_inters = utils.times_to_intervals(ref_times)
    (ref_inters, ref_labels) = mir_eval.util.adjust_intervals(ref_inters, ref_labels, t_min=0, t_max=ref_inters.max())
    duration = jam.file_metadata.duration
    return ref_labels, ref_times, duration



def read_references(audio_path, estimates, annotator_id=0, hier=False):
    """Reads the boundary times and the labels.

    Parameters
    ----------
    audio_path : str
        Path to the audio file

    Returns
    -------
    ref_times : list
        List of boundary times
    ref_labels : list
        List of labels

    Raises
    ------
    IOError: if `audio_path` doesn't exist.
    """
    # Dataset path
    ds_path = os.path.dirname(os.path.dirname(audio_path))


    if not estimates:
    # Read references
        try:
            jam_path = os.path.join(ds_path, ds_config.references_dir,
                                    os.path.basename(audio_path)[:-4] +
                                    ds_config.references_ext)
            

            jam = jams.load(jam_path, validate=False)
        except:
            jam_path = os.path.join(ds_path, ds_config.references_dir,
                                    os.path.basename(audio_path)[:-5] +
                                    ds_config.references_ext)
            

            jam = jams.load(jam_path, validate=False)
    else:
        try:
            jam_path = os.path.join(ds_path, ds_config.references_dir+'/estimates/',
                                    os.path.basename(audio_path)[:-4] +
                                    ds_config.references_ext)
            

            jam = jams.load(jam_path, validate=False)
        except:
            jam_path = os.path.join(ds_path, ds_config.references_dir+'/estimates/',
                                    os.path.basename(audio_path)[:-5] +
                                    ds_config.references_ext)
            

            jam = jams.load(jam_path, validate=False)


    ##################
    low = True # Low parameter for SALAMI
    ##################


    if not hier:
        if low:  
            try:
                ann = jam.search(namespace='segment_salami_lower.*')[0]
            except:
                try:
                    ann = jam.search(namespace='segment_salami_upper.*')[0]
                except:
                    ann = jam.search(namespace='segment_.*')[annotator_id]
        else:
            try:
                ann = jam.search(namespace='segment_salami_upper.*')[0]
            except:
                ann = jam.search(namespace='segment_.*')[annotator_id]
        
        ref_inters, ref_labels = ann.to_interval_values()
        ref_times = utils.intervals_to_times(ref_inters)
        
        return ref_times, ref_labels

    else:

        list_ref_times, list_ref_labels = [], []
        upper = jam.search(namespace='segment_salami_upper.*')[0]
        ref_inters_upper, ref_labels_upper = upper.to_interval_values()
        
        list_ref_times.append(utils.intervals_to_times(ref_inters_upper))
        list_ref_labels.append(ref_labels_upper)

        annotator = upper['annotation_metadata']['annotator']
        lowers = jam.search(namespace='segment_salami_lower.*')

        for lower in lowers:
            if lower['annotation_metadata']['annotator'] == annotator:
                ref_inters_lower, ref_labels_lower = lower.to_interval_values()
                list_ref_times.append(utils.intervals_to_times(ref_inters_lower))
                list_ref_labels.append(ref_labels_lower)

        return list_ref_times, list_ref_labels





def read_references_2annot(audio_path, index):
    ds_path = os.path.dirname(os.path.dirname(audio_path))

    # Read references
    try:
        jam_path = os.path.join(ds_path, ds_config.references_dir,
                                os.path.basename(audio_path)[:-4] +
                                ds_config.references_ext)
        

        jam = jams.load(jam_path, validate=False)
    except:
        jam_path = os.path.join(ds_path, ds_config.references_dir,
                                os.path.basename(audio_path)[:-5] +
                                ds_config.references_ext)
        

        jam = jams.load(jam_path, validate=False)


    list_ref_times, list_ref_labels = [], []
    upper = jam.search(namespace='segment_salami_upper.*')[index]
    ref_inters_upper, ref_labels_upper = upper.to_interval_values()
    duration = jam.file_metadata.duration
    ref_inters_upper = utils.intervals_to_times(ref_inters_upper)
    #
    ref_inters_upper, ref_labels_upper = utils.remove_empty_segments(ref_inters_upper, ref_labels_upper)
    ref_inters_upper = utils.times_to_intervals(ref_inters_upper)
    (ref_inters_upper, ref_labels_upper) = mir_eval.util.adjust_intervals(ref_inters_upper, ref_labels_upper, t_min=0, t_max=duration)
    list_ref_times.append(ref_inters_upper)
    list_ref_labels.append(ref_labels_upper)


    lower = jam.search(namespace='segment_salami_lower.*')[index]
    ref_inters_lower, ref_labels_lower = lower.to_interval_values()
    ref_inters_lower = utils.intervals_to_times(ref_inters_lower)
    #(ref_inters_lower, ref_labels_lower) = mir_eval.util.adjust_intervals(ref_inters_lower, ref_labels_lower, t_min=0, t_max=duration)
    ref_inters_lower, ref_labels_lower = utils.remove_empty_segments(ref_inters_lower, ref_labels_lower)
    ref_inters_lower = utils.times_to_intervals(ref_inters_lower)
    (ref_inters_lower, ref_labels_lower) = mir_eval.util.adjust_intervals(ref_inters_lower, ref_labels_lower, t_min=0, t_max=duration)
    list_ref_times.append(ref_inters_lower)
    list_ref_labels.append(ref_labels_lower)



    return list_ref_times, list_ref_labels, duration




def read_references_jsd(audio_path, level):

    ds_path = os.path.dirname(os.path.dirname(audio_path))

    
    namespaces = ["segment_salami_upper", "segment_salami_function",
                  "segment_open", "segment_tut", "segment_salami_lower", "multi_segment"]

    ds_path = os.path.dirname(os.path.dirname(audio_path))

    try:
        jam_path = os.path.join(ds_path, ds_config.references_dir+'_hier',
                                os.path.basename(audio_path)[:-4] +
                                ds_config.references_ext)
        
        jam = jams.load(jam_path, validate=False)
    except:
        jam_path = os.path.join(ds_path, ds_config.references_dir+'_hier',
                                os.path.basename(audio_path)[:-5] +
                                ds_config.references_ext)

    jam = jams.load(jam_path, validate=False)
    
    file_struct = FileStruct(audio_path)

    with open(file_struct.beat_file, 'r') as f:
        out_json = ujson.load(f)
    duration = float(out_json["globals"]["duration"])
    #duration = get_duration(FileStruct(audio_path).json_file)
    

    # Build hierarchy references
    for i in jam['annotations']['multi_segment']:
        bounds, labels = [], []
        for bound in i['data']:
            if bound.value['level'] == level:
                bounds.append(bound.time)
                labels.append(bound.value['label'])

    bounds = bounds + [duration]
    ref_int = utils.times_to_intervals(bounds)
    (ref_int, labels) = mir_eval.util.adjust_intervals(ref_int, labels, t_min=0, t_max=duration)
    bounds = utils.intervals_to_times(ref_int) 
    return bounds, labels#, duration


def get_duration(features_file):
    """Reads the duration of a given features file.

    Parameters
    ----------
    features_file: str
        Path to the JSON file containing the features.

    Returns
    -------
    dur: float
        Duration of the analyzed file.
    """
    with open(features_file) as f:
        feats = json.load(f)
    return float(feats["globals"]["duration"])


def write_beats(beat_times, file_struct, feat_config):

    # Construct a new JAMS object and annotation records
    
    # Save feature configuration in JSON file
    json_file = file_struct.json_file
    if json_file.exists() and os.path.isfile(json_file):
        with open(json_file, 'r') as f:
            out_json = ujson.load(f)
    else:
        json_file.parent.mkdir(parents=True, exist_ok=True)
        out_json = utils.create_json_metadata(file_struct.audio_file, 0, feat_config)
    out_json["est_beats"] = list(beat_times)
    with open(json_file, "w") as f:
        ujson.dump(out_json, f, indent=4)

def read_beats(json_file):
    with open(json_file, 'r') as f:
        out_json = ujson.load(f)
    beat_strings = out_json["est_beats"].split('[')[1].split(']')[0].split(',')
    duration = float(out_json["globals"]["duration"])
    beat_times = [int(i) for i in beat_strings]
    return beat_times, duration

def read_beats_harmonix(file_struct):
    df = pd.read_csv(file_struct.annot_beat_file, sep="\t", header=None) 
    df = np.array(df) 
    beat_times = df[:,0]
    beat_frames = librosa.time_to_frames(beat_times, sr=22050, hop_length=256)
    return beat_frames

def write_features(features, file_struct, feat_id, feat_config, beat_frames, duration=None):
    # Save actual feature file in .npy format
    feat_file = file_struct.get_feat_filename(feat_id)
    json_file = file_struct.json_file
    feat_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(feat_file, features)
    print('File saved')

    # Save feature configuration in JSON file
    if json_file.exists() and os.path.isfile(json_file):
        with open(json_file, 'r') as f:
            out_json = ujson.load(f)
    else:
        json_file.parent.mkdir(parents=True, exist_ok=True)
        out_json = utils.create_json_metadata(file_struct.audio_file, duration,
                                        feat_config)
    out_json[feat_id] = {}
    variables = vars(getattr(feat_config, feat_id))
    for var in variables:
        out_json[feat_id][var] = str(variables[var])
    with open(json_file, "w") as f:
        ujson.dump(out_json, f, indent=4)

    
    json_file = file_struct.beat_file
    if beat_frames != []:
        if json_file.exists():
            with open(json_file, 'r') as f:
                out_json = ujson.load(f)
        else:
            json_file.parent.mkdir(parents=True, exist_ok=True)
            out_json = utils.create_json_metadata(file_struct.audio_file, duration,
                                            feat_config)
        out_json['est_beats'] = json.dumps([int(i) for i in beat_frames])
        with open(json_file, "w") as f:
            ujson.dump(out_json, f, indent=4)


def write_beats(file_struct, feat_config, beat_frames, duration):
    json_file = file_struct.beat_file
    if json_file.exists():
        print('BEAT FILE EXISTS')
        with open(json_file, 'r') as f:
            out_json = ujson.load(f)
            #print(out_json['est_beats'])
    else:
        json_file.parent.mkdir(parents=True, exist_ok=True)
        out_json = create_json_metadata(file_struct.audio_file, duration,
                                        feat_config)
    out_json['est_beats'] = json.dumps([int(i) for i in beat_frames])
    with open(json_file, "w") as f:
        ujson.dump(out_json, f, indent=4)

    if json_file.exists():
        with open(json_file, 'r') as f:
            out_json = ujson.load(f)
            #print(out_json['est_beats'])
    



def create_json_metadata(audio_file, duration, feat_config):
    if duration is None:
        duration = -1
    out_json = collections.OrderedDict({"metadata": {
        "versions": {"librosa": librosa.__version__,
                     "numpy": np.__version__},
        "timestamp": datetime.datetime.today().strftime(
            "%Y/%m/%d %H:%M:%S")}})
    # Global parameters
    out_json["globals"] = {
        "duration": duration,
        "sample_rate": feat_config.sample_rate,
        "hop_length": feat_config.hop_length,
        "audio_file": str(audio_file.name)
        }
    return out_json
