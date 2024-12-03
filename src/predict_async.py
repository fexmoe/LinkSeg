import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")
from data_utils import *
import librosa
import mir_eval
import ujson
from scipy.stats import mode

metrics = ['Precision@0.5', 'Recall@0.5', 'F-measure@0.5', 'Precision@3.0', 'Recall@3.0', 'F-measure@3.0', 'Pairwise F-measure', 'NCE F-measure']

def apply_async_with_callback_peaks(bound_curves_list, class_curves_list, tracklist, level, max_len, nb_section_labels, return_tracklist=False):
    print('eval_segmentation_peaks')
    jobs = [ joblib.delayed(eval_segmentation_peak)(i[0], i[1], j[0], level, max_len, nb_section_labels) for i,j in zip(bound_curves_list, class_curves_list) ]
    out = joblib.Parallel(n_jobs=32, verbose=1)(jobs)
    scores = [i[1] for i in out]
    tracklist = [i[0] for i in out]
    out_scores = np.zeros((len(scores), len(metrics)))
    for i in range(len(scores)):
        for j in range(len(metrics)):
            out_scores[i,j] = scores[i][metrics[j]]
    out = out_scores
    print('Results on annotation level =', level, ', whole embedding matrix')
    print('P1 =', np.nanmean(out[:,0]),'+/-', np.nanstd(out[:,0]))
    print('R1 =', np.nanmean(out[:,1]),'+/-', np.nanstd(out[:,1]))
    print('F1 =', np.nanmean(out[:,2]),'+/-', np.nanstd(out[:,2]))
    print('P3 =', np.nanmean(out[:,3]),'+/-', np.nanstd(out[:,3]))
    print('R3 =', np.nanmean(out[:,4]),'+/-', np.nanstd(out[:,4]))
    print('F3 =', np.nanmean(out[:,5]),'+/-', np.nanstd(out[:,5]))
    print('PFC =', np.nanmean(out[:,6]),'+/-', np.nanstd(out[:,6]))
    print('NCE =', np.nanmean(out[:,7]),'+/-', np.nanstd(out[:,7]))

    if return_tracklist:
        return out, tracklist
    else:
        return out
    


def eval_segmentation_peak(audio_file, bound_curve, class_curve, level, max_len, nb_section_labels):

    file_struct = FileStruct(audio_file)
    ref_labels, ref_times, duration = get_ref_labels(file_struct, level, 0)

    ref_inter = times_to_intervals(ref_times)
    (ref_inter, ref_labels) = mir_eval.util.adjust_intervals(ref_inter, list(ref_labels), t_min=0, t_max=ref_inter.max())
    
    ref_times = intervals_to_times(ref_inter)
    if nb_section_labels == 7:
        ref_labels = merge_labels(ref_labels, indices, substrings)
    elif nb_section_labels == 9:
        ref_labels = merge_labels(ref_labels, indices_9classes, substrings_9classes)
    ref_inter = times_to_intervals(ref_times)
    beat_frames, duration = read_beats(file_struct.beat_file)
    beat_frames = librosa.util.fix_frames(beat_frames)

    if len(beat_frames)>max_len:
        beat_frames = beat_frames[::2]
    
    beat_times = librosa.frames_to_time(beat_frames, sr=22050, hop_length=256)

    beat_times = [(beat_times[i] + beat_times[i+1])/2 for i in range(len(beat_times)-1)]

    assert len(beat_times) == len(bound_curve)

    
    bound_curve = bound_curve.reshape(-1)

    
    
    est_idxs = pick_peaks_times(bound_curve, beat_times, avg_future=12, avg_past=12, max_future=6, max_past=6, tau=0)
    
        
    if len(est_idxs)>0:
        if est_idxs[0] != 0:

            est_idxs_ = [0] + list(est_idxs)
    else:
        est_idxs_ = [0] 

    est_idxs_ = est_idxs_ + [len(beat_times)-1]
    est_labels = []
    
    for i in range(len(est_idxs_)-1):
        bound_left = est_idxs_[i]
        bound_right = est_idxs_[i+1]
        class_predictions = np.argmax(class_curve[bound_left:bound_right], axis=-1)
        est_labels.append(mode(class_predictions)[0])

    
    
    est_idxs = [0] + [beat_times[int(i)] for i in est_idxs] + [duration]
    
    est_inter = times_to_intervals(est_idxs)

    scores = mir_eval.segment.evaluate(ref_inter, ref_labels, est_inter, est_labels, trim=True)
    
    return audio_file, scores
    

def get_indices(beat_times, index, avg_future=6, avg_past=12):
    limit_left = 0
    limit_right = len(beat_times)-1
    for i in range(index, 0, -1):
        if beat_times[index]-beat_times[i]>avg_future:
            limit_left = i-1
            break
    for i in range(index, len(beat_times)):
        if beat_times[i]-beat_times[index]>avg_past:
            limit_right = i-1
            break
    return limit_left, limit_right



def pick_peaks_times(nc, beat_times, avg_future=12, avg_past=12, max_future=12, max_past=12, tau=0):
    
    peaks = []
    for i in range(1, nc.shape[0] - 1):
        limit_left_max, limit_right_max = get_indices(beat_times, i, max_future, max_past)
        try:
            max_left = np.max([nc[j] for j in range(i-1, limit_left_max, -1)])
        except:
            max_left = 0
        try:
            max_right = np.max([nc[j] for j in range(i+1, limit_right_max)])
        except:
            max_right = 0
        if max_left < nc[i] and nc[i] > max_right:
            limit_left, limit_right = get_indices(beat_times, i, avg_future, avg_past)
            mean_left = np.mean([nc[j] for j in range(i, limit_left, -1)])
            mean_right = np.mean([nc[j] for j in range(i, limit_right)])
            if nc[i] > mean_left and nc[i] > mean_right and nc[i]>tau:
                peaks.append(i)
    return peaks


