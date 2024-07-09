import os
import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
import torchaudio
import librosa
import numpy as np

from tqdm import tqdm

from loader import load_model
from model import LinkSeg
#from utils import export 
from pre_processing import pre_process
from post_processing import post_process, export_to_jams
from input_output import read_beats
from configuration import config



def _predict(x: torch.Tensor,
             model: LinkSeg) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return model(x)


def predict(x: torch.Tensor,
            config: dict=config,
            model_name: str = "Harmonix_full",
            inference_mode: bool = True,
            no_grad: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Main prediction function.

    Args:
        x (torch.Tensor): input audio tensor, can be provided as a batch but should be mono,
            shape (num_samples) or (batch_size, num_samples)
        model_name: name of LinkSeg model. Can be a path to a custom LinkSeg checkpoint or the name of a standard model.
        inference_mode (bool): whether to run with `torch.inference_mode`.
        no_grad (bool): whether to run with `torch.no_grad`. If set to `False`, argument `inference_mode` is ignored.

    Returns:
        embeddings (torch.Tensor): embedding vectors for each time-step (beat) in the input track, shape (N, embed_dim)
        bound_curve (torch.Tensor): boundary probability curve for each time-step, shape (N, 1)
        class_curves (torch.Tensor): class activation curves for each time-step, shape (N, n_classes)
        A_pred (torch.Tensor): predicted self-similarity matrix from link feature extractor, shape (N, N, 3)
    """
    inference_mode = inference_mode and no_grad
    with torch.no_grad() if no_grad and not inference_mode else torch.inference_mode(mode=inference_mode):
        model = load_model(model_name, config).to(x.device)

        return _predict(x, model)
    

def predict_from_files(
        audio_files: Union[str, Sequence[str]],
        config: dict=config,
        model_name: str = "Harmonix_full",
        output: Optional[str] = None,
        export_format: Sequence[str] = ("jams",),
        gpu: int = -1):
    r"""

    Args:
        audio_files: audio files to process
        config: config file
        model_name: name of the model. Currently only `Harmonix_full` is supported.
        output:
        export_format (Sequence[str]): format to export the predictions to.
            Currently format supported is: ["jams"].
        gpu: index of GPU to use (-1 for CPU)
    """
    if isinstance(audio_files, str):
        audio_files = [audio_files]

    if gpu >= 0 and not torch.cuda.is_available():
        warnings.warn("You're trying to use the GPU but no GPU has been found. Using CPU instead...")
        gpu = -1
    device = torch.device(f"cuda:{gpu:d}" if gpu >= 0 else "cpu")

    # define model
    model = load_model(model_name, config).to(device)

    pbar = tqdm(audio_files)

    with torch.inference_mode():  # here the purpose is to write results in disk, so there is no point storing gradients
        for file in pbar:
            pbar.set_description(file)
            file_struct = pre_process(file, config)
            # load audio file
            try:
                
                beat_frames, duration = read_beats(file_struct.beat_file)
                beat_frames = librosa.util.fix_frames(beat_frames)
                beat_times = librosa.frames_to_time(beat_frames, sr=22050, hop_length=256)
                beat_frames = librosa.time_to_frames(beat_times, sr=22050, hop_length=1)
                pad_width = ((config.hop_length*config.n_embedding) - 2)//2 
                waveform, sr = torchaudio.load(file_struct.audio_file)
                if sr != config.sample_rate:
                    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=config.sample_rate)
                waveform = waveform.mean(dim=0)
                features_padded = np.pad(waveform, pad_width=((pad_width, pad_width)), mode='edge')
                features = np.stack([features_padded[i:i+pad_width*2] for i in beat_frames], axis=0)
                x = torch.tensor(features)

            except Exception as e:
                print(e, f"Skipping {file}...")
                continue

            # compute the predictions
            embeddings, bound_curve, class_curves, A_pred = _predict(x, model=model)

            # post-process predictions (peak picking & majority vote)
            est_times, est_labels = post_process(file, beat_times, duration, bound_curve, class_curves)

            # write predictions to jams format
            print(est_times, est_labels)
            
            export_to_jams(file_struct, duration, est_times, est_labels)
            # output_file = os.path.join(output, os.path.basename(output_file))#

            #predictions = [p.cpu().numpy() for p in predictions]
            #for fmt in export_format:
            #    export(fmt, output_file, *predictions)