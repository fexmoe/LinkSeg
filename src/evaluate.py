# input: test data path, model weights path
# what it does: load model, evaluate on test data, returns results

import os
import torch
import logging
import argparse
import numpy as np
import pytorch_lightning as pl

from models import LinkSeg, FrameEncoder
from lightning_model import PLModel
from data_utils import *
from data_loader import ssm_dataloader
from predict_async import apply_async_with_callback_peaks


def inference(args):
    # data loading
    results = {}
    tracklist = clean_tracklist_audio(args.test_data_path, annotations=True)  
    test_dataloader = ssm_dataloader(split='valid', batch_size=1, tracklist=tracklist, max_len=args.max_len, n_embedding=args.n_embedding, hop_length=args.hop_length, num_workers=args.num_workers)

    lightning_model = PLModel.load_from_checkpoint(os.path.join(args.checkpoint_path, 'checkpoints', 'best_model.ckpt'))
    print('Module loaded!')

    num_gpus = torch.cuda.device_count()
    trainer = pl.Trainer(devices=num_gpus, accelerator="auto")

    predictions = trainer.test(lightning_model, dataloaders=test_dataloader)

    embeddings_list = lightning_model.embeddings_list
    bound_curves_list = lightning_model.bound_curves_list
    class_curves_list = lightning_model.class_curves_list
    acc_classes_list = lightning_model.val_acc_list
    print('Class pred:', np.mean(acc_classes_list), '+/-', np.std(acc_classes_list))
    tracklist = [i[0] for i in embeddings_list]

    dataset_name = os.path.basename(os.path.normpath(args.test_data_path))
    for i in range(len(tracklist)):
        track_name = tracklist[i]
        results[track_name] = {
                'Embeddings' : embeddings_list[i][1],
                'Bound Curve' : bound_curves_list[i],
                'Class Curve' : class_curves_list[i]
            }

    #out_peaks = apply_async_with_callback_peaks(bound_curves_list, class_curves_list, tracklist, 0, args.max_len)
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # trainer args
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--gradient_clip_val', type=float, default=.5)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--enable_progress_bar', type=int, default=1)
    parser.add_argument('--pre_trained_encoder', type=int, default=1)
    
    # lightning module args
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--random_seed', type=int, default=42)

    # stft parameters
    parser.add_argument('--n_mels', type=int, default=64)
    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--hop_length', type=int, default=256)
    parser.add_argument('--f_min', type=int, default=0)
    parser.add_argument('--f_max', type=int, default=11025)
    parser.add_argument('--sample_rate', type=int, default=22050)

    # input parameters
    parser.add_argument('--n_embedding', type=int, default=64)
    parser.add_argument('--max_len', type=int, default=1500)

    # frame encoder parameters
    parser.add_argument('--conv_ndim', type=int, default=32)
    parser.add_argument('--attention_ndim', type=int, default=32)
    parser.add_argument('--attention_nheads', type=int, default=8)
    parser.add_argument('--attention_nlayers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.1)

    # GNN parameters 
    parser.add_argument('--nb_ssm_classes', type=int, default=3)
    parser.add_argument('--nb_section_labels', type=int, default=7)
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--output_channels', type=int, default=16)
    parser.add_argument('--dropout_gnn', type=float, default=.1)
    parser.add_argument('--dropout_cnn', type=float, default=.2)
    parser.add_argument('--dropout_egat', type=float, default=.5)

    # paths
    parser.add_argument('--test_data_path', type=str)
    parser.add_argument('--checkpoint_path', type=str)

    args = parser.parse_args()

    print(args)
    inference(args)