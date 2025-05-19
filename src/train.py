import os
import torch
import logging
import argparse
import numpy as np
import pytorch_lightning as pl
from collections import OrderedDict

from torch import nn
from models import LinkSeg, FrameEncoder
from lightning_model import PLModel
from callback_loggers import get_loggers, get_callbacks
from training_utils import DirManager
from losses import get_losses


def train(args):
    dir_manager = DirManager(output_dir=args.output_dir)

    # model
    _frame_encoder = FrameEncoder(
                n_mels=args.n_mels,
                conv_ndim=args.conv_ndim,
                sample_rate=args.sample_rate,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                n_embedding=args.n_embedding,
                f_min=args.f_min,
                f_max=args.f_max,
                dropout=args.dropout,
                hidden_dim=args.hidden_dim,
                attention_ndim=args.attention_ndim,
                attention_nlayers=args.attention_nlayers,
                attention_nheads=args.attention_nheads,
    )


    if args.pre_trained_encoder:
        old_state_dict = torch.load('../data/backbone_repetition.pt', map_location=torch.device('cpu'))
        new_state_dict = OrderedDict()
        for k, v in old_state_dict.items():
            name = k.split('network.')[-1]
            new_state_dict[name] = v
        _frame_encoder.load_state_dict(new_state_dict, strict=True) 
        print('Pre-trained frame encoder loaded !')
        

    _network = LinkSeg(
                _frame_encoder, 
                nb_ssm_classes=args.nb_ssm_classes, 
                nb_section_labels=args.nb_section_labels, 
                hidden_size=args.hidden_size, 
                output_channels=args.output_channels,
                dropout_gnn=args.dropout_gnn,
                dropout_cnn=args.dropout_cnn,
                dropout_egat=args.dropout_egat,
                max_len=args.max_len,
    )

    print(_network)

    # count gpus
    num_gpus = torch.cuda.device_count()

    # callbacks
    callbacks = get_callbacks(patience=30, dir_manager=dir_manager, monitor='valid_HRF', mode='max')
    #checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=args.output_dir, filename='{epoch}-{valid_HRF:.2f}', save_top_k=1, monitor="valid_HRF", mode="max", verbose=True, enable_version_counter=False) 
    #loggers = get_loggers(tb_save_dir=dir_manager.tensorboard_dir)

    # loss functions
    losses_dict = get_losses()

    # lightning module
    model = PLModel(
        network=_network,
        loss_function=losses_dict,
        data_path=args.data_path,
        val_data_path=args.val_data_path,
        nb_section_labels = args.nb_section_labels,
        max_len=args.max_len,
        n_embedding=args.n_embedding,
        hop_length=args.hop_length,
        learning_rate=args.learning_rate,
        optimizer_class=torch.optim.AdamW,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        random_seed=args.random_seed,
    )  

    # trainer
    trainer = pl.Trainer(
        devices=num_gpus,
        num_nodes=args.num_nodes,
        callbacks=callbacks,
        max_epochs=args.max_epochs,
        enable_model_summary=True,
        num_sanity_val_steps=0,
        accelerator="auto",
        gradient_clip_val=args.gradient_clip_val,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        accumulate_grad_batches=args.accumulate_grad_batches,
        enable_progress_bar=args.enable_progress_bar,
        logger=False
    )

    trainer.fit(model)
    logging.info('Training completed.')

    # load the best model and save it 
    print('Loading best model and saving it')
    model = PLModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    print('best model path : ',trainer.checkpoint_callback.best_model_path)
    print('dir_manager.best_model_statedict', dir_manager.best_model_statedict)
    torch.save(model.state_dict(), dir_manager.best_model_statedict)



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
    #parser.add_argument('--data_path', type=str, default='./../data/')
    parser.add_argument('--data_path', nargs="+", type=str, default=[])
    parser.add_argument('--val_data_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./results/exp')

    args = parser.parse_args()

    print(args)
    train(args)