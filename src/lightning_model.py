import torch
import logging
import numpy as np
import pytorch_lightning as pl
from predict_async import apply_async_with_callback_peaks
from torch.optim import lr_scheduler

from data_loader import ssm_dataloader
from data_utils import make_splits



class PLModel(pl.LightningModule):
    def __init__(
            self,
            network,
            loss_function,
            data_path,
            val_data_path,
            nb_section_labels,
            max_len,
            n_embedding,
            hop_length,
            learning_rate,
            optimizer_class,
            batch_size,
            num_workers,
            check_val_every_n_epoch,
            random_seed,
    ):
        super().__init__()
        self.network = network
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.optimizer_class = optimizer_class
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.data_path = data_path
        self.val_data_path = val_data_path
        self.nb_section_labels = nb_section_labels
        self.max_len = max_len
        self.n_embedding = n_embedding
        self.hop_length = hop_length
        self.random_seed = random_seed

        self.embeddings_list = []
        self.bound_curves_list = []
        self.class_curves_list = []
        self.class_gt_list = []
        self.valid_loss_list = []
        self.val_acc_list = []
        self.tracklist = []

        self.save_hyperparameters()
        logging.info('Building pytorch lightning model - done')

        self.train_split, self.val_split = make_splits(self.data_path, self.val_data_path, p=0.15, seed=self.random_seed)

    def train_dataloader(self):
        return self.get_dataloader(split='train', batch_size=self.batch_size, tracklist=self.train_split, max_len=self.max_len, n_embedding=self.n_embedding, hop_length=self.hop_length,nb_section_labels=self.nb_section_labels)
    
    def val_dataloader(self):
        return self.get_dataloader(split='valid', batch_size=self.batch_size, tracklist=self.val_split, max_len=self.max_len, n_embedding=self.n_embedding, hop_length=self.hop_length,nb_section_labels=self.nb_section_labels)
    
    def get_dataloader(self, split, batch_size, tracklist, max_len, n_embedding, hop_length, nb_section_labels):
        return ssm_dataloader(split=split, batch_size=batch_size, tracklist=tracklist, max_len=max_len, n_embedding=n_embedding, hop_length=hop_length, nb_section_labels=nb_section_labels, num_workers=self.num_workers)
        
    
    def training_step(self, batch, batch_idx):
        # get batch
        track, features, (SSM_segment, SSM_merged), labels = batch 
        SSM_segment = SSM_segment.squeeze().float()
        SSM_merged = SSM_merged.squeeze().float()
        labels = labels.squeeze(0).float()
        features = features.squeeze(0)

        # get predictions
        embeddings, bound_pred, labels_pred, A_pred = self.network.forward(features)
        embeddings = embeddings.squeeze()

        # get boundary ground-truth & loss
        bound_gt = 1-torch.diagonal(SSM_segment, offset=1)
        loss_bound = self.loss_function['dice_loss'](bound_pred, bound_gt)

        # get SSM ground-truth & loss
        ssm_gt = (SSM_segment + SSM_merged).long()
        loss_A_pred, acc_ssm = self.loss_function['class_loss_ssm'](A_pred, ssm_gt, balance=False)

        # mincut loss
        min_cut, ortho = self.loss_function['mincut_loss'](embeddings, labels_pred, SSM_merged) 
        loss_mincut = min_cut + ortho

        # get section labels ground-truth & loss
        labels_gt = torch.argmax(labels, -1)
        loss_classes, acc_classes = self.loss_function['class_loss_section'](labels_pred, labels_gt, balance=False)

        loss = loss_bound + loss_A_pred + loss_classes + loss_mincut 

        self.log('train_loss_step', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        # get batch
        track, features, (SSM_segment, SSM_merged), labels = batch 
        SSM_segment = SSM_segment.squeeze().float()
        SSM_merged = SSM_merged.squeeze().float()
        labels = labels.squeeze(0).float()
        features = features.squeeze(0)

        # get predictions
        embeddings, bound_pred, labels_pred, A_pred = self.network.forward(features)
        embeddings = embeddings.squeeze()

        # get boundary ground-truth & loss
        bound_gt = 1-torch.diagonal(SSM_segment, offset=1)
        loss_bound = self.loss_function['dice_loss'](bound_pred, bound_gt)

        # get SSM ground-truth & loss
        ssm_gt = (SSM_segment + SSM_merged).long()
        loss_A_pred, acc_ssm = self.loss_function['class_loss_ssm'](A_pred, ssm_gt, balance=False)

        # mincut loss
        min_cut, ortho = self.loss_function['mincut_loss'](embeddings, labels_pred, SSM_merged) 
        loss_mincut = min_cut + ortho

        # get section labels ground-truth & loss
        labels_gt = torch.argmax(labels, -1)
        loss_classes, acc_classes = self.loss_function['class_loss_section'](labels_pred, labels_gt, balance=False)

        loss = loss_bound + loss_A_pred + loss_classes + loss_mincut 
        self.log('valid_loss_step', loss)

        self.embeddings_list.append([track, embeddings.cpu().detach().numpy()])
        self.bound_curves_list.append([track[0], bound_pred.cpu().detach().numpy()])
        self.class_curves_list.append([labels_pred.cpu().detach().numpy()])
        self.class_gt_list.append(torch.argmax(labels, -1).cpu().detach().numpy())
        self.valid_loss_list.append(loss.item())
        self.val_acc_list.append(acc_classes)
        self.tracklist.append(track)


    def on_validation_epoch_end(self):
        out_peaks = apply_async_with_callback_peaks(self.bound_curves_list, self.class_curves_list, self.tracklist, 0, self.max_len, self.nb_section_labels)
        F1, F3, PFC, NCE = np.nanmean(out_peaks[:,2]), np.nanmean(out_peaks[:,5]), np.nanmean(out_peaks[:,6]), np.nanmean(out_peaks[:,7])
        self.log('valid_HRF', (F1+F3+PFC+NCE)/4, sync_dist=True)    
        self.log('valid_loss', np.mean(self.valid_loss_list), sync_dist=True)
        self.embeddings_list = []
        self.bound_curves_list = []
        self.class_curves_list = []
        self.class_gt_list = []
        self.valid_loss_list = []
        self.tracklist = []
        self.val_acc_list = []



    def test_step(self, batch, batch_idx):
        # get batch
        track, features = batch 
        features = features.squeeze(0)

        # get predictions
        embeddings, bound_pred, labels_pred, A_pred = self.network.forward(features)
        embeddings = embeddings.squeeze()

        self.embeddings_list.append([track, embeddings.cpu().detach().numpy()])
        self.bound_curves_list.append([track[0], bound_pred.cpu().detach().numpy()])
        self.class_curves_list.append([labels_pred.cpu().detach().numpy()])
        self.tracklist.append(track)


    def configure_optimizers(self):
     
        optimizer = self.optimizer_class(self.network.parameters(), lr=self.learning_rate, weight_decay=5e-5) 
        schedulers = {
            'scheduler': lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=10, factor=0.9, mode="max"),
            'monitor': 'valid_HRF', 
            'interval': 'epoch',
            'frequency': self.check_val_every_n_epoch,
        }
        return [optimizer], [schedulers]