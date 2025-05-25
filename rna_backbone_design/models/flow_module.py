"""
Code adapted from
https://github.com/microsoft/protein-frame-flow/blob/main/models/flow_module.py
"""

import torch
import time
import os
import random
import wandb
import numpy as np
import pandas as pd
import logging
from pytorch_lightning import LightningModule

from rna_backbone_design.analysis import metrics
from rna_backbone_design.analysis import utils as au
from rna_backbone_design.models.flow_model import FlowModel
from rna_backbone_design.models import utils as mu
from rna_backbone_design.data.interpolant import Interpolant 
from rna_backbone_design.data import utils as du
from rna_backbone_design.data import all_atom as rna_all_atom
from rna_backbone_design.data import so3_utils
from rna_backbone_design.data import nucleotide_constants
from rna_backbone_design.analysis import utils as au
from pytorch_lightning.loggers.wandb import WandbLogger

class FlowModule(LightningModule):
    def __init__(self, cfg, folding_cfg=None):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._interpolant_cfg = cfg.interpolant

        # Set-up vector field prediction model
        self.model = FlowModel(cfg.model)

        # Set-up interpolant
        self.interpolant = Interpolant(cfg.interpolant)

        self._sample_write_dir = self._exp_cfg.checkpointer.dirpath
        os.makedirs(self._sample_write_dir, exist_ok=True)

        self.validation_epoch_metrics = []
        self.validation_epoch_samples = []
        self.save_hyperparameters()
        
    def on_train_start(self):
        self._epoch_start_time = time.time()
        
    def on_train_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        self.log(
            'train/epoch_time_minutes',
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False
        )
        self._epoch_start_time = time.time()

    def model_step(self, noisy_batch):
        """
        Params:
            noisy_batch (dict) : dictionary of tensors corresponding to corrupted Frame objects

        Remarks:
            Computes the different core and auxiliary losses between ground truth and predicted backbones

        Returns:
            Dictionary of core and auxiliary losses
        """
        training_cfg = self._exp_cfg.training
        loss_mask = noisy_batch['res_mask']
        is_na_residue_mask = noisy_batch["is_na_residue_mask"]
        
        if training_cfg.min_plddt_mask is not None:
            plddt_mask = noisy_batch['res_plddt'] > training_cfg.min_plddt_mask
            loss_mask *= plddt_mask
        
        num_batch, num_res = loss_mask.shape

        torsions_start_index = 0
        torsions_end_index = 8
        num_torsions = torsions_end_index - torsions_start_index

        if training_cfg.num_non_frame_atoms == 0:
            bb_filtered_atom_idx = [2, 3, 6] # [C3', C4', O4']
        elif training_cfg.num_non_frame_atoms == 3:
            bb_filtered_atom_idx = [2, 3, 6] + [0, 7, 9] # [C3', C4', O4'] + [C1', O3', P]
        elif training_cfg.num_non_frame_atoms == 7:
            bb_filtered_atom_idx = [2, 3, 6] + [0, 4, 7, 9, 10, 11, 12] # [C3', C4', O4'] + [C1', C5', O3', P, OP1, OP2, N1]
        else:
            # NOTE: default is the original frame
            bb_filtered_atom_idx = [2, 3, 6] # [C3', C4', O4']
        
        n_merged_atoms = len(bb_filtered_atom_idx)

        # Ground truth labels
        gt_trans_1 = noisy_batch['trans_1']
        gt_rotmats_1 = noisy_batch['rotmats_1']
        gt_torsions_1 = noisy_batch['torsion_angles_sin_cos'][:, :, torsions_start_index:torsions_end_index, :].reshape(num_batch, num_res, num_torsions * 2)
        rotmats_t = noisy_batch['rotmats_t']
        gt_rot_vf = so3_utils.calc_rot_vf(rotmats_t, gt_rotmats_1.type(torch.float32))

        gt_bb_atoms = rna_all_atom.to_atom37_rna(
                            gt_trans_1, gt_rotmats_1, 
                            torch.ones_like(is_na_residue_mask),
                            torsions=gt_torsions_1
                        )
        gt_bb_atoms = gt_bb_atoms[:, :, bb_filtered_atom_idx]

        # Timestep used for normalization.
        t = noisy_batch['t']
        norm_scale = 1 - torch.min(t[..., None], torch.tensor(training_cfg.t_normalize_clip))
        
        # Model output predictions.
        model_output = self.model(noisy_batch)
        pred_trans_1 = model_output['pred_trans']
        pred_rotmats_1 = model_output['pred_rotmats']
        pred_torsions_1 = model_output['pred_torsions'].reshape(num_batch, num_res, num_torsions * 2)
        pred_rots_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)

        # Backbone atom loss
        pred_bb_atoms = rna_all_atom.to_atom37_rna(
                            pred_trans_1, pred_rotmats_1, 
                            torch.ones_like(is_na_residue_mask),
                            torsions=pred_torsions_1
                        )
        pred_bb_atoms = pred_bb_atoms[:, :, bb_filtered_atom_idx]
        
        gt_bb_atoms *= training_cfg.bb_atom_scale / norm_scale[..., None]
        pred_bb_atoms *= training_cfg.bb_atom_scale / norm_scale[..., None]
        loss_denom = torch.sum(loss_mask, dim=-1) * n_merged_atoms
        bb_atom_loss = torch.sum((gt_bb_atoms - pred_bb_atoms) ** 2 * loss_mask[..., None, None], dim=(-1, -2, -3)) / loss_denom

        # Translation VF loss
        trans_error = (gt_trans_1 - pred_trans_1) / norm_scale * training_cfg.trans_scale
        loss_denom = torch.sum(loss_mask, dim=-1) * 3  # 3 frame atoms
        trans_loss = training_cfg.translation_loss_weight * torch.sum(
            trans_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom

        # Rotation VF loss
        rots_vf_error = (gt_rot_vf - pred_rots_vf) / norm_scale
        loss_denom = torch.sum(loss_mask, dim=-1) * 3  # 3 frame atoms
        rots_vf_loss = training_cfg.rotation_loss_weights * torch.sum(
            rots_vf_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom

        # Pairwise distance loss
        gt_flat_atoms = gt_bb_atoms.reshape([num_batch, num_res * n_merged_atoms, 3]) 
        gt_pair_dists = torch.linalg.norm(gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
        pred_flat_atoms = pred_bb_atoms.reshape([num_batch, num_res * n_merged_atoms, 3]) 
        pred_pair_dists = torch.linalg.norm(pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, n_merged_atoms))
        flat_loss_mask = flat_loss_mask.reshape([num_batch, num_res * n_merged_atoms])
        flat_res_mask = torch.tile(loss_mask[:, :, None], (1, 1, n_merged_atoms))
        flat_res_mask = flat_res_mask.reshape([num_batch, num_res * n_merged_atoms]) 

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        dist_mat_loss = torch.sum((gt_pair_dists - pred_pair_dists)**2 * pair_dist_mask, dim=(1, 2))
        dist_mat_loss /= (torch.sum(pair_dist_mask, dim=(1, 2)) - num_res)

        # Torsion angles loss
        pred_torsions_1 = pred_torsions_1.reshape(num_batch, num_res, num_torsions, 2)
        gt_torsions_1 = gt_torsions_1.reshape(num_batch, num_res, num_torsions, 2)
        loss_denom = torch.sum(loss_mask, dim=-1) * 8  # 8 torsion angles
        tors_loss = training_cfg.tors_loss_scale * torch.sum(
            torch.linalg.norm(pred_torsions_1 - gt_torsions_1, dim=-1) ** 2 * loss_mask[..., None], dim=(-1, -2)
        ) / loss_denom

        assert bb_atom_loss.shape[0] == dist_mat_loss.shape[0] == tors_loss.shape[0], f"Loss tensors shape mismatch: {bb_atom_loss.shape} vs {dist_mat_loss.shape} vs {tors_loss.shape}"
        
        se3_vf_loss = trans_loss + rots_vf_loss
        auxiliary_loss = (bb_atom_loss + dist_mat_loss + tors_loss) * (t[:, 0] > training_cfg.aux_loss_t_pass)
        auxiliary_loss *= self._exp_cfg.training.aux_loss_weight

        if torch.isnan(auxiliary_loss).any():
            print ("NaN loss in aux_loss")
            auxiliary_loss = torch.zeros_like(auxiliary_loss).to(se3_vf_loss.device)
        
        if torch.isnan(se3_vf_loss).any():
            # raise ValueError('NaN loss encountered')
            print ("NaN loss in se3_vf_loss")
            se3_vf_loss = torch.zeros_like(se3_vf_loss).to(se3_vf_loss.device)
        
        return {
            "bb_atom_loss": bb_atom_loss,
            "trans_loss": trans_loss,
            "dist_mat_loss": dist_mat_loss,
            "auxiliary_loss": auxiliary_loss,
            "rots_vf_loss": rots_vf_loss,
            "se3_vf_loss": se3_vf_loss,
            "torsion_loss": tors_loss
        }

    def validation_step(self, batch, batch_idx):
        """
        Generates samples and computes (average) local structural metrics 
        such as C4'-C4' distances, steric clashes, gyration radius.
        """

        res_mask = batch['res_mask']
        is_na_residue_mask = batch['is_na_residue_mask'].bool()
        self.interpolant.set_device(res_mask.device)
        num_batch = res_mask.shape[0]
        num_res = is_na_residue_mask.sum(dim=-1).max().item()
        
        samples = self.interpolant.sample(
            num_batch,
            num_res,
            self.model,
        )[0][-1].numpy()

        batch_metrics = []

        for i in range(num_batch):
            final_pos = samples[i]
            saved_rna_path = au.write_complex_to_pdbs( # Save RNA atoms to PDB
                final_pos,
                os.path.join(self._sample_write_dir, f'sample_{i}_idx_{batch_idx}_len_{num_res}.pdb'),
                is_na_residue_mask=is_na_residue_mask.detach().cpu().numpy()[i],
                no_indexing=True
            )
            if isinstance(self.logger, WandbLogger):
                self.validation_epoch_samples.append(
                    [saved_rna_path, self.global_step, wandb.Molecule(saved_rna_path)]
                )

            c4_idx = nucleotide_constants.atom_order["C4\'"]
            rna_c4_c4_matrics = metrics.calc_rna_c4_c4_metrics(final_pos[:, c4_idx])
            batch_metrics.append(rna_c4_c4_matrics)

        batch_metrics = pd.DataFrame(batch_metrics)
        self.validation_epoch_metrics.append(batch_metrics)
        
    def on_validation_epoch_end(self):
        if len(self.validation_epoch_samples) > 0:
            self.logger.log_table(
                key='valid/samples',
                columns=["sample_path", "global_step", "RNA"],
                data=self.validation_epoch_samples)
            self.validation_epoch_samples.clear()

        val_epoch_metrics = pd.concat(self.validation_epoch_metrics)
        
        for metric_name, metric_val in val_epoch_metrics.mean().to_dict().items():
            self._log_scalar(
                f'valid/{metric_name}',
                metric_val,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=len(val_epoch_metrics),
            )
        self.validation_epoch_metrics.clear()

    def _log_scalar(
            self,
            key,
            value,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=None,
            sync_dist=False,
            rank_zero_only=True
        ):
        if sync_dist and rank_zero_only:
            raise ValueError('Unable to sync dist when rank_zero_only=True')
        self.log(
            key,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only
        )

    def training_step(self, batch, stage):
        """
        Performs one iteration of SE(3) flow matching and returns total training loss
        using the core and auxiliary losses computed in `model_step`.
        """
        step_start_time = time.time()
        self.interpolant.set_device(batch['res_mask'].device)
        noisy_batch = self.interpolant.corrupt_batch(batch)

        if self._interpolant_cfg.self_condition and random.random() > 0.5:
            with torch.no_grad():
                model_sc = self.model(noisy_batch)
                noisy_batch['trans_sc'] = model_sc['pred_trans']
        
        batch_losses = self.model_step(noisy_batch)
        num_batch = batch_losses['bb_atom_loss'].shape[0]
        total_losses = {
            k: torch.mean(v) for k,v in batch_losses.items()
        }
        
        for k,v in total_losses.items():
            self._log_scalar(f"train/{k}", v, prog_bar=False, batch_size=num_batch)
        
        # Losses to track. Stratified across t.
        t = torch.squeeze(noisy_batch['t'])
        self._log_scalar(
            "train/t",
            np.mean(du.to_numpy(t)),
            prog_bar=False, batch_size=num_batch)
        for loss_name, loss_dict in batch_losses.items():
            stratified_losses = mu.t_stratified_loss(
                t, loss_dict, loss_name=loss_name)
            for k,v in stratified_losses.items():
                self._log_scalar(f"train/{k}", v, prog_bar=False, batch_size=num_batch)

        # Training throughput
        self._log_scalar("train/length", batch['res_mask'].shape[1], prog_bar=False, batch_size=num_batch)
        self._log_scalar("train/batch_size", num_batch, prog_bar=False)
        
        step_time = time.time() - step_start_time
        self._log_scalar("train/eps", num_batch / step_time)
        
        train_loss = (
            total_losses[self._exp_cfg.training.loss] +
            total_losses['auxiliary_loss']
        )
        self._log_scalar("train/loss", train_loss, batch_size=num_batch)

        return train_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.model.parameters(),
            **self._exp_cfg.optimizer
        )

    def predict_step(self, batch):
        """
        Params:
            batch (dict) : dictionary of number of RNA samples and respective number of RNA nucleotides to model

        Remarks:
            Used for unconditional sampling from the Interpolant. 
            Saves the generated all-atom backbones as PDB files at the specified saving directory in the inference config file.
        """

        device = f'cuda:{torch.cuda.current_device()}'
        interpolant = Interpolant(self._infer_cfg.interpolant) 
        interpolant.set_device(device)

        sample_length = batch['num_res'].item()
        diffuse_mask = torch.ones(1, sample_length).long().unsqueeze(0)
        sample_id = batch['sample_id'].item()
        sample_dir = os.path.join(self._output_dir, f'length_{sample_length}')
        os.makedirs(sample_dir, exist_ok=True)

        atom37_traj, _, _ = interpolant.sample(1, sample_length, self.model)
        bb_traj = du.to_numpy(torch.concat(atom37_traj, dim=0)) # convert from torch to numpy for easier processing

        sample = bb_traj[-1] # store last state (at the final ODESolve timestep) as completed sample
        
        # store RNA backbone sample as PDB file at `saved_rna_path` specified above
        saved_rna_path = au.write_complex_to_pdbs(
                sample,
                os.path.join(sample_dir, f'sample_{sample_id}'),
                is_na_residue_mask=diffuse_mask.detach().cpu(),
            )
        
        saved_rna_traj_path = au.write_complex_to_pdbs(
                bb_traj,
                os.path.join(sample_dir, f'sample_{sample_id}'),
                is_na_residue_mask=diffuse_mask.detach().cpu(),
            )