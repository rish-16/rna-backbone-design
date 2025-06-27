"""
Code adapted from
https://github.com/microsoft/protein-frame-flow/blob/main/experiments/inference_se3_flows.py
"""

import os
import time
import numpy as np
import hydra
import torch
import GPUtil
from pytorch_lightning import Trainer
from omegaconf import DictConfig, OmegaConf

import rna_backbone_design.utils as eu
from rna_backbone_design.models.flow_module import FlowModule
from rna_backbone_design.data.pdb_na_dataset_base import LengthDataset
from rna_backbone_design.analysis.evalsuite import EvalSuite

torch.set_float32_matmul_precision('high')
log = eu.get_pylogger(__name__)

class Sampler:
    def __init__(self, cfg: DictConfig):
        """Initialize sampler.

        Args:
            cfg: inference config.
        """
        ckpt_path = cfg.inference.ckpt_path
        ckpt_dir = os.path.dirname(ckpt_path)
        ckpt_cfg = OmegaConf.load(os.path.join(ckpt_dir, 'config_flashipa.yaml'))

        # Set-up config.
        OmegaConf.set_struct(cfg, False)
        OmegaConf.set_struct(ckpt_cfg, False)
        cfg = OmegaConf.merge(cfg, ckpt_cfg)
        cfg.experiment.checkpointer.dirpath = './'

        self._cfg = cfg
        self._infer_cfg = cfg.inference
        self._samples_cfg = self._infer_cfg.samples
        self._rng = np.random.default_rng(self._infer_cfg.seed)

        # Set-up directories to write results to
        self._ckpt_name = '/'.join(ckpt_path.replace('.ckpt', '').split('/')[-3:])
        self._output_dir = os.path.join(
            self._infer_cfg.output_dir,
            self._infer_cfg.name,
        )
        os.makedirs(self._output_dir, exist_ok=True)
        log.info(f'Saving results to {self._output_dir}')
        config_path = os.path.join(self._output_dir, 'config_flashipa.yaml')
        with open(config_path, 'w') as f:
            OmegaConf.save(config=self._cfg, f=f)
        log.info(f'Saving inference config to {config_path}')

        # Read checkpoint and initialize module.
        self._flow_module = FlowModule.load_from_checkpoint(checkpoint_path=ckpt_path, cfg=cfg)
        
        self._flow_module.eval()
        self._flow_module._infer_cfg = self._infer_cfg
        self._flow_module._samples_cfg = self._samples_cfg
        self._flow_module._output_dir = self._output_dir

    def run_sampling(self):
        devices = GPUtil.getAvailable(order='memory', limit = 8)[:self._infer_cfg.num_gpus]
        log.info(f"Using devices: {devices}")
        
        eval_dataset = LengthDataset(self._samples_cfg)
        
        dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False, drop_last=False)
        
        trainer = Trainer(
            accelerator="gpu",
            strategy="ddp",
            devices=devices,
        )

        start_time = time.time()
        trainer.predict(self._flow_module, dataloaders=dataloader)
        elapsed_time = time.time() - start_time
        log.info(f'Finished in {elapsed_time:.2f}s')
        log.info(f'Generated samples are stored here: {self._cfg.inference.output_dir}/{self._cfg.inference.name}/')

@hydra.main(version_base=None, config_path="./camera_ready_ckpts", config_name="inference")
def run(cfg: DictConfig) -> None:

    # Read model checkpoint and run inference
    if cfg.inference.run_inference:
        log.info(f'Starting inference with {cfg.inference.num_gpus} GPUs')
        sampler = Sampler(cfg)
        sampler.run_sampling()

    # Run optional eval
    if cfg.inference.evalsuite.run_eval:
        print ("Starting EvalSuite on generated backbones ...")
        print (f"Sample directory: {cfg.inference.output_dir}/{cfg.inference.name}/")

        rna_bb_samples_dir = f"{cfg.inference.output_dir}/{cfg.inference.name}"
        saving_dir = cfg.inference.evalsuite.eval_save_dir
        
        # init evaluation module
        evalsuite = EvalSuite(
                    save_dir=saving_dir,
                    paths=cfg.inference.evalsuite.paths,
                    constants=cfg.inference.evalsuite.constants,
                    gpu_id1=0, # cuda:0 -> for inverse-folding model
                    gpu_id2=1,  # cuda:1 -> for forward-folding model
                )
        
        # run self-consistency pipeline
        metric_dict = evalsuite.perform_eval(
                                rna_bb_samples_dir,
                                flatten_dir=True
                            )

        # print out global self-consistency metrics
        metrics_fp = os.path.join(saving_dir, "final_metrics.pt")
        metric_dict = evalsuite.load_from_metric_dict(metrics_fp)
        evalsuite.print_metrics(metric_dict) # print eval metrics

if __name__ == '__main__':
    run()
