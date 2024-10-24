"""
Some helper methods for angle computation have been taken from `graphein`:
https://github.com/a-r-j/graphein/blob/master/graphein/protein/tensor/angles.py#L448
Credits: Arian R. Jamasb
"""

import os, gc, json, torch
from tqdm import tqdm
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis import rms
import torch.nn.functional as F

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# external tools (gRNAde, RhoFold)
from src.tools.grnade_api.gRNAde import gRNAde
from src.tools.rhofold_api.rhofold.rf import RhoFold
from src.tools.rhofold_api.rhofold.config import rhofold_config
from src.tools.rhofold_api.rhofold.utils.alphabet import get_features
from src.analysis import metrics

class EvalSuite:
    def __init__(self, 
            save_dir,
            paths=None,
            constants=None,
            gpu_id1=0,
            gpu_id2=1,
            use_invfold=True,
            load_models=True,
        ):

        """
        Params:
            save_dir (str) : target directory to save temp files and metrics in torch's .pt format
            
            paths (cfg) : sub-config containining the following paths:
                - gt_dir (str) : directory containing the ground truth samples in PDB format (to compute novelty)
                - usalign_path (str) : filepath to ./USalign to compute TM-scores between generated samples and ground truth samples (novelty)
                - qtmclust_path (str) : filepath to ./qTMclust to compute TM-scores among generated samples (diversity)
                - tmscore_path (str) : filepath to ./TMscore to compute TM-scores between generated samples and predicted structures from structure predictor (smTM)
                - metadata_path (str) : filepath to rna_metadata.csv for bookkeeping

            constants (cfg) : sub-config containing the following constants:
                - metadata_len_filter (list[int, int]) : min-max sequence lengths to compare generated samples to
                - tm_thresh (float) : threshold in [0, 1] to consider two structures similar using TM-score
                - rmsd_thresh (float) : maximum RMSD between generated sample and predicted structure to be considered designable
            
            gpu_id1 (int) : GPU index for inverse folding model (gRNAde)
            gpu_id2 (int) : GPU index for structure prediction model (RhoFold)
            
            use_invfold (bool) : flag to decide whether to inverse fold the backbones or rely on existing sequence inside the sample's PDB file
            load_models (bool) : flag to decide whether to load models to GPU or not (in case you want to use some of the helper functions without overhead from loading)

        Files/folders saved locally:
            - fwdfold_pdbs/ : directory containing all forward-folded samples in PDB format
            - invfold_seqs/ : directory containing all inverse-folded samples in FASTA format
            - final_metrics.pt : pytorch file of all global+local metrics computed throughout EvalSuite run
            - final_metrics_local.pt : pytorch file containing all local metrics (stored in case global metrics fail)
            - final_metrics_global.pt : pytorch file containing all global metrics (stored in case local metrics fail)
            - gen_list.list : text file containing names of all generated samples from RNA-FrameFlow
            - gt_list.list : text file containing names of all training samples within length range (here, [40,150]) to compare generated samples to
            - MASTER_QTM_RESULTS.txt : text file (tsv) containing qTMclust module outputs
            - MASTER_USALIGN_RESULTS.txt : text file (tsv) containing US-align module outputs
            - TMSCORE_TMP_RESULTS.txt: temporary text file containing computed TM-scores using TMscore module outputs
        """

        os.makedirs(save_dir, exist_ok=True)
        print (f"Created savedir: {save_dir}")

        grnade_save_path = os.path.join(save_dir, "invfold_seqs")
        rhofold_save_path = os.path.join(save_dir, "fwdfold_pdbs")
        tmp_save_dict = save_dir
        
        self.save_dir = save_dir
        self.use_invfold = use_invfold
        
        self.usalign_path = paths.usalign_path
        self.qtmclust_path = paths.qtmclust_path
        self.tmscore_path = paths.tmscore_path
        self.metadata_path = paths.rnasolo_metadata_path
        self.gt_dir = paths.rnasolo_path

        self.tm_thresh = constants.tm_thresh # threshold for qTMclust
        self.rmsd_thresh = constants.rmsd_thresh # threshold for scRMSD
        self.metadata_len_filter = constants.seqlen_range

        # temp files
        self.grnade_save_path = grnade_save_path
        self.rhofold_save_path = rhofold_save_path
        self.tmp_save_dict = tmp_save_dict

        # tools
        if load_models:
            self.gRNAde_module = gRNAde(split='all', max_num_conformers=1, gpu_id=gpu_id1)
            self.rhofold = RhoFold(rhofold_config)
            
            RHOFOLD_CKPT = "./src/tools/rhofold_api/checkpoints/RhoFold_pretrained.pt"
            self.rhofold.load_state_dict(torch.load(RHOFOLD_CKPT, map_location=torch.device('cpu'))['model'])
            self.rhofold.eval()
            self.device = f"cuda:{gpu_id2}"
            self.rhofold = self.rhofold.to(self.device)

        if not os.path.exists(grnade_save_path):
            print (f"Creating directory for gRNAde generated sequences: {grnade_save_path}")
            os.mkdir(grnade_save_path)

        if not os.path.exists(rhofold_save_path):
            print (f"Creating directory for RhoFold predicted structures: {rhofold_save_path}")
            os.mkdir(rhofold_save_path)

    def perform_eval(self, gen_dir, flatten_dir=False, plot=False, exclude=[], source=None):
        """
        Params:
            - gen_dir : path of the generated samples (inference or validation)
                -> This directory must be 1-dimensional without any subdirectories. 
                   Ensure the PDB files are named uniquely to avoid overwriting them.
            - flatten_dir : boolean to flatten directory in case samples are separated by sequence length (default: False). 
                -> If set to True, if `gen_dir` has subdirectories based on length, a new
                   directory called "flattened_samples/" is created which contains all the samples
                   with a new naming convention (seqlen_na_samples_idx.pdb)
            - exclude : list of strings corresponding to which evals not to perform
                -> pick from ["struct_dihedrals", "struct_bond_angles", "struct_bond_dist", "batch_clusters", "batch_pdbTM"]
            - plot : boolean to plot histograms of sturctural metrics (eg: distance, angles, dihedrals)

        Notes:
            structural metrics computed:
                1. Histogram of adjacent C4' bond distances averaged over entire backbone
                2. Histogram of 8 torsion angles averaged over all residues

            batch metrics computed:
                1. diversity percent (using qTMclust)
                2. novelty percent (using USAlign)
                3. designability percent (using gRNAde + E2EFold/RhoFold)

        - This is done regardless of sequence length. There is no extra bucketing for length.
        - Only run this method, do not use any of the other helper methods with a `_compute_xyz()` name
            - Use the `exclude` param if want to ignore specific evaluations
                
        Returns:
            dictionary of all metrics
        """

        structural_eval_dict = {
            "struct_dihedrals": self._compute_dihedral_angles,
            "struct_bond_angles": self._compute_bond_angles,
            "struct_bond_dist": self._compute_c4_to_c4_distance,
        }

        batch_eval_dict = {
            "batch_clusters": self._compute_qTMclust_metrics, # uses gen_dir only
            "batch_pdbTM": self._compute_usalign_metrics # uses gen_dir and self.gt_dir
        }

        all_keys = list(structural_eval_dict.keys()) + list(batch_eval_dict.keys())
        final_metric_dict = {key:[] for key in all_keys}

        if flatten_dir:
            """
            To be used if `gen_dir` looks like this:
            |- gen_dir
            |--- length_a/
            |----- na_sample_1.pdb
            |----- na_sample_2.pdb
            |----- na_sample_3.pdb
            |--- length_b/
            |----- na_sample_1.pdb
            |----- na_sample_2.pdb
            |----- na_sample_3.pdb
            |--- length_c/
            |----- na_sample_1.pdb
            |----- na_sample_2.pdb
            |----- na_sample_3.pdb

            The final result will be:

            |- flattened_samples/ # copied + renamed files
            |--- a_na_sample_1.pdb
            |--- a_na_sample_2.pdb
            |--- a_na_sample_3.pdb
            |--- b_na_sample_1.pdb
            |--- b_na_sample_2.pdb
            |--- b_na_sample_3.pdb            
            |--- c_na_sample_1.pdb
            |--- c_na_sample_2.pdb
            |--- c_na_sample_3.pdb   
            |- gen_dir/ # originals not affected
            |--- length_a/
            |--- length_b/
            |--- length_c/
            """
            print ("Flattening directory ...")
            FWD_PATH = f"{gen_dir}/flattened_samples/" # assume `gen_dir` and `flattened_samples` are at the same level
            self._flatten_dir(gen_dir, FWD_PATH)
            gen_dir = FWD_PATH # set new location as `gen_dir`

        os.makedirs(self.save_dir, exist_ok=True)

        metadata_df = pd.read_csv(self.metadata_path)
        metadata_df = metadata_df[metadata_df['modeled_na_seq_len'] >= self.metadata_len_filter[0]]
        metadata_df = metadata_df[metadata_df['modeled_na_seq_len'] <= self.metadata_len_filter[1]+1]
        metadata_df_clust = metadata_df.groupby("cluster_structsim0.45")
        gt_sample_count = 0

        with open(f"{self.tmp_save_dict}/gt_list.list", "a+") as f:
            for clust_id, minidf in metadata_df_clust:
                sample_row = minidf.sample(n=1)
                name = sample_row['pdb_name'].values.tolist()[0]
                f.write(f"{name}.pdb\n")
                gt_sample_count += 1
        print (f"Loaded metadata CSV with {gt_sample_count} filtered cluster samples ...")

        if source:
            print ("Loading rankings from source ...")
            ranked_designable_seqlen_sample_dict = torch.load(source)
            flattened_ranked_designable_seqlen_sample_arr = []
            for _, sample_arr in ranked_designable_seqlen_sample_dict.items():
                flattened_ranked_designable_seqlen_sample_arr.extend(sample_arr)
        else:
            scrmsd_sample_list = self._get_designable_samples(gen_dir)
            ranked_designable_seqlen_sample_dict = self._rank_designable_topk(scrmsd_sample_list)

            flattened_ranked_designable_seqlen_sample_arr = []
            for _, sample_arr in ranked_designable_seqlen_sample_dict.items():
                flattened_ranked_designable_seqlen_sample_arr.extend(sample_arr)

            torch.save(
                ranked_designable_seqlen_sample_dict, 
                os.path.join(self.save_dir, f"final_metrics_scrmsd_ranked_samples_dict.pt")
            )
            torch.save(
                flattened_ranked_designable_seqlen_sample_arr, 
                os.path.join(self.save_dir, f"final_metrics_scrmsd_ranked_samples_arr.pt")
            )
        
        with open(f"{self.tmp_save_dict}/gen_list.list", "a+") as f:
            for line in flattened_ranked_designable_seqlen_sample_arr:
                f.write(line['pdb_name'] + ".pdb" + "\n") # line[1] corresponsd to the pdb_name without the .pdb extension

        try:
            # structural metrics
            for eval_name, func in structural_eval_dict.items():
                if eval_name not in exclude: # do not compute excluded eval metrics
                    print (f"#### Performing eval: '{eval_name}'")
                    for sample_name in os.listdir(gen_dir):
                        fp = os.path.join(gen_dir, sample_name)
                        uni = mda.Universe(fp, format="PDB")
                        c4p_atoms = uni.select_atoms("name C4'") # only select
                        c4p_coords = torch.from_numpy(c4p_atoms.positions) 

                        ret_arr = func(c4p_coords) # collect from samples
                        final_metric_dict[eval_name].extend(ret_arr)
                print (f"\tComputed {len(final_metric_dict[eval_name])}-long array ...")
        except Exception as e:
            print ("Could not compute structural metrics ...")
            print (e)

        final_metric_dict_struct = {key: final_metric_dict[key] for key in list(structural_eval_dict.keys())}
        torch.save(
            final_metric_dict_struct, 
            os.path.join(self.save_dir, "final_metrics_local.pt")
        )

        try:
            # whole-batch metrics
            for eval_name, func in batch_eval_dict.items():
                if eval_name not in exclude: # do not compute excluded eval metrics
                    print (f"#### Performing eval: '{eval_name}'")
                    ret_arr = func(gen_dir)
                    final_metric_dict[eval_name].extend(ret_arr)
        except Exception as e:
            print ("Could not compute tool-based metrics ...")
            print (e)

        final_metric_dict_tools = {key: final_metric_dict[key] for key in list(batch_eval_dict.keys())}
        torch.save(
            final_metric_dict_tools, 
            os.path.join(self.save_dir, "final_metrics_global.pt")
        )

        # save final metric dict
        torch.save(
            final_metric_dict, 
            os.path.join(self.save_dir, "final_metrics.pt")
        )

        return final_metric_dict
    
    def _flatten_dir(self, gen_dir, FWD_PATH):
        if not os.path.exists(FWD_PATH):
            print ("Created flattened directory ...")
            os.mkdir(FWD_PATH)

        for pdb_size_dir in os.listdir(gen_dir):
            fp = os.path.join(gen_dir, pdb_size_dir)
            if os.path.isdir(fp) and "length_" in fp:
                length = int(pdb_size_dir.split("_")[-1])
                for pdb_name in os.listdir(fp):
                    if pdb_name.endswith(".pdb") and "traj" not in pdb_name: # ignore trajectories, only keep final crystral structures
                        try:
                            idx = int(pdb_name.strip(".pdb").split("_")[-1])
                            new_name = os.path.join(gen_dir, pdb_size_dir, f"{length}_na_sample_{idx}.pdb")
                            fpp = os.path.join(gen_dir, pdb_size_dir, pdb_name)
                            
                            # copy and shift all samples
                            os.system(f"cp {fpp} {new_name}") # copy and store in the same location
                            os.system(f"mv {new_name} {FWD_PATH}") # move copied file to new location
                        except Exception as e:
                            print (pdb_name, length)
                            print (e)
                            print ()

    def _compute_c4_to_c4_distance(self, coords):
        """
        gen_dir : path of the generated samples (inference or validation)

        Returns:
            Array of all adjacent-residue C4' bond distances over the entire backbone, for all samples in `gen_dir`
        """
        rolled_coords = torch.roll(coords, shifts=1, dims=0)
        dist = torch.linalg.norm((coords - rolled_coords), dim=-1, ord=2)[1:].mean().item() # we add [1:] to ignore the first column (rolled up from the bottom)
        return [dist]
    
    def _compute_dihedral_angles(self, coords):
        """
        gen_dir : path of the generated samples (inference or validation)

        Returns:
            Array of all 8 torsion angles averaged over all residues for every sample in `gen_dir`
        """
        def _dihedral_angle_helper(a, b, c, d, eps = 1e-7):
            eps = torch.tensor(eps, device=a.device) # type: ignore

            # bc = F.normalize(b - c, dim=2)
            bc = F.normalize(b - c, dim=-1)
            # n1 = torch.cross(F.normalize(a - b, dim=2), bc)
            n1 = torch.cross(F.normalize(a - b, dim=-1), bc)
            # n2 = torch.cross(bc, F.normalize(c - d, dim=2))
            n2 = torch.cross(bc, F.normalize(c - d, dim=-1))
            # x = (n1 * n2).sum(dim=2)
            x = (n1 * n2).sum(dim=-1)
            x = torch.clamp(x, -1 + eps, 1 - eps)
            x[x.abs() < eps] = eps

            y = (torch.cross(n1, bc) * n2).sum(dim=-1)
            return torch.atan2(y, x).numpy().tolist()    

        C4_i0 = coords[0:-3, :]
        C4_i1 = coords[1:-2, :]
        C4_i2 = coords[2:-1, :]
        C4_i3 = coords[3:, :]
        angles = _dihedral_angle_helper(C4_i0, C4_i1, C4_i2, C4_i3)
        return angles

    def _compute_bond_angles(self, coords):
        """
        gen_dir : path of the generated samples (inference or validation)

        Returns:
            Array of all bond angles between C4' atom triplets between adjacent residues
        """        
        def _to_ang_helper(a, b, c):
            if a.ndim == 1:
                a = a.unsqueeze(0)
            if b.ndim == 1:
                b = b.unsqueeze(0)
            if c.ndim == 1:
                c = c.unsqueeze(0)

            ba = b - a
            bc = b - c
            return torch.acos(
                (ba * bc).sum(dim=-1)
                / (torch.norm(ba, dim=-1) * torch.norm(bc, dim=-1))
            ).numpy().tolist()  

        C4_i0 = coords[0:-2, :]
        C4_i1 = coords[1:-1, :]
        C4_i2 = coords[2:, :]
        angles = _to_ang_helper(C4_i0, C4_i1, C4_i2)
        return angles

    @torch.no_grad()
    def _get_designable_samples(self, gen_dir, n_seqs_per_bb=8):
        """
        For each generated PDB,
            1. inverse fold using gRNAde
            2. forward fold using RhoFold
            3. compute metrics between generated and predicted structures

        Metrics include: 
            scTM, scRMSD, scGDT
        
        Return: 
            final array of designable samples and respective metrics
        """

        gen_pdb_rmsds = []

        # inverse-fold all generated samples
        pbar = tqdm(os.listdir(gen_dir))
        for _, pdb_name in enumerate(pbar): # outer loop: generated PDBs
            pbar.set_description(pdb_name)
            gen_fp = os.path.join(gen_dir, pdb_name)
            
            # inner loop 1: generate sequences
            # for seq_idx in range(n_seqs_per_bb):
            # savepath: dir/<seq_len>_na_sample_<idx>_<seq_idx>.fasta
            fasta_fn = pdb_name.strip(".pdb")
            out_fp = os.path.join(self.grnade_save_path, f"{fasta_fn}.fasta")
            
            # save n_seqs_per_bb FASTA files for each generated backbone
            if self.use_invfold:
                gen_seqs, _, _, _ = self.gRNAde_module.design_from_pdb_file(
                                                                pdb_filepath=gen_fp,
                                                                output_filepath=None,
                                                                n_samples=n_seqs_per_bb,
                                                                temperature=0.1
                                                            )
            else:
                raw_data = self.gRNAde_module.get_raw_data(pdb_filepath=gen_fp)
                og_seq = raw_data['sequence'] # original sequence in the PDB 
                gen_seqs = []
                for idx in range(n_seqs_per_bb):       
                    gen_seqs.append(
                        SeqRecord(
                                Seq(og_seq), 
                                id=f"sample={idx}",
                                description=f"original:{og_seq}"
                            )
                    )

            # print (f"Generated structure for: {pdb_name}")
            torch.cuda.empty_cache()
            gc.collect()

            seq_files = []
            # split sequences across N_seq different FASTA files
            for seq_idx in range(n_seqs_per_bb):
                """
                1. Read `n_seqs_per_bb` sequences from out_fp
                2. Create `n_seqs_per_bb` separate files
                3. Save each sequence in new file
                """
                seq = gen_seqs[seq_idx] # TODO: read sequences from out_fp
                updated_fasta_fn = pdb_name.strip(".pdb") + f"_{seq_idx}"
                new_out_fp = os.path.join(self.grnade_save_path, updated_fasta_fn)
                seq_files.append([
                        new_out_fp, 
                        updated_fasta_fn,
                        seq_idx
                    ])
                SeqIO.write([seq], new_out_fp, "fasta")

            # inner loop 2: for each predicted sequence, predict the structure
            all_pdb_names = []
            for rec in seq_files:
                """
                Pass `seq_file` through RhoFold using self.rmsd_thresh and --input_fas flag
                """
                seq_fp, fasta_fn, seq_idx = rec
                input_a3m = input_fas = seq_fp
                data_dict = get_features(input_fas, input_a3m)
                outputs = self.rhofold(
                                tokens=data_dict['tokens'].to(self.device), 
                                rna_fm_tokens=data_dict['rna_fm_tokens'].to(self.device), 
                                seq=data_dict['seq']
                            )
                
                output = outputs[-1]
                unrelaxed_model = os.path.join(self.rhofold_save_path, f'{fasta_fn}.pdb')
                node_cords_pred = output['cord_tns_pred'][-1].squeeze(0)

                all_pdb_names.append(f'{fasta_fn}.pdb')

                self.rhofold.structure_module.converter.export_pdb_file(
                                                            data_dict['seq'],
                                                            node_cords_pred.data.cpu().numpy(),
                                                            path=unrelaxed_model, 
                                                            chain_id=None,
                                                            confidence=output['plddt'][0].data.cpu().numpy(),
                                                            logger=None
                                                        )
        
                torch.cuda.empty_cache()
                gc.collect()
                
            # inner loop 3: compute RMSDs between predicted and original generated backbone
            gen_uni = mda.Universe(gen_fp, format="PDB")
            gen_c4p_atoms = gen_uni.select_atoms("name C4'")
            
            # temp metrics
            cur_min_rmsd = float("inf") # very large number
            cur_max_scTM = float("-inf")
            cur_scGDT = float("-inf")
            cur_best_pred_pdb_name = "" # the training PDB that gives the closest match to current generated structure
            n_res = len(gen_c4p_atoms)

            for k in range(n_seqs_per_bb):
                # rhofold_pdb_name : <seq_len>_na_sample_<idx>_<seq_idx>.pdb
                rhofold_fp = os.path.join(self.rhofold_save_path, pdb_name.strip(".pdb") + f"_{k}.pdb")
                rf_uni = mda.Universe(rhofold_fp, format="PDB")
                rf_c4p_atoms = rf_uni.select_atoms("name C4'")

                if gen_c4p_atoms.positions.shape[0] != rf_c4p_atoms.positions.shape[0]:
                    continue # skip if size mismatch (negligible)

                # compute scTM
                tm = self._compute_tm_score(
                            gen_path=gen_fp,
                            pred_path=rhofold_fp,
                            gen_pdb_name=pdb_name,
                            pred_pdb_name=pdb_name.strip(".pdb") + f"_{k}.pdb"
                        ) # compute C4' TM-score
                
                # compute scGDT
                gdt = self._compute_gdt(
                            gen_c4p_atoms.positions,
                            rf_c4p_atoms.positions, 
                        ) # compute C4' GDT

                # compute scRMSD
                r_score = rms.rmsd(
                            gen_c4p_atoms.positions, 
                            rf_c4p_atoms.positions, 
                            center=True, 
                            superposition=True
                        ) # compute C4' RMSD
                
                # use scTM to track the best structure among all RhoFold predictions
                if cur_max_scTM < tm:
                    cur_min_rmsd = r_score
                    cur_best_pred_pdb_name = pdb_name.strip(".pdb") + f"_{k}" # store the best predicted match's PDB filename from RhoFold
                    cur_max_scTM = tm # store scTM of best structure prediction (assumed to be the best as well)
                    cur_scGDT = gdt # store scGDT of best structure prediction (assumed to be the best as well)

            # store record for faster future retrieval
            gen_pdb_rmsds.append({
                "gen_dir": gen_dir,
                "pdb_name": pdb_name.strip(".pdb"),
                "min_scRMSD": cur_min_rmsd,
                "n_res": n_res,
                "cur_best_pred_pdb_name": cur_best_pred_pdb_name,
                "min_scTM": cur_max_scTM,
                "min_scGDT": cur_scGDT
            })
                
        return gen_pdb_rmsds
    
    def _rank_designable_topk(self, sample_arr, rankby="min_scTM"):
        """
        Params:
            sample_arr (list) : list of samples with respective self-consistency details
            rankby (str): select from ["min_scRMSD", "min_scTM", "min_scGDT"]

        Remarks:
            Sorts samples according to given metric (scTM, scRMSD, scGDT) to compute validity

        Returns:
            Sorted list of valid samples
        """

        """
        NOTE: 
        The older version of this method also extracted the best "K" samples (K was a method parameter). 
        We've removed this functionality and take all samples, even if they aren't among the best "K". 
        """

        ranked_samples = {}
        for sample in sample_arr:
            # _, _, _, num_res, _ = sample
            num_res = sample['n_res']
            if num_res not in ranked_samples:
                ranked_samples[num_res] = [sample]
            else:
                ranked_samples[num_res].append(sample)

        for n_res, seqlen_sample_arr in ranked_samples.items():
            # pdb_name_wo_ext doesn't have the .pdb extension

            if rankby == "min_scTM" or rankby == "min_scGDT":
                ranked_seqlen_sample_arr = sorted(seqlen_sample_arr, key=lambda rec : rec[rankby], reverse=True) # sort in descending order by scTM/scGDT
            elif rankby == "min_scRMSD":
                ranked_seqlen_sample_arr = sorted(seqlen_sample_arr, key=lambda rec : rec[rankby], reverse=False) # sort in ascending order by scRMSD
        
            filtered_seqlen_sample_arr = ranked_seqlen_sample_arr
            ranked_samples[n_res] = filtered_seqlen_sample_arr # extract the top-K and assign back to that size bucket
        
        return ranked_samples # return final ranked+filtered size buckets 
    
    def _compute_gdt(self, gen_pos, pred_pos):
        return metrics.gdt_score(
                                x=torch.from_numpy(gen_pos).unsqueeze(0), 
                                y=torch.from_numpy(pred_pos).unsqueeze(0)
                            )
    
    def _compute_tm_score(self, gen_path, pred_path, gen_pdb_name, pred_pdb_name):
        tmscore_tmp_dir = "tmscore_tmp_dir/"
        os.makedirs(os.path.join(self.tmp_save_dict, tmscore_tmp_dir), exist_ok=True)

        gen_tmp_dir = os.path.join(self.tmp_save_dict, tmscore_tmp_dir, "gen")
        pred_tmp_dir = os.path.join(self.tmp_save_dict, tmscore_tmp_dir, "pred")

        os.makedirs(gen_tmp_dir, exist_ok=True)
        os.makedirs(pred_tmp_dir, exist_ok=True)
        os.system(f"cp {gen_path} {gen_tmp_dir}")
        os.system(f"cp {pred_path} {pred_tmp_dir}")
        
        gen_list_path = os.path.join(self.tmp_save_dict, tmscore_tmp_dir, "gen_tmscore.list")
        pred_list_path = os.path.join(self.tmp_save_dict, tmscore_tmp_dir, "pred_tmscore.list")
        with open(gen_list_path, "w") as f:
            f.write(gen_pdb_name)
        with open(pred_list_path, "w") as f:
            f.write(pred_pdb_name)

        ret_path = os.path.join(self.tmp_save_dict, "TMSCORE_TMP_RESULTS.txt")
        CMD = f"{self.tmscore_path} -dir1 {gen_tmp_dir} {gen_list_path} -dir2 {pred_tmp_dir} {pred_list_path} -outfmt 2 > {ret_path}"
        os.system(CMD)

        df = pd.read_csv(ret_path, delimiter='\t')
        df = df[['#PDBchain1', "PDBchain2", "TM2", "Lali"]]
        tmscore = df['TM2'].values.tolist()[0]

        # delete all temp files
        os.system(f"rm -rf {tmscore_tmp_dir}")
        # os.system(f"rm {ret_path}")

        return tmscore
    
    def compute_tmscore_from_usalign(self, usalign_report_path):
        df = pd.read_csv(usalign_report_path, delimiter="\t")
        df = df[['#PDBchain1', 'PDBchain2', 'TM2']]

        df_ = df.groupby("#PDBchain1")
        max_tmscore_per_pdb = df_['TM2'].max()

        tm_json = max_tmscore_per_pdb.to_json()
        tm_json = json.loads(tm_json)
        
        seqlens = []
        pdb_names = []
        tm_scores = []
        for pdb_name, tm2 in tm_json.items():
            seqlen = int(pdb_name.split("_")[0].strip("/"))
            seqlens.append(seqlen)
            pdb_names.append(pdb_name)
            tm_scores.append(tm2)
        
        dct = {
            "seqlen": seqlens,
            "pdb_name": pdb_names,
            "tm2": tm_scores
        }

        tm_report_df = pd.DataFrame.from_dict(dct)
        return tm_report_df

    def _compute_usalign_metrics(self, gen_dir):
        """
        gen_dir : path of the generated samples (inference or validation)
        """

        ret_path = os.path.join(self.tmp_save_dict, "MASTER_USALIGN_RESULTS.txt")
        gen_path = os.path.join(gen_dir)
        gt_path = os.path.join(self.gt_dir)
        gen_list_path = os.path.join(self.tmp_save_dict, "gen_list.list")
        gt_list_path = os.path.join(self.tmp_save_dict, "gt_list.list")

        # cancel if nothing is considered designable
        with open(gen_list_path, "r") as f:
            d = f.readlines()
        if len(d) < 1:
            return []
        
        CMD = f"{self.usalign_path} -dir1 {gen_dir} {gen_list_path} -dir2 {gt_path} {gt_list_path} -outfmt 2 > {ret_path}"
        os.system(CMD)

        df = pd.read_csv(ret_path, delimiter="\t")
        df = df[['#PDBchain1', 'PDBchain2', 'TM2']]
        df['seqlen'] = df.apply(lambda row : row['#PDBchain1'].split("_")[0], axis=1)
        
        # For each generated X, look at Y={y1, ..., yn}, get max across all (X,yi) pairs and then take mean of |X| generated samples
        df_ = df.groupby("#PDBchain1")
        avg_pdbTM = df_['TM2'].max().mean()
        return [avg_pdbTM]

    def _compute_qTMclust_metrics(self, gen_dir):
        """
        gen_dir : path of the generated samples (inference or validation)
        """

        n_samples = len(os.listdir(gen_dir))
        
        ret_path = os.path.join(self.tmp_save_dict, "MASTER_QTM_RESULTS.txt")
        gen_path = os.path.join(gen_dir)
        gen_list_path = os.path.join(self.tmp_save_dict, "gen_list.list")

        # cancel if nothing is considered designable
        with open(gen_list_path, "r") as f:
            d = f.readlines()
        if len(d) < 1:
            return [-1]

        CMD = f"{self.qtmclust_path} -dir {gen_path} {gen_list_path} -TMcut {self.tm_thresh} -o {ret_path}"
        os.system(CMD)
        
        with open(ret_path, "r+") as f:
            clusters = f.readlines()

        # diversity = number of unique structural clusters / total samples
        div = len(clusters) / n_samples
        return [div]

    def load_from_metric_dict(self, metric_dict_path):
        return torch.load(metric_dict_path)
    
    def print_metrics(self, metric_dict):
        print ("\n\nMETRICS:")
        print (f"Diversity (#clusters / #designable): {metric_dict['batch_clusters']}")
        print (f"Novelty (pdbTM): {metric_dict['batch_pdbTM']}")

        sorted_samples_path = f"{self.save_dir}/final_metrics_scrmsd_ranked_samples_dict.pt"
        sorted_scrmsd = torch.load(sorted_samples_path)

        temp = []
        total = 0
        filt = 0

        for l, v in sorted_scrmsd.items():
            rmsd = list(map(lambda x : x['min_scRMSD'], v))
            rmsd = list(sorted(rmsd))
            filt_rmsd = list(filter(lambda x : x <= self.rmsd_thresh, rmsd))
            temp.extend(rmsd)
            total += len(rmsd)
            filt += len(filt_rmsd)
        print (f"Validity (% <= {self.rmsd_thresh}A): {filt/total:.5f} | {filt} / {total}")
        
        temp = []
        total = 0
        filt = 0

        for _, v in sorted_scrmsd.items():
            tm = list(map(lambda x : x['min_scTM'], v))
            tm = list(sorted(tm))
            filt_tm = list(filter(lambda x : x >= self.tm_thresh, tm))
            temp.extend(tm)
            total += len(tm)
            filt += len(filt_tm)
        print (f"Validity (% >= {self.tm_thresh}): {filt/total:.5f} | {filt} / {total}")
