import os
import pandas as pd
from gRNAde import gRNAde

# GEN_SAMPLES_PATH = "rnager_inference_outputs/B400-with-aux-tors-loss/2024-03-31_15-50-29/last/run_2024-04-01_01-44/"
# all_pdb_paths = []

gRNAde_module = gRNAde(split='all', max_num_conformers=1, gpu_id=0)
print (gRNAde_module)

fp = "rnager_large_outputs/B400-with-aux-tors-loss/2024-03-31_15-50-29/last/run_2024-04-02_12-38/flattened_samples/150_na_sample_6.pdb"
out_fp = f"./invfold_grnade_seqs/"
if not os.path.exists(out_fp):
    os.mkdir(out_fp)
out_fp = os.path.join(out_fp, "sample.fasta")
gen_sequences, gen_samples, _, _ = gRNAde_module.design_from_pdb_file(
                                pdb_filepath=fp,
                                output_filepath=out_fp,
                                n_samples=10
                            )
    
print (gen_sequences)
# print (gen_samples)