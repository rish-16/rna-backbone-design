# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for MMDiff (https://github.com/Profluent-Internships/MMDiff):
# -------------------------------------------------------------------------------------------------------------------------------------
"""Script for preprocessing PDB (non-mmCIF) files."""

import argparse
import collections
import functools as fn
import multiprocessing as mp
import os
import time
from tqdm import tqdm
from typing import Any, Dict, Optional

import mdtraj as md
import numpy as np
import pandas as pd
import torch
from Bio import PDB

from rna_backbone_design.data import utils
from rna_backbone_design.data import parsers

# Define the parser
parser = argparse.ArgumentParser(description="PDB processing script.")
parser.add_argument("--pdb_dir", help="Path to directory with RNA PDB files.", type=str)
parser.add_argument("--num_processes", help="Number of processes.", type=int, default=16)
parser.add_argument(
    "--write_dir", help="Path to write results to.", type=str, default="./preprocessed_pdbs"
)
parser.add_argument(
    "--skip_existing", help="Whether to skip processed files.", action="store_true"
)
parser.add_argument("--debug", help="Turn on for debugging.", action="store_true")
parser.add_argument("--verbose", help="Whether to print everything.", action="store_true")


def process_file(
    file_path: str,
    write_dir: str,
    inter_chain_interact_dist_threshold: float = 7.0,
    skip_existing: bool = False,
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    """Processes protein file into usable, smaller pickles.

    Args:
        file_path: Path to file to read.
        write_dir: Directory to write pickles to.
        inter_chain_interact_dist_threshold: Euclidean distance under which
            to classify a pairwise inter-chain residue-atom distance as an interaction.
        skip_existing: Whether to skip processed files.
        verbose: Whether to log everything.

    Returns:
        Saves processed protein to pickle and returns metadata.

    Raises:
        DataError if a known filtering rule is hit.
        All other errors are unexpected and are propagated.
    """
    metadata = {}
    pdb_name = os.path.basename(file_path).replace(".pdb", "")
    metadata["pdb_name"] = pdb_name

    pdb_subdir = os.path.join(write_dir, pdb_name[1:3].lower())
    os.makedirs(pdb_subdir, exist_ok=True)
    processed_path = os.path.join(pdb_subdir, f"{pdb_name}.pkl")
    metadata["processed_path"] = os.path.abspath(processed_path)
    metadata["raw_path"] = file_path
    if skip_existing and os.path.exists(metadata["processed_path"]):
        return None
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_name, file_path)

    # Extract all chains
    struct_chains = {chain.id.upper(): chain for chain in structure.get_chains()}
    metadata["num_chains"] = len(struct_chains)

    # Extract features
    all_seqs = set()
    struct_feats = []
    num_na_chains = 0
    
    na_natype, chain_dict = None, None
    
    for chain_id, chain in struct_chains.items():
        # Convert chain id into int
        chain_index = utils.chain_str_to_int(chain_id)
        chain_mol = parsers.process_chain_pdb(chain, chain_index, chain_id, verbose=verbose)
        
        if chain_mol is None:
            # Note: Indicates that neither a protein chain nor a nucleic acid chain was found
            continue
        elif chain_mol[-1]["molecule_type"] == "na":
            num_na_chains += 1
            na_natype = (
                chain_mol[-2]
                if na_natype is None
                else torch.cat((na_natype, chain_mol[-2]), dim=0)
            )

        chain_mol_constants = chain_mol[-1]["molecule_constants"]
        chain_mol_backbone_atom_name = chain_mol[-1]["molecule_backbone_atom_name"]
        chain_dict = parsers.macromolecule_outputs_to_dict(chain_mol)
        
        chain_dict = utils.parse_chain_feats_pdb(
            chain_feats=chain_dict,
            molecule_constants=chain_mol_constants,
            molecule_backbone_atom_name=chain_mol_backbone_atom_name,
        )
        all_seqs.add(tuple(chain_dict["aatype"]))
        struct_feats.append(chain_dict)
    
    if chain_dict is None:
        # Note: Indicates that no protein chains or nucleic acid chains were found for the input complex
        if verbose:
            print (f"No chains were found for PDB {file_path}. Skipping...")
        return None
    
    if len(all_seqs) == 1:
        metadata["quaternary_category"] = "homomer"
    else:
        metadata["quaternary_category"] = "heteromer"

    # Add assembly features from AlphaFold-Multimer,
    # relying on references to each `chain_dict` object
    # to propagate back to the contents of `struct_feats` below
    seq_to_entity_id = {}
    grouped_chains = collections.defaultdict(list)
    for chain_dict in struct_feats:
        seq = tuple(chain_dict["aatype"])
        if seq not in seq_to_entity_id:
            seq_to_entity_id[seq] = len(seq_to_entity_id) + 1
        grouped_chains[seq_to_entity_id[seq]].append(chain_dict)

    new_all_chain_dict = {}
    chain_id = 1
    for entity_id, group_chain_features in grouped_chains.items():
        for sym_id, chain_dict in enumerate(group_chain_features, start=1):
            new_all_chain_dict[f"{utils.int_id_to_str_id(entity_id)}_{sym_id}"] = chain_dict
            seq_length = len(chain_dict["aatype"])
            chain_dict["asym_id"] = chain_id * np.ones(seq_length)
            chain_dict["sym_id"] = sym_id * np.ones(seq_length)
            chain_dict["entity_id"] = entity_id * np.ones(seq_length)
            chain_id += 1

    # Concatenate all collected features
    complex_feats = utils.concat_np_features(struct_feats, add_batch_dim=False)
    if complex_feats["bb_mask"].sum() < 1.0:
        # Note: Indicates an example did not contain any parseable residues
        return None
    assert len(complex_feats["bb_mask"]) == len(
        complex_feats["aatype"]
    ), "Number of core atoms must match number of residues."

    # Record molecule metadata
    metadata["num_protein_chains"] = 0
    metadata["num_na_chains"] = num_na_chains

    # Process geometry features
    complex_aatype = complex_feats["aatype"]
    metadata["seq_len"] = len(complex_aatype)
    # metadata["protein_seq_len"] = 0 if protein_aatype is None else len(protein_aatype)
    metadata["na_seq_len"] = 0 if na_natype is None else len(na_natype)
    # Note: Residue indices `20` and `26`, respectively, correspond to missing protein and nucleic acid residue types
    modeled_idx = np.where((complex_aatype != 20) & (complex_aatype != 26))[0]
    # protein_modeled_idx = None if protein_aatype is None else np.where(protein_aatype != 20)[0]
    na_modeled_idx = None if na_natype is None else np.where(na_natype != 26)[0]
    if np.sum((complex_aatype != 20) & (complex_aatype != 26)) == 0:
        raise utils.LengthError("No modeled residues")
    metadata["modeled_seq_len"] = np.max(modeled_idx) - np.min(modeled_idx) + 1
    metadata["modeled_protein_seq_len"] = 0
    # metadata["modeled_protein_seq_len"] = (
        # 0
        # if protein_aatype is None
        # else np.max(protein_modeled_idx) - np.min(protein_modeled_idx) + 1
    # )
    metadata["modeled_na_seq_len"] = (
        0 if na_natype is None else np.max(na_modeled_idx) - np.min(na_modeled_idx) + 1
    )
    complex_feats["modeled_idx"] = modeled_idx
    # complex_feats["protein_modeled_idx"] = protein_modeled_idx
    complex_feats["na_modeled_idx"] = na_modeled_idx

    # Find all inter-chain interface residues (e.g., <= 7.0 Angstrom from an inter-chain non-hydrogen atom)
    num_atoms_per_res = complex_feats["atom_positions"].shape[1]
    bb_pos = torch.from_numpy(complex_feats["bb_positions"]).unsqueeze(0)
    atom_pos = (
        torch.from_numpy(complex_feats["atom_positions"])
        .unsqueeze(0)
        .flatten(start_dim=1, end_dim=2)
    )
    bb_asym_id = torch.from_numpy(complex_feats["asym_id"]).unsqueeze(0)
    atom_asym_id = torch.repeat_interleave(
        bb_asym_id.unsqueeze(2), num_atoms_per_res, dim=2
    ).flatten(start_dim=1, end_dim=2)
    dist_mat = torch.cdist(bb_pos, atom_pos)
    inter_chain_mask = bb_asym_id.unsqueeze(-1) != atom_asym_id.unsqueeze(-2)
    non_h_mask = torch.ones_like(
        inter_chain_mask
    )  # Note: This assumes that the PDB parsing excluded all `H` atoms
    interacting_res_mask = (
        inter_chain_mask & non_h_mask & (dist_mat <= inter_chain_interact_dist_threshold)
    )
    complex_feats["inter_chain_interacting_idx"] = torch.nonzero(
        interacting_res_mask.squeeze(0), as_tuple=False
    )[..., 0].unique()

    try:
        # MDtraj
        traj = md.load(file_path)
    except Exception as e:
        if verbose:
            print (f"Mdtraj failed to load file {file_path} with error {e}")
        traj = None
    try:
        # SS calculation
        pdb_ss = md.compute_dssp(traj, simplified=True) if traj is not None else None
    except Exception as e:
        if verbose:
            print (f"Mdtraj's call to DSSP failed with error {e}")
        pdb_ss = None
    try:
        # DG calculation
        pdb_rg = md.compute_rg(traj) if traj is not None else None
    except Exception as e:
        if verbose:
            print (f"Mdtraj's call to RG failed with error {e}")
        pdb_rg = None

    metadata["coil_percent"] = (
        np.sum(pdb_ss == "C") / metadata["modeled_seq_len"] if pdb_ss is not None else np.nan
    )
    metadata["helix_percent"] = (
        np.sum(pdb_ss == "H") / metadata["modeled_seq_len"] if pdb_ss is not None else np.nan
    )
    metadata["strand_percent"] = (
        np.sum(pdb_ss == "E") / metadata["modeled_seq_len"] if pdb_ss is not None else np.nan
    )

    # Radius of gyration
    metadata["radius_gyration"] = pdb_rg[0] if pdb_rg is not None else np.nan

    # Write features to pickles.
    utils.write_pkl(processed_path, complex_feats)

    # Return metadata
    return metadata


def process_serially(all_paths, write_dir, skip_existing=False, verbose=False):
    all_metadata = []
    for file_path in tqdm(all_paths):
        try:
            start_time = time.time()
            metadata = process_file(
                file_path, write_dir, skip_existing=skip_existing, verbose=verbose
            )
            elapsed_time = time.time() - start_time
            print (f"Finished {file_path} in {elapsed_time:2.2f}s")
            if metadata is not None:
                all_metadata.append(metadata)
        except utils.DataError as e:
            print (f"Failed {file_path}: {e}")
    return all_metadata


def process_fn(file_path, write_dir=None, skip_existing=False, verbose=False):
    try:
        start_time = time.time()
        metadata = process_file(file_path, write_dir, skip_existing=skip_existing, verbose=verbose)
        elapsed_time = time.time() - start_time
        if verbose:
            print (f"Finished {file_path} in {elapsed_time:2.2f}s")
        return metadata
    except utils.DataError as e:
        if verbose:
            print (f"Failed {file_path}: {e}")


def main(args):
    pdb_dir = args.pdb_dir
    all_file_paths = [
        os.path.join(pdb_dir, item)
        for item in os.listdir(pdb_dir)
        if ".pdb" in item
    ]
    # all_file_paths = os.listdir(pdb_dir)
    # all_file_paths = [os.path.join(all_file_paths, fp) for fp in all_file_paths if ".pdb" in fp]
    total_num_paths = len(all_file_paths)
    write_dir = args.write_dir
    os.makedirs(write_dir, exist_ok=True)
    if args.debug:
        metadata_file_name = "rna_metadata_debug.csv"
    else:
        metadata_file_name = "rna_metadata.csv"
    metadata_path = os.path.join(write_dir, metadata_file_name)
    print (f"Files will be written to {write_dir}")

    # Process each PDB file
    if args.num_processes == 1 or args.debug:
        all_metadata = process_serially(
            all_file_paths, write_dir, skip_existing=args.skip_existing, verbose=args.verbose
        )
    else:
        _process_fn = fn.partial(
            process_fn, write_dir=write_dir, skip_existing=args.skip_existing, verbose=args.verbose
        )
        with mp.Pool(processes=args.num_processes) as pool:
            all_metadata = pool.map(_process_fn, all_file_paths)
        all_metadata = [x for x in all_metadata if x is not None]
    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(metadata_path, index=False)
    succeeded = len(all_metadata)
    print (f"Finished processing {succeeded}/{total_num_paths} files")


if __name__ == "__main__":
    # Don't use GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args = parser.parse_args()
    main(args)

    # python process_rna_pdb_files.py --pdb_dir data/rnasolo/ --write_dir data/
