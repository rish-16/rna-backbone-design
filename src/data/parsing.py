# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for (https://github.com/Profluent-Internships/MMDiff):
# -------------------------------------------------------------------------------------------------------------------------------------

from typing import Optional

import numpy as np
import torch
from Bio.PDB import MMCIFParser, PDBParser, Structure
from src.data import vocabulary

def structure_to_XCS(
    structure: Structure,
    constants,
    chain_id: Optional[str] = None,
    nmr_okay: bool = False,
    skip_nonallowable_restypes: bool = True,
    with_gaps=False,
):
    """Convert a Bio.PDB.Structure object into X, C, S, and metadata.

    Parameters
    ----------
    structure : Bio.PDB.Structure
        The structure to convert.
    chain_id : str, optional
        The chain ID to use. If None, use the first chain.
    nmr_okay : bool, optional
        Whether to allow NMR structures. If False, raise an error if the structure
        contains multiple models.
    skip_nonallowable_restypes : bool, optional
        Whether to skip residues that are not in the allowable residue types.
    with_gaps: bool, optional
        Whether to handle for potential sequence gaps.

    Returns
    -------
    X : torch.Tensor
        The coordinates of the atoms in the structure.
    C : torch.Tensor
        The chain IDs and mask for the structure.
    S : torch.Tensor
        The sequence of the structure.
    metadata : dict
        The metadata for the structure.
    """

    models = list(structure.get_models())
    if len(models) != 1 and not nmr_okay:
        raise ValueError(f"Only single model PDBs are supported. Found {len(models)} models.")
    model = models[0]

    X, C, S = [], [], []
    atom_mask, chain_ids, b_factors, deoxy = [], [], [], []

    chain_idx = 1
    last_res_idx = None
    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for res in chain:
            if res.resname in vocabulary.alternative_restypes_map:
                res.resname = vocabulary.alternative_restypes_map[res.resname]
            if res.resname not in constants.restype_name_to_compact_atom_order:
                if skip_nonallowable_restypes:
                    continue
                else:
                    print(
                        f"Skipping chain {chain.id} because it contains unallowed residue {res.resname}."
                    )
                    break

            res_shortname = vocabulary.restype_3to1.get(
                res.resname, vocabulary.unknown_protein_restype
            )
            restype_idx = vocabulary.restype_order.get(res_shortname)
            pos = np.zeros((constants.compact_atom_type_num, 3))
            mask = np.zeros((constants.compact_atom_type_num,))
            res_b_factors = np.zeros((constants.compact_atom_type_num,))
            for atom in res:
                compact_atom_order = constants.restype_name_to_compact_atom_order[res.resname]
                if atom.name not in compact_atom_order:
                    continue
                atom_idx = compact_atom_order[atom.name]
                pos[atom_idx] = atom.coord
                mask[atom_idx] = 1.0
                res_b_factors[atom_idx] = atom.bfactor

            # Currently this will skip nucleotides because we have not defined their atoms
            # if not with_gaps:
            if np.sum(mask) == 0:
                continue

            if with_gaps:
                if last_res_idx is None:
                    last_res_idx = res.id[1] - 1
                res_idx = res.id[1]
                while res_idx > last_res_idx + 1:
                    atom_mask.append(np.zeros((constants.compact_atom_type_num,)))
                    chain_ids.append(chain.id)
                    b_factors.append(np.zeros((constants.compact_atom_type_num,)))
                    X.append(np.zeros((constants.compact_atom_type_num, 3)))
                    C.append(-chain_idx)
                    S.append(vocabulary.gap_token)
                    last_res_idx += 1
                last_res_idx = res_idx

            atom_mask.append(mask)
            chain_ids.append(chain.id)
            b_factors.append(res_b_factors)

            # if constants has deoxy_restypes attribute, check if resname is in it
            if hasattr(constants, "deoxy_restypes") and res.resname in constants.deoxy_restypes:
                deoxy.append(True)
            else:
                deoxy.append(False)

            has_all_atoms = int(sum(mask)) == constants.restype_name_atom_num[res.resname]
            has_all_atoms = 1 if has_all_atoms else -1

            X.append(pos)
            C.append(chain_idx * has_all_atoms)
            S.append(restype_idx)

        chain_idx += 1

    X = torch.tensor(np.array(X))
    C = torch.tensor(np.array(C))
    S = torch.tensor(np.array(S))

    atom_mask = torch.tensor(np.array(atom_mask))
    chain_ids = np.array(chain_ids).tolist()
    b_factors = torch.tensor(np.array(b_factors))
    deoxy = torch.tensor(np.array(deoxy))

    metadata = {
        "atom_mask": atom_mask,
        "chain_ids": chain_ids,
        "b_factors": b_factors,
        "deoxy": deoxy,
    }

    return X, C, S, metadata


def pdb_to_XCS(
    pdb_file: str,
    constants,
    chain_id: Optional[str] = None,
    nmr_okay: bool = False,
    skip_nonallowable_restypes: bool = True,
    with_gaps=False,
):
    """Convert a PDB file to macromolecular metadata.

    Parameters
    ----------
    pdb_file : str
        The path to the PDB file.
    chain_id : str, optional
        The chain ID to use. If None, use the first chain.
    nmr_okay : bool, optional
        Whether to allow NMR structures. If False, raise an error if the structure
        contains multiple models.
    skip_nonallowable_restypes : bool, optional
        Whether to skip residues that are not in the allowable residue types.

    Returns
    -------
    Macromolecular metadata.
    """

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("", pdb_file)

    X, C, S, metadata = structure_to_XCS(
        structure,
        constants,
        chain_id=chain_id,
        nmr_okay=nmr_okay,
        skip_nonallowable_restypes=skip_nonallowable_restypes,
        with_gaps=with_gaps,
    )
    metadata["source_file"] = pdb_file

    return X, C, S, metadata


def mmcif_to_XCS(
    cif_file: str,
    constants,
    chain_id: Optional[str] = None,
    nmr_okay: bool = False,
    skip_nonallowable_restypes: bool = True,
    with_gaps=False,
):
    """Convert a mmCIF file to macromolecular metadata.

    Parameters
    ----------
    cif_file : str
        The path to the mmCIF file.
    chain_id : str, optional
        The chain ID to use. If None, use the first chain.
    nmr_okay : bool, optional
        Whether to allow NMR structures. If False, raise an error if the structure
        contains multiple models.
    skip_nonallowable_restypes : bool, optional
        Whether to skip residues that are not in the allowable residue types.

    Returns
    -------
    Macromolecular metadata.
    """

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("", cif_file)

    X, C, S, metadata = structure_to_XCS(
        structure,
        constants,
        chain_id=chain_id,
        nmr_okay=nmr_okay,
        skip_nonallowable_restypes=skip_nonallowable_restypes,
        with_gaps=with_gaps,
    )
    metadata["source_file"] = cif_file

    return X, C, S, metadata
