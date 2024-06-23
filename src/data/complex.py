# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for (https://github.com/Profluent-Internships/MMDiff):
# -------------------------------------------------------------------------------------------------------------------------------------
"""Complex data type."""
import dataclasses

import numpy as np
from beartype.typing import Any, Mapping

from src.data import complex_constants, nucleotide_constants, protein_constants

FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.

# Complete sequence of chain IDs supported by the PDB format.
PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)  # := 62.


@dataclasses.dataclass(frozen=True)
class Complex:
    """Complex structure representation."""

    # Cartesian coordinates of atoms in angstroms. The 37 atom types correspond to
    # residue_constants.atom_types. e.g., The first three are `N, CA, CB` for
    # `protein` residues, whereas the first three are `C1', C2', C3'` for
    # `nucleic acid` residues.
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    # Residue (i.e., acid) type for each residue represented as an integer between 0 and
    # 36, where 20 is 'X' and 29 is 'x'.
    restype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. e.g., This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # Molecule type with which each residue associated. A `long` value of `1` in
    # the first column denotes a `protein` residue, whereas a `long` value of `1` in
    # the second column represents a `nucleic acid` residue.
    residue_molecule_type: np.ndarray  # [num_res, 2]

    # 0-indexed number corresponding to the chain in the protein that this residue
    # belongs to.
    chain_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_res, num_atom_type]

    def __post_init__(self):
        if len(np.unique(self.chain_index)) > PDB_MAX_CHAINS:
            raise ValueError(
                f"Cannot build an instance with more than {PDB_MAX_CHAINS} chains "
                "because these cannot be written to PDB format."
            )


def complex_to_pdb(complex: Complex, model=1, add_end=True, molecule_type_to_write=-1) -> str:
    """Converts a `Complex` instance to a PDB string.

    Args:
      complex: The `Complex` object to convert to PDB.
      model: Integer denoting which PDB model to write.
      add_end: Whether to add an `END` line at the bottom
        of the constructed PDB string.
      molecule_type_to_write: Which molecule type to write
        in isolation within the constructed PDB string.
        Defaults to `-1` to denote that all molecule types
        should be written. A value of `0` indicates that
        only `protein` molecules should be written, whereas
        a value of `1` represents that only `nucleic acid`
        molecules should be written.

    Returns:
      PDB string.
    """
    restype_name_to_full_atom_names = protein_constants.restype_name_to_full_atom_names.copy()
    restype_name_to_full_atom_names.update(
        nucleotide_constants.restype_name_to_full_atom_names.copy()
    )

    pdb_lines = []

    atom_positions = complex.atom_positions
    restype = complex.restype
    atom_mask = complex.atom_mask
    residue_index = complex.residue_index.astype(int)
    residue_molecule_type = complex.residue_molecule_type.astype(int)
    chain_index = complex.chain_index.astype(int)
    b_factors = complex.b_factors

    if np.any(restype > complex_constants.restype_num):
        raise ValueError("Invalid restypes.")

    # Construct a mapping from chain integer indices to chain ID strings.
    chain_ids = {}
    for i in np.unique(chain_index):  # np.unique gives sorted output.
        if i >= PDB_MAX_CHAINS:
            raise ValueError(f"The PDB format supports at most {PDB_MAX_CHAINS} chains.")
        chain_ids[i] = PDB_CHAIN_IDS[i]

    pdb_lines.append(f"MODEL     {model}")
    atom_index = 1
    last_chain_index = chain_index[0]
    res_name_3 = None
    # Add all atom sites.
    for i in range(restype.shape[0]):
        current_chain_index = chain_index[i]
        restype[i] = 25

        res_is_gap_token = restype[i] == complex_constants.gap_token
        res_is_mask_token = (
            restype[i] == complex_constants.unknown_protein_token
            or restype[i] == complex_constants.unknown_nucleic_token
        )
        res_is_of_ignored_molecule_type = (
            residue_molecule_type[i, molecule_type_to_write] == 0
            if molecule_type_to_write in [0, 1]
            else False
        )
        if res_is_gap_token or res_is_mask_token or res_is_of_ignored_molecule_type:
            continue

        res_name_3 = complex_constants.restypes_3[restype[i]]

        # Close the previous chain if in a multichain PDB.
        if last_chain_index != chain_index[i]:
            chain_end = "TER"
            end_resname = res_name_3
            end_chain_id = chain_ids[last_chain_index]
            end_res_idx = residue_index[i] - 1
            chain_end_line = (
                f"{chain_end:<6}{str(atom_index):>5}      {end_resname:>3} "
                f"{end_chain_id:>1}{str(end_res_idx):>4}"
            )
            pdb_lines.append(chain_end_line)
            atom_index += 1  # Atom index increases at the TER symbol.

        atom_types = restype_name_to_full_atom_names[res_name_3]
        for atom_name, pos, mask, b_factor in zip(
            atom_types, atom_positions[i], atom_mask[i], b_factors[i]
        ):
            if mask < 0.5:
                continue

            record_type = "ATOM"
            name = atom_name if len(atom_name) == 4 else f" {atom_name}"
            alt_loc = ""
            insertion_code = ""
            occupancy = 1.00
            element = atom_name[0]  # Proteins support only C, N, O, and S atoms, so this works.
            charge = ""
            # PDB is a columnar format, every space matters here!
            atom_line = (
                f"{record_type:<6}{str(atom_index):>5} {name:<4}{alt_loc:>1}"
                f"{res_name_3:>3} {chain_ids[current_chain_index]:>1}"
                f"{str(residue_index[i]):>4}{insertion_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                f"{element:>2}{charge:>2}"
            )
            pdb_lines.append(atom_line)
            atom_index += 1

        last_chain_index = chain_index[i]

    # Close the final chain.
    if res_name_3 is None:
        # note: indicates that the complex only consists of a single type of molecule
        return ""
    chain_end = "TER"
    end_resname = res_name_3
    end_chain_id = chain_ids[last_chain_index]
    end_res_idx = residue_index[i]
    chain_end_line = (
        f"{chain_end:<6}{str(atom_index):>5}      {end_resname:>3} "
        f"{end_chain_id:>1}{str(end_res_idx):>4}"
    )
    pdb_lines.append(chain_end_line)
    pdb_lines.append("ENDMDL")
    if add_end:
        pdb_lines.append("END")

    # Pad all lines to 80 characters.
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    return "\n".join(pdb_lines) + "\n"  # Add terminating newline.
