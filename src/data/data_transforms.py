# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from openfold (https://github.com/aqlaboratory/openfold):
# -------------------------------------------------------------------------------------------------------------------------------------

# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from functools import wraps

import numpy as np
import torch
from src.data.rigid_utils import Rigid, Rotation

from src.data import nucleotide_constants as nc
from src.data import vocabulary
from src.data.complex_constants import NUM_PROT_NA_TORSIONS

MSA_FEATURE_NAMES = [
    "msa",
    "deletion_matrix",
    "msa_mask",
    "msa_row_mask",
    "bert_mask",
    "true_msa",
]

def curry1(f):
    """Supply all arguments but the first."""

    @wraps(f)
    def fc(*args, **kwargs):
        return lambda x: f(x, *args, **kwargs)

    return fc

def batched_gather(data, inds, dim=0, no_batch_dims=0):
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [
        slice(None) for _ in range(len(data.shape) - no_batch_dims)
    ]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]

def atom23_list_to_atom27_list(batch, atom23_data_names, inplace=False):
    aatype = batch["aatype"].to(torch.long)
    assert (
        21 <= aatype.min() <= aatype.max() <= 26
    ), "Only nucleic acid residue inputs are allowed in `atom23_list_to_atom27_list()`."

    atom27_data_list = []
    for data_name in atom23_data_names:
        atom23 = batch[data_name]
        atom23 = atom23.view(
            *atom23.shape[:2], -1
        )  # note: must be of shape [batch_size, num_nodes, -1]
        atom27_data = batched_gather(
            atom23,
            batch["residx_atom27_to_atom23"],
            dim=-2,
            no_batch_dims=len(atom23.shape[:-2]),
        )

        atom27_data = (atom27_data * batch["atom27_atom_exists"][..., None]).squeeze(-1)
        if inplace:
            batch[data_name] = atom27_data
        atom27_data_list.append(atom27_data)

    return atom27_data_list

def construct_aatype9_non_deoxy_offsets(aatype):
    # build the aatype9 index offsets for amino acids of non-DNA nucleic acid molecules (e.g., RNA)
    non_deoxy_offsets = torch.zeros_like(aatype)
    non_deoxy_offsets[aatype == 0] = 4
    non_deoxy_offsets[aatype == 1] = 5
    non_deoxy_offsets[aatype == 2] = 6
    non_deoxy_offsets[aatype == 3] = 7
    non_deoxy_offsets[aatype == 4] = 7
    non_deoxy_offsets[aatype == 8] = 8
    return non_deoxy_offsets


def convert_na_aatype6_to_aatype9(
    aatype, deoxy_offset_mask=None, return_na_within_original_range=False
):
    aatype_inputs_present = aatype.numel() > 0
    if aatype_inputs_present and aatype.min() > vocabulary.protein_restype_num:
        aatype -= vocabulary.protein_restype_num + 1
    if aatype_inputs_present:
        assert (
            0 <= aatype.min() <= aatype.max() <= 5
        ), "Only nucleic acid residue inputs are allowed in `convert_na_aatype6_to_aatype9()`."
    aatype[
        aatype == nc.NA_AATYPE6_MASK_RESIDUE_INDEX
    ] = (
        nc.NA_AATYPE9_MASK_RESIDUE_INDEX
    )  # re-number mask token for all nucleic acid molecule types
    if deoxy_offset_mask is not None:
        non_deoxy_offsets = construct_aatype9_non_deoxy_offsets(aatype)
        aatype[~deoxy_offset_mask] = non_deoxy_offsets[~deoxy_offset_mask]
    if aatype_inputs_present:
        assert 0 <= aatype.min() <= aatype.max() <= 8
        if return_na_within_original_range:
            aatype += vocabulary.protein_restype_num + 1
    return aatype

def atom27_to_frames(na, eps=1e-8):
    aatype = na["aatype"].clone()
    all_atom_positions = na["all_atom_positions"] # in ATOM27 format
    all_atom_mask = na["all_atom_mask"] # NOTE: [1]_i if atom i exists in the residue's molecular structure (out of 27)
    na_deoxy = na["atom_deoxy"]

    assert (
        21 <= aatype.min() <= aatype.max() <= 26
    ), "Only nucleic acid residue inputs are allowed in `atom27_to_frames()`."
    aatype = convert_na_aatype6_to_aatype9(aatype, deoxy_offset_mask=na_deoxy)

    batch_dims = len(aatype.shape[:-1])
    # 9 NT types (i.e., DA, DC, DG, DT, A, C, G, U, plus x for missing residues), 11 groups
    nttype_rigidgroup_base_atom_names = np.full([9, 11, 3], "", dtype=object)
    # atoms that constitute backbone frame 1
    nttype_rigidgroup_base_atom_names[:, 0, :] = ["O4'", "C4'", "C3'"] # NOTE: you can change this to alter the frame design

    for restype, restype_letter in enumerate(nc.restypes):
        # keep one-letter format for DNA/RNA
        resname = restype_letter # ["DA", "DC", "DG", "DT", "A", "C", "G", "U"]
        
        for torsion_idx in range(NUM_PROT_NA_TORSIONS):
            if nc.chi_angles_mask[resname][torsion_idx]:
                names = nc.chi_angles_atoms[resname][torsion_idx]
                if (names): # note: DNA molecules do not have `["N9"/"N1", "C1'", "C2'", "O2'"]` frames
                    nttype_rigidgroup_base_atom_names[restype, torsion_idx + 1, :] = names[1:]

    nttype_rigidgroup_mask = all_atom_mask.new_zeros((*aatype.shape[:-1], 9, 11),)
    nttype_rigidgroup_mask[..., 0] = 1
    nttype_rigidgroup_mask[..., :8, 1:] = all_atom_mask.new_tensor(
        # note: Python 3.8 natively maintains key-value insertion order when iterating through dictionaries
        list(nc.chi_angles_mask.values())
    )

    lookuptable = nc.atom_order.copy()
    lookuptable[""] = 0
    lookup = np.vectorize(lambda x: lookuptable[x])
    # get atom index in atom_types that is defined in nucleotide_constants
    nttype_rigidgroup_base_atom27_idx = lookup(
        nttype_rigidgroup_base_atom_names,
    )
    # 9 (NT types) * 10 (torsions) * 3 (frame atom indices)
    nttype_rigidgroup_base_atom27_idx = aatype.new_tensor(
        nttype_rigidgroup_base_atom27_idx,
    )
    # # 1 * 9 (NT types) * 10 (torsions) * 3 (frame atom indices)
    nttype_rigidgroup_base_atom27_idx = nttype_rigidgroup_base_atom27_idx.view(
        *((1,) * batch_dims), *nttype_rigidgroup_base_atom27_idx.shape
    )
    # # N * 9 (NT types) * 10 (torsions) * 3 (frame atom indices)
    ntidx_rigidgroup_base_atom27_idx = batched_gather(
        nttype_rigidgroup_base_atom27_idx,
        aatype.to(torch.long),
        dim=-3,
        no_batch_dims=batch_dims,
    )
    base_atom_pos = batched_gather(
        all_atom_positions,
        ntidx_rigidgroup_base_atom27_idx.to(torch.long),
        dim=-2,
        no_batch_dims=len(all_atom_positions.shape[:-2]),
    )
    # # 0, 1, 2 are the index of frame atoms
    gt_frames = Rigid.from_3_points(
        p_neg_x_axis=base_atom_pos[..., 0, :],
        origin=base_atom_pos[..., 1, :],
        p_xy_plane=base_atom_pos[..., 2, :],
        eps=eps,
    )

    group_exists = batched_gather(
        nttype_rigidgroup_mask,
        aatype.type(torch.long),
        dim=-2,
        no_batch_dims=batch_dims,
    )

    gt_atoms_exist = batched_gather(
        all_atom_mask,
        ntidx_rigidgroup_base_atom27_idx.to(torch.long),
        dim=-1,
        no_batch_dims=len(all_atom_mask.shape[:-1]),
    )
    gt_exists = torch.min(gt_atoms_exist, dim=-1)[0] * group_exists

    rots = torch.eye(3, dtype=all_atom_mask.dtype, device=aatype.device)
    rots = torch.tile(rots, (*((1,) * batch_dims), 11, 1, 1))
    rots[..., 0, 0, 0] = -1
    rots[..., 0, 2, 2] = -1
    rots = Rotation(rot_mats=rots)

    gt_frames = gt_frames.compose(Rigid(rots, None))

    gt_frames_tensor = gt_frames.to_tensor_4x4()

    residx_rigidgroup_is_ambiguous = torch.zeros_like(group_exists)
    alt_gt_frames_tensor = torch.zeros_like(gt_frames_tensor)

    na["rigidgroups_gt_frames"] = gt_frames_tensor
    na["rigidgroups_gt_exists"] = gt_exists
    na["rigidgroups_group_exists"] = group_exists
    na["rigidgroups_group_is_ambiguous"] = residx_rigidgroup_is_ambiguous
    na["rigidgroups_alt_gt_frames"] = alt_gt_frames_tensor

    return na

def make_atom23_masks(na):
    """Construct denser atom positions (23 dimensions instead of 27)."""
    restype_atom23_to_atom27 = []
    restype_atom27_to_atom23 = []
    restype_atom23_mask = []

    na_aatype = na["aatype"].to(torch.long).clone()
    na_deoxy = na["atom_deoxy"].to(torch.bool)
    assert (
        21 <= na_aatype.min() <= na_aatype.max() <= 26
    ), "Only nucleic acid residue inputs are allowed in `make_atom23_masks()`."

    for nt in nc.restypes:
        atom_names = nc.restype_name_to_compact_atom_names[nt]
        restype_atom23_to_atom27.append(
            [(nc.atom_order[name] if name else 0) for name in atom_names]
        )
        atom_name_to_idx23 = {name: i for i, name in enumerate(atom_names)}
        restype_atom27_to_atom23.append(
            [
                (atom_name_to_idx23[name] if name in atom_name_to_idx23 else 0)
                for name in nc.atom_types
            ]
        )

        restype_atom23_mask.append([(1.0 if name else 0.0) for name in atom_names])

    # Add dummy mapping for restype 'UNK'
    restype_atom23_to_atom27.append([0] * 23)
    restype_atom27_to_atom23.append([0] * 27)
    restype_atom23_mask.append([0.0] * 23)

    restype_atom23_to_atom27 = torch.tensor(
        restype_atom23_to_atom27,
        dtype=torch.int32,
        device=na["aatype"].device,
    )
    restype_atom27_to_atom23 = torch.tensor(
        restype_atom27_to_atom23,
        dtype=torch.int32,
        device=na["aatype"].device,
    )
    restype_atom23_mask = torch.tensor(
        restype_atom23_mask,
        dtype=torch.float32,
        device=na["aatype"].device,
    )

    assert (
        21 <= na_aatype.min() <= na_aatype.max() <= 26
    ), "Only nucleic acid residue inputs are allowed in `make_atom23_masks()`."
    na_aatype = convert_na_aatype6_to_aatype9(na_aatype, deoxy_offset_mask=na_deoxy)

    # create the mapping for (residx, atom23) --> atom27, i.e. an array
    # with shape (num_res, 23) containing the atom27 indices for this protein
    residx_atom23_to_atom27 = restype_atom23_to_atom27[na_aatype]
    residx_atom23_mask = restype_atom23_mask[na_aatype]

    na["atom23_atom_exists"] = residx_atom23_mask
    na["residx_atom23_to_atom27"] = residx_atom23_to_atom27.long()

    # create the gather indices for mapping back
    residx_atom27_to_atom23 = restype_atom27_to_atom23[na_aatype]
    na["residx_atom27_to_atom23"] = residx_atom27_to_atom23.long()

    # create the corresponding mask
    restype_atom27_mask = torch.zeros([9, 27], dtype=torch.float32, device=na["aatype"].device)
    for restype, restype_letter in enumerate(nc.restypes):
        restype_name = restype_letter
        atom_names = nc.residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = nc.atom_order[atom_name]
            restype_atom27_mask[restype, atom_type] = 1

    residx_atom27_mask = restype_atom27_mask[na_aatype]
    na["atom27_atom_exists"] = residx_atom27_mask

    return na

def get_chi_atom_indices(molecule_type):
    """Returns atom indices needed to compute chi angles for all residue types.

    Returns:
      A tensor of shape [residue_types=21, chis=4, atoms=4] for `molecule_type=protein` and
      [residue_types=9, chis=10, atoms=4] for `molecule_type=NA`. The residue types are
      in the order specified in pc.restypes (or nc.restypes) + unknown residue type at the end.
      For chi angles which are not defined on the residue, the positions indices
      are by default set to 0.
    """
    chi_atom_indices = []
    for residue_name in nc.restypes:
        residue_chi_angles = nc.chi_angles_atoms[residue_name]
        atom_indices = []
        for chi_angle in residue_chi_angles:
            if chi_angle is None:
                atom_indices.append(
                    [0, 0, 0, 0]
                )  # For DNA molecules that do not include a specific frame
            else:
                atom_indices.append([nc.atom_order[atom] for atom in chi_angle])
        for _ in range(NUM_PROT_NA_TORSIONS - len(atom_indices)):
            atom_indices.append([0, 0, 0, 0])  # For chi angles not defined on the NT.
        chi_atom_indices.append(atom_indices)

    chi_atom_indices.append([[0, 0, 0, 0]] * NUM_PROT_NA_TORSIONS)  # For UNKNOWN residue.

    return chi_atom_indices

@curry1
def atom27_to_torsion_angles(na, prefix="", randomly_noise_torsion_atoms_to_place=False):
    """Convert coordinates to torsion angles.

    This function is extremely sensitive to floating point imprecisions
    and should be run with double precision whenever possible.

    Args:
        Dict containing:
            * (prefix)aatype:
                [*, N_res] residue indices
            * (prefix)all_atom_positions:
                [*, N_res, 27, 3] atom positions (in atom27
                format)
            * (prefix)all_atom_mask:
                [*, N_res, 27] atom position mask
        Whether to randomly perturb the positions of atoms to place using torsion angles,
        to randomly rotate the corresponding atoms for debugging purposes.
    Returns:
        The same dictionary updated with the following features:

        "(prefix)torsion_angles_sin_cos" ([*, N_res, 7, 2])
            Torsion angles
        "(prefix)torsion_angles_mask" ([*, N_res, 7])
            Torsion angles mask
    """
    aatype = na[prefix + "aatype"]
    all_atom_positions = na[prefix + "all_atom_positions"]
    all_atom_mask = na[prefix + "all_atom_mask"]
    na_deoxy = na["atom_deoxy"]

    assert (
        21 <= aatype.max() <= 26
    ), "Only nucleic acid residue inputs are allowed in `atom27_to_torsion_angles()`."
    aatype = convert_na_aatype6_to_aatype9(aatype, deoxy_offset_mask=na_deoxy)

    chi_atom_indices = torch.as_tensor(get_chi_atom_indices("NA"), device=aatype.device)

    atom_indices = chi_atom_indices[..., aatype, :, :]
    chis_atom_pos = batched_gather(
        all_atom_positions, atom_indices, -2, len(atom_indices.shape[:-2])
    )

    chi_angles_mask = list(nc.chi_angles_mask.values())
    chi_angles_mask.append([0.0 for _ in range(NUM_PROT_NA_TORSIONS)])
    chi_angles_mask = all_atom_mask.new_tensor(chi_angles_mask)

    chis_mask = chi_angles_mask[aatype, :]

    chi_angle_atoms_mask = batched_gather(
        all_atom_mask,
        atom_indices,
        dim=-1,
        no_batch_dims=len(atom_indices.shape[:-2]),
    )
    chi_angle_atoms_mask = torch.prod(
        chi_angle_atoms_mask, dim=-1, dtype=chi_angle_atoms_mask.dtype
    )
    chis_mask = chis_mask * chi_angle_atoms_mask
    # In the order of delta, gamma, beta, alpha1, alpha2, tm, chi
    torsions_atom_pos = chis_atom_pos

    torsion_angles_mask = chis_mask

    torsion_frames = Rigid.from_3_points(
        torsions_atom_pos[..., 1, :],
        torsions_atom_pos[..., 2, :],
        torsions_atom_pos[..., 0, :],
        eps=1e-8,
    )

    fourth_atom_pos = (
        (torsions_atom_pos[..., 3, :] + torch.rand_like(torsions_atom_pos[..., 3, :]))
        if randomly_noise_torsion_atoms_to_place
        else torsions_atom_pos[..., 3, :]
    )
    fourth_atom_rel_pos = torsion_frames.invert().apply(fourth_atom_pos)

    torsion_angles_sin_cos = torch.stack(
        [fourth_atom_rel_pos[..., 2], fourth_atom_rel_pos[..., 1]], dim=-1
    )

    denom = torch.sqrt(
        torch.sum(
            torch.square(torsion_angles_sin_cos),
            dim=-1,
            dtype=torsion_angles_sin_cos.dtype,
            keepdims=True,
        )
        + 1e-8
    )
    torsion_angles_sin_cos = torsion_angles_sin_cos / denom

    torsion_angles_sin_cos = (
        torsion_angles_sin_cos
        * all_atom_mask.new_tensor(
            [1.0 for _ in range(NUM_PROT_NA_TORSIONS)],
        )[((None,) * len(torsion_angles_sin_cos.shape[:-2])) + (slice(None), None)]
    )
    alt_torsion_angles_sin_cos = torch.zeros_like(torsion_angles_sin_cos)

    na["torsion_angles_sin_cos"] = torsion_angles_sin_cos
    na["alt_torsion_angles_sin_cos"] = alt_torsion_angles_sin_cos
    na["torsion_angles_mask"] = torsion_angles_mask

    return na