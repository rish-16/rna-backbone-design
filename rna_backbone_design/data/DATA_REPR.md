# Guide to Molecular Representations

In our work, we use several different intermediate data representation schemes for RNA molecules. Every RNA nucleotide's backbone contains 13 atoms in total, comprising Carbon, Oxygen, Hydrogen, and Phosphorous. The different atomic representation formats rely on the index in an array that correspond to each unique atom type. Specifically, this array can be 37, 27, or 23 elements long, giving rise to the three formats `ATOM37`, `ATOM27`, and `ATOM23`. We use them interchangeably when going from frames to all-atom backbones. Here, we document these different schemes and what they comprise.

### ATOM23

This system describes the 23 (or fewer) heavy atoms in any RNA base (AUCG). Different bases have different combinations of atoms from this set of 23. Guanine (G) has the maximum number of heavy atoms, hence 23. To represent any base, a multi-hot-encoded vector comprising of 1s and 0s is created, corresponding to the heavy atoms present in each base. 

Here's the number of heavy atoms in each of the RNA bases:
```python
A: 22
U: 20
G: 23
C: 20
```

### ATOM27

Here, take the **unique** atom names from both DNA and RNA and comine them together. We are left with the following combined set of 27 heavy atoms:

```python
nucleic_acid_atom_types_in_order = [
    "C1'",
    "C2'",
    "C3'",
    "C4'",
    "C5'",
    "O5'",
    "O4'",
    "O3'",
    "O2'",
    "P",  
    "OP1",
    "OP2",
    "N1",
    "N2",
    "N3",
    "N4",
    "N6",
    "N7",
    "N9",
    "C2",
    "C4",
    "C5",
    "C6",
    "C8",
    "O2",
    "O4",
    "O6",
]
```

When representing a base in `ATOM27`, we rely on 1s in the relevant indexes and 0s otherwise. For example, Guanine in RNA has O2' but not in DNA. So the corresponding index of O2' would be 1 when representing an RNA nucleotide and 0 when representing a DNA nucleotide.

### ATOM37

This is the standard scheme from AlphaFold2/RFDiffusion to represent all the atoms present in proteins. To represent atomic coordinates for a protein residue's backbone, we use a sparse $37 \times 3$ tensor where only the relevant indices along the first dimension (of size 37) have $(x,y,z)$ coordinate values.

```python
protein_atom_types = [
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
]
```

When representing a nucleic acid, certain indices in this array (of size 37) are "overloaded" to represent an atom either from a protein or a nucleic acid. This way, we can continue using 37 instead of 64 (37 from proteins + 27 from nucleic acids). For instance, in RNA-FrameFlow, we use the following re-mapping to map RNA atom types into the `ATOM37` scheme:

```python
Protein ATOM37: ["N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", ...]
RNA ATOM37: ['C1'', 'C2'', 'C3'', 'C4'', 'C5'', '05'', '04'', 'O3'', 'O2'', 'P', 'OP1', 'OP2', 'N1', 'N2', 'N3', 'N4', 'N6', 'N7', 'N9', ...]
```

In `all_atom.py`, this remapping is done in the `compute_backbone` method:

```python
# note: nucleic acids' atom37 bb order = ['C1'', 'C2'', 'C3'', 'C4'', 'C5'', '05'', '04'', 'O3'', 'O2'', 'P', 'OP1', 'OP2', 'N1', 'N2', 'N3', 'N4', 'N6', 'N7', 'N9', ...]
#                                                         2      3      4             6      7            9    10     11     12
atom37_bb_pos[..., 2:4, :][is_na_residue_mask] = atom23_pos[..., :2, :][is_na_residue_mask]  # reindex C3' and C4'
atom37_bb_pos[..., 6, :][is_na_residue_mask] = atom23_pos[..., 2, :][is_na_residue_mask]  # reindex O4'
atom37_bb_pos[..., 1, :][is_na_residue_mask] = atom23_pos[..., 3, :][is_na_residue_mask]  # reindex C2'
atom37_bb_pos[..., 0, :][is_na_residue_mask] = atom23_pos[..., 4, :][is_na_residue_mask]  # reindex C1'
atom37_bb_pos[..., 4, :][is_na_residue_mask] = atom23_pos[..., 5, :][is_na_residue_mask]  # reindex C5'
atom37_bb_pos[..., 7, :][is_na_residue_mask] = atom23_pos[..., 6, :][is_na_residue_mask]  # reindex O3'
atom37_bb_pos[..., 5, :][is_na_residue_mask] = atom23_pos[..., 7, :][is_na_residue_mask]  # reindex O5'
atom37_bb_pos[..., 9:12, :][is_na_residue_mask] = atom23_pos[..., 8:11, :][is_na_residue_mask]  # reindex P, OP1, and OP2
atom37_bb_pos[..., 18, :][is_na_residue_mask] = atom23_pos[..., 11, :][is_na_residue_mask]  # reindex N9, which instead reindexes N1 for pyrimidine residues
```

This way, we can represent RNA molecules using `ATOM37` as well despite it being primarily used for proteins.