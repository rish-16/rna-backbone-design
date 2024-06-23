# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for MMDiff (https://github.com/Profluent-Internships/MMDiff):
# -------------------------------------------------------------------------------------------------------------------------------------

# This is the standard residue order when coding residues type as a number.
restypes = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
    "X",
    "a",
    "c",
    "g",
    "t",
    "u",
    "x",
    "-",
    "_",
    "1",
    "2",
    "3",
    "4",
    "5",
]
restype_order = {restype: i for i, restype in enumerate(restypes)}

restype_1to3 = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
    "X": "UNK",
    "a": "A",
    "c": "C",
    "g": "G",
    "t": "T",
    "u": "U",
    "x": "unk",
    "-": "GAP",
    "_": "PAD",
    "1": "SP1",
    "2": "SP2",
    "3": "SP3",
    "4": "SP4",
    "5": "SP5",
}
restype_3to1 = {v: k for k, v in restype_1to3.items()}
restype_3to1.update(
    {
        "DA": "a",
        "DC": "c",
        "DG": "g",
        "DT": "t",
    }
)

protein_restypes = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]
protein_restype_order = {restype: i for i, restype in enumerate(protein_restypes)}

protein_restype_1to3 = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}
protein_restype_3to1 = {v: k for k, v in protein_restype_1to3.items()}

protein_restypes_with_x = protein_restypes + ["X"]
protein_restype_order_with_x = {restype: i for i, restype in enumerate(protein_restypes_with_x)}
nucleic_restypes = ["a", "c", "g", "t", "u"]
special_restypes = ["-", "_", "1", "2", "3", "4", "5"]
unknown_protein_restype = "X"
unknown_nucleic_restype = "x"
gap_token = restypes.index("-")
pad_token = restypes.index("_")

restype_num = len(restypes)  # := 34.
protein_restype_num = len(protein_restypes)  # := 20
protein_restype_num_with_x = len(protein_restypes_with_x)  # := 21

protein_resnames = [restype_1to3[r] for r in protein_restypes]
protein_resname_to_idx = {resname: i for i, resname in enumerate(protein_resnames)}

alternative_restypes_map = {
    # Protein
    "MSE": "MET",
}
allowable_restypes = set(restypes + list(alternative_restypes_map.keys()))


def is_protein_sequence(sequence: str) -> bool:
    """Check if a sequence is a protein sequence."""

    return all([s in protein_restypes for s in sequence])


def is_nucleic_sequence(sequence: str) -> bool:
    """Check if a sequence is a nucleic acid sequence."""

    return all([s in nucleic_restypes for s in sequence])
