AMINO_ACIDS = {
    "Ala": "A",
    "Ile": "I",
    "Arg": "R",
    "Leu": "L",
    "Asn": "N",
    "Lys": "K",
    "Asp": "D",
    "Met": "M",
    "Asx": "B",
    "Phe": "F",
    "Cys": "C",
    "Pro": "P",
    "Gln": "Q",
    "Ser": "S",
    "Glu": "E",
    "Thr": "T",
    "Glx": "Z",
    "Trp": "W",
    "Gly": "G",
    "Tyr": "Y",
    "His": "H",
    "Val": "V",
}

VARIANTS_FILTERS = {
    "Type": "single nucleotide variant",
    "Assembly": "GRCh38",
    "ReviewStatus": [
        "criteria provided, single submitter",
        "reviewed by expert panel",
        "practice guideline",
    ],
    "ClinicalSignificance": [
        "Pathogenic",
        "Likely pathogenic",
        "Benign",
        "Likely benign",
    ],
}

MAX_SEQUENCE_LENGTH = 2048