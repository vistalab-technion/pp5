CODON_TYPE_ANY = "ANY"
SS_TYPE_ANY = "ANY"
SS_TYPE_HELIX = "HELIX"
SS_TYPE_SHEET = "SHEET"
SS_TYPE_TURN = "TURN"
SS_TYPE_OTHER = "OTHER"
SS_TYPES = (SS_TYPE_HELIX, SS_TYPE_SHEET, SS_TYPE_TURN, SS_TYPE_OTHER)
DSSP_TO_SS_TYPE = {
    # The DSSP codes for secondary structure used here are:
    # H        Alpha helix (4-12)
    # B        Isolated beta-bridge residue
    # E        Strand
    # G        3-10 helix
    # I        Pi helix
    # T        Turn
    # S        Bend
    # -        None
    "E": SS_TYPE_SHEET,
    "H": SS_TYPE_HELIX,
    "G": SS_TYPE_OTHER,  # maybe also helix?
    "I": SS_TYPE_OTHER,  # maybe also helix?
    "T": SS_TYPE_TURN,
    "S": SS_TYPE_OTHER,  # maybe also turn?
    "B": SS_TYPE_OTHER,  # maybe also sheet?
    "-": None,
    "": None,
}
