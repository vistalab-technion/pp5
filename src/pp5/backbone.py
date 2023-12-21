from __future__ import annotations

import math
import itertools
from typing import Dict, Union, Optional, Sequence
from contextlib import contextmanager

import numpy as np
from Bio.PDB.Atom import Atom, DisorderedAtom
from Bio.PDB.Residue import Residue

BACKBONE_ATOM_N = "N"
BACKBONE_ATOM_CA = "CA"
BACKBONE_ATOM_C = "C"
BACKBONE_ATOM_O = "O"
BACKBONE_ATOMS = (BACKBONE_ATOM_N, BACKBONE_ATOM_CA, BACKBONE_ATOM_C)
BACKBONE_ATOMS_O = tuple([*BACKBONE_ATOMS, BACKBONE_ATOM_O])
NO_ALTLOC = "~"
NORMALIZED_POSTFIX = "_norm"

CONST_8PI2 = math.pi * math.pi * 8


AltlocAtom = Union[Atom, DisorderedAtom]


def atom_location_sigma(atom: Atom) -> float:
    """
    Returns the standard deviation (sigma) in Angsroms, of the location of an atom,
    based on its isotropic B-factor.

    This is based on
        B = 8*pi^2 * U^2
    where B is the isotropic B-factor and U is the mean-square displacement (or
    variance) of the atom location, in units of A^2.

    :param atom: The atom to calculate the sigma for.
    :return: The sigma in Angstroms.
    """
    return math.sqrt(atom.get_bfactor() / CONST_8PI2)


def residue_backbone_atoms(res: Residue) -> Sequence[Atom]:
    """
    Returns a list of all backbone atoms in a residue.

    :param res: The residue to check.
    :return: The list of backbone atoms.
    """
    return tuple(a for a in res.get_atoms() if a.get_name() in BACKBONE_ATOMS)


def atom_altloc_ids(
    *atoms: Sequence[AltlocAtom],
    allow_disjoint: bool = False,
    include_none: bool = False,
) -> Sequence[str]:
    """
    Returns a list of all altloc ids which exist in a list of atoms.

    :param atoms: The atoms to check.
    :param allow_disjoint: Whether to return altloc ids which are not present in all
    the given atoms. If False (default) only the common altloc ids are returned.
    :param include_none: Whether to include the special id NO_ALTLOC in the output.
    :return: The list of altloc ids.
    """

    if not atoms:
        return tuple()

    per_atom_altloc_ids: Sequence[set] = [
        set(a.disordered_get_id_list()) if isinstance(a, DisorderedAtom) else set()
        for a in atoms
    ]

    if not allow_disjoint:
        altloc_ids = set.intersection(*per_atom_altloc_ids)
    else:
        altloc_ids = set.union(*per_atom_altloc_ids)

    if include_none:
        altloc_ids.add(NO_ALTLOC)

    return tuple(sorted(altloc_ids))


def residue_altloc_ids(
    res: Residue,
    backbone_only: bool = True,
    allow_disjoint: bool = False,
    include_none: bool = False,
) -> Sequence[str]:
    """
    Returns a list of all altloc ids which exist in a residue.

    :param res: The residue to check.
    :param backbone_only: Whether to only check backbone atoms.
    :param allow_disjoint: Whether to return altloc ids which are not present in all
    the relevant atoms. If False (default) only the common altloc ids are returned.
    :param include_none: Whether to return a sequence with the special altloc id
    NO_ALTLOC which represents that no altlocs are defined.
    """
    atoms = tuple(res.get_atoms() if not backbone_only else residue_backbone_atoms(res))
    return atom_altloc_ids(
        *atoms, allow_disjoint=allow_disjoint, include_none=include_none
    )


def residue_backbone_coords(
    res: Residue, with_oxygen: bool = False, with_altlocs: bool = False
) -> Dict[str, Optional[np.ndarray]]:
    """
    Returns the backbone atom locations of a Residue.

    :param res: A Residue.
    :param with_oxygen: Whether to include the oxygen atom.
    :param with_altlocs: Whether to include backbone locations for each altloc.
    :return: A dictionary mapping atom name (with possible altloc) to location, e.g.:
    CA -> [x1, y1, z1]
    CA_A -> [x2, y2, z2]
    CA_B -> [x3, y3, z3]
    """
    atom_names = BACKBONE_ATOMS_O if with_oxygen else BACKBONE_ATOMS
    atoms: Dict[str, Optional[AltlocAtom]] = {
        atom_name: res[atom_name] if atom_name in res else None
        for atom_name in atom_names
    }

    altloc_ids = [NO_ALTLOC]
    if with_altlocs:
        altloc_ids = atom_altloc_ids(
            *[a for a in atoms.values() if a is not None],
            allow_disjoint=True,
            include_none=True,
        )

    coords = {}
    for atom_name, atom in atoms.items():
        for altloc_id in altloc_ids:
            altloc_postfix = "" if altloc_id == NO_ALTLOC else f"_{altloc_id}"
            atom_name_altloc = f"{atom_name}{altloc_postfix}"

            with altloc_ctx(atom, altloc_id) as _atom:
                c = _atom.coord if _atom is not None else None
                coords[atom_name_altloc] = c

    return coords


@contextmanager
def altloc_ctx(atom: AltlocAtom, altloc_id: str) -> Optional[Atom]:
    """
    Context that sets and then restores the selected altloc for a potentially
    disordered atom, and yields the selected atom.
    If the atom is not disordered or if the altloc id is NO_ALTLOC,
    yields the given atom as is.

    :param atom: The atom to set the altloc for.
    :param altloc_id: The altloc id to select.
    :return: The selected atom or None if the altloc id does not exist.
    """
    if isinstance(atom, DisorderedAtom) and altloc_id != NO_ALTLOC:
        selected_altloc = atom.get_altloc()

        if atom.disordered_has_id(altloc_id):
            atom.disordered_select(altloc_id)
            yield atom.selected_child

        else:
            yield None

        atom.disordered_select(selected_altloc)
    else:
        yield atom


@contextmanager
def altloc_ctx_all(
    atoms: Sequence[AltlocAtom], altloc_id: str
) -> Sequence[Optional[Atom]]:
    """
    Context that sets and then restores the selected altloc for a list of
    potentially disordered atoms. A new list is yielded, containing the selected atoms.

    :param atoms: The atoms to set the altloc for.
    :param altloc_id: The altloc id to select.
    """
    if not atoms:
        yield []
    else:
        with altloc_ctx(atoms[0], altloc_id) as a0:
            with altloc_ctx_all(atoms[1:], altloc_id) as a1_to_aN:
                yield [a0, *a1_to_aN]


def get_selected_altloc(atom: AltlocAtom) -> Optional[str]:
    """
    Returns the altloc id selected for an atom. If the atom is not disordered, returns
    None.
    """
    if not isinstance(atom, DisorderedAtom):
        return None

    a_to_id = {child_atom: child_id for child_id, child_atom in atom.child_dict.items()}
    selected_id = a_to_id[atom.selected_child]
    return selected_id


def verify_altloc(atoms: Sequence[AltlocAtom], altloc_id: str):
    """
    Verifies that all given atoms have the same altloc id selected if they are
    disordered. Raises an AssertionError if not. Regular (non disordered) atoms are
    ignored.

    :param atoms: The atoms to check.
    :param altloc_id: The altloc id to check with.
    """
    for a in atoms:
        if isinstance(a, DisorderedAtom):
            selected_altloc = get_selected_altloc(a)
            if selected_altloc != altloc_id:
                raise ValueError(
                    f"Atom {a} has {selected_altloc=} but expected {altloc_id=}"
                )


def verify_not_disordered(atoms: Sequence[AltlocAtom]):
    """
    Verifies that none of the given atoms are disordered. Raises an error otherwise.
    :param atoms: The atoms to check.
    """

    for a in atoms:
        if isinstance(a, DisorderedAtom):
            raise ValueError(f"Atom {a} is disordered")


def verify_disordered_selection(
    atoms: Sequence[Optional[AltlocAtom]],
    selected_atoms: Sequence[Optional[Atom]],
    altloc_id: str,
):
    """
    Verifies that the given selected atoms match the given altloc id of the
    (potentially) disordered atoms.

    :param atoms: The potentially disordered atoms.
    :param selected_atoms: The selected atoms.
    :param altloc_id: The altloc id that was selected.
    """
    assert len(atoms) == len(selected_atoms)

    for atom, selected_atom in zip(atoms, selected_atoms):
        if atom is None:
            assert selected_atom is None

        elif isinstance(atom, DisorderedAtom):
            if selected_atom is None:
                # No atom was selected - make sure the altloc is indeed missing in
                # this disordered atom.
                assert not atom.disordered_has_id(altloc_id)
            else:
                # An atom was selected - make sure it's the correct one.
                assert atom.disordered_has_id(altloc_id)
                assert atom.selected_child == selected_atom
                assert atom.name == selected_atom.name

        elif isinstance(atom, Atom):
            assert atom == selected_atom

        else:
            raise ValueError(f"Unexpected atom type for {atom}")


def residue_altloc_sigmas(
    res: Residue, atom_names: Sequence[str] = BACKBONE_ATOMS
) -> Dict[str, Dict[str, float]]:
    """
    Calculates the standard deviation (sigma) in Angstroms, of the location of each
    atom per altloc in a residue.

    :param res:
    :param atom_names:
    :return: A nested dict, mapping atom_name -> altloc_id -> sigma.
    """
    assert atom_names

    sigmas: Dict[str, Dict[str, float]] = {atom_name: {} for atom_name in atom_names}
    for atom_name in atom_names:
        if atom_name not in res:
            continue
        atom: AltlocAtom = res[atom_name]
        for altloc_id in atom_altloc_ids(atom):
            with altloc_ctx(atom, altloc_id) as _atom:
                sigmas[atom_name][altloc_id] = atom_location_sigma(_atom)

    return sigmas


def residue_altloc_ca_dists(res: Residue, normalize: bool = False) -> Dict[str, float]:
    """
    Calculates the pairwise distances between CA atoms in a residue, for each altloc.
    :param res: The residue to check.
    :param normalize: Whether to normalize the distances by the isotropic B-factors of
        the atoms. If true, the distance between altlocs A and B will be
        d_AB / sqrt( sigma_A * sigma_B )
        where d_AB is the CA-CA distance between altlocs A and B, and sigma_A/B
        are their isotropic B-factors in angstroms.
        The normalized distances will be added to the output in addition to the
        regular distances, with a NORMALIZED_POSTFIX suffix.
    :return: A dictionary mapping two joined altloc ids (e.g. "AB") to the pairwise
        distances between CA in altlocs A and B.
    """

    sigmas: Optional[Dict[str, float]]  # altloc_id -> sigma
    if normalize:
        sigmas = residue_altloc_sigmas(res, atom_names=[BACKBONE_ATOM_CA])[
            BACKBONE_ATOM_CA
        ]
        altloc_ids = sigmas.keys()
    else:
        sigmas = None
        altloc_ids = atom_altloc_ids(res[BACKBONE_ATOM_CA])

    # Get the location of each CA atom per altloc that exists for the CA atom.
    ca_locations: Dict[str, np.ndarray] = {}  # altloc_id -> location
    for altloc_id in altloc_ids:
        ca: AltlocAtom = res[BACKBONE_ATOM_CA]
        with altloc_ctx(ca, altloc_id) as _ca:
            ca_locations[altloc_id] = _ca.get_coord()

    # Calculate the pairwise distances between CA atoms of altlocs.
    ca_dists: Dict[str, np.ndarray] = {}
    for altloc_id1, altloc_id2 in itertools.combinations(ca_locations.keys(), 2):
        dist = np.linalg.norm(ca_locations[altloc_id1] - ca_locations[altloc_id2])
        ca_dists[f"{altloc_id1}{altloc_id2}"] = dist.item()

        if normalize:
            dist = dist / np.sqrt(sigmas[altloc_id1] * sigmas[altloc_id2])
            ca_dists[f"{altloc_id1}{altloc_id2}{NORMALIZED_POSTFIX}"] = dist.item()

    return ca_dists


def residue_altloc_peptide_bond_lengths(
    res1: Residue, res2: Optional[Residue], normalize: bool = False
) -> Dict[str, float]:
    """
    Calculates the peptide bond lengths between two residues.
    If any or both residues contain altlocs, the peptide bond lengths will be
    calculated for each combination of altlocs from the first and second residues.

    :param res1: The first residue.
    :param res2: The residue immediately following res1, or None if res1 is the last
    residue.
    :param normalize: Whether to normalize the distances by the isotropic B-factors
    of the atom locations. The normalized distances will be added to the output with a
    NORMALIZED_POSTFIX suffix. The normalization factor is
    sqrt(sigma(res1_c) * sigma(res2_n)).
    :return: A dictionary mapping two joined altloc ids (e.g. "AB") to peptide bond
    length. In case one or both have no altlocs, the NO_ALTLOC string will be used
    as a placeholder.
    """

    # Pathological cases
    if res2 is None or BACKBONE_ATOM_C not in res1 or BACKBONE_ATOM_N not in res2:
        return {}

    # Get atoms participating in the peptide bond
    res1_c: AltlocAtom = res1[BACKBONE_ATOM_C]
    res2_n: AltlocAtom = res2[BACKBONE_ATOM_N]

    # Get list of altlocs for each residue, with the special altloc id NO_ALTLOC if
    # there are no altlocs for either of them.
    res1_altloc_ids, res2_altloc_ids = [
        atom_altloc_ids(a, include_none=True) for a in [res1_c, res2_n]
    ]

    # Iterate over all combinations of altlocs and calculate peptide bond lengths.
    peptide_bond_lengths: Dict[str, float] = {}
    for res1_altloc_id, res2_altloc_id in itertools.product(
        res1_altloc_ids, res2_altloc_ids
    ):
        _res1_c: Atom
        _res2_n: Atom
        with (
            altloc_ctx(res1_c, res1_altloc_id) as _res1_c,
            altloc_ctx(res2_n, res2_altloc_id) as _res2_n,
        ):
            # Both should not be none because we used the specific atoms to get the
            # altloc ids. It's possible the one or both didn't have altlocs, in which
            # case we'll have res1_c==_rec1_c and res2_n==_res2_n.
            assert _res1_c is not None and _res2_n is not None

            # peptide bond length
            pb_len = np.linalg.norm(_res1_c.get_coord() - _res2_n.get_coord())
            peptide_bond_lengths[f"{res1_altloc_id}{res2_altloc_id}"] = pb_len.item()

            if normalize:
                # Normalize by the isotropic B-factors of the atoms.
                pb_len = pb_len / np.sqrt(
                    atom_location_sigma(_res1_c) * atom_location_sigma(_res2_n)
                )
                peptide_bond_lengths[
                    f"{res1_altloc_id}{res2_altloc_id}{NORMALIZED_POSTFIX}"
                ] = pb_len.item()

    return peptide_bond_lengths
