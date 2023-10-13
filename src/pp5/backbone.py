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
NO_ALTLOC = "_"
AltlocAtom = Union[Atom, DisorderedAtom]


CONST_8PI2 = math.pi * math.pi * 8


def residue_backbone_atoms(res: Residue) -> Sequence[Atom]:
    """
    Returns a list of all backbone atoms in a residue.

    :param res: The residue to check.
    :return: The list of backbone atoms.
    """
    return tuple(a for a in res.get_atoms() if a.get_name() in BACKBONE_ATOMS)


def atom_altloc_ids(
    atoms: Sequence[AltlocAtom], allow_disjoint: bool = False
) -> Sequence[str]:
    """
    Returns a list of all altloc ids which exist in a list of atoms.

    :param atoms: The atoms to check.
    :param allow_disjoint: Whether to return altloc ids which are not present in all
    the given atoms. If False (default) only the common altloc ids are returned.
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

    return tuple(sorted(altloc_ids))


def residue_altloc_ids(
    res: Residue, backbone_only: bool = True, allow_disjoint: bool = False
) -> Sequence[str]:
    """
    Returns a list of all altloc ids which exist in a residue.

    :param res: The residue to check.
    :param backbone_only: Whether to only check backbone atoms.
    :param allow_disjoint: Whether to return altloc ids which are not present in all
    the relevant atoms. If False (default) only the common altloc ids are returned.
    """
    atoms = tuple(res.get_atoms() if not backbone_only else residue_backbone_atoms(res))
    return atom_altloc_ids(atoms, allow_disjoint=allow_disjoint)


@contextmanager
def altloc_ctx(atom: AltlocAtom, altloc_id: str) -> Optional[Atom]:
    """
    Context that sets and then restores the selected altloc for a potentially
    disordered atom, and yields the selected atom.
    If the atom is not disordered, yields the atom as is.

    :param atom: The atom to set the altloc for.
    :param altloc_id: The altloc id to select.
    :return: The selected atom or None if the altloc id does not exist.
    """
    if isinstance(atom, DisorderedAtom):
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


def residue_altloc_ca_dists(res: Residue, normalize: bool = False) -> Dict[str, float]:
    """
    Calculates the pairwise distances between CA atoms in a residue, for each altloc.
    :param res: The residue to check.
    :param normalize: Whether to normalize the distances by the isotropic B-factors of
        the atoms. If true, the distance between altlocs A and B will be
        sqrt(d_AB^2 / ( sigma_A * sigma_B))
        where d_AB is the CA-CA distance between altlocs A and B, and sigma_A/B
        are their isotropic B-factors in angstroms.
    :return: A dictionary mapping two joined altloc ids (e.g. "AB") to the pairwise
        distances between CA in altlocs A and B.
    """
    ca_locations: Dict[str, np.ndarray] = {}
    sigmas: Dict[str, np.ndarray] = {}
    altloc_ids = residue_altloc_ids(res, backbone_only=True)

    for altloc_id in altloc_ids:
        ca: AltlocAtom = res[BACKBONE_ATOM_CA]
        with altloc_ctx(ca, altloc_id):
            ca_locations[altloc_id] = ca.get_coord()
            sigmas[altloc_id] = ca.get_bfactor() / CONST_8PI2  # convert to Angstroms

    ca_dists: Dict[str, np.ndarray] = {}
    for altloc_id1, altloc_id2 in itertools.combinations(ca_locations.keys(), 2):
        dist = np.linalg.norm(ca_locations[altloc_id1] - ca_locations[altloc_id2])

        if normalize:
            dist = np.sqrt(dist**2 / (sigmas[altloc_id1] * sigmas[altloc_id2]))

        ca_dists[f"{altloc_id1}{altloc_id2}"] = dist.item()

    return ca_dists
