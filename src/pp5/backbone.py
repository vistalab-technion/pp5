from __future__ import annotations

import itertools
from typing import Union, Optional, Sequence
from contextlib import contextmanager

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


def residue_backbone_atoms(res: Residue) -> Sequence[Atom]:
    """
    Returns a list of all backbone atoms in a residue.

    :param res: The residue to check.
    :return: The list of backbone atoms.
    """
    return tuple(a for a in res.get_atoms() if a.get_name() in BACKBONE_ATOMS)


def atom_altloc_ids(atoms: Sequence[AltlocAtom]) -> Sequence[str]:
    """
    Returns a list of all altloc ids which exist in a list of atoms.
    :param atoms: The atoms to check.
    :return: The list of altloc ids.
    """
    return sorted(
        set(
            itertools.chain(
                *[
                    a.disordered_get_id_list()
                    for a in atoms
                    if isinstance(a, DisorderedAtom)
                ]
            )
        )
    )


def residue_altloc_ids(res: Residue, backbone_only: bool = True) -> Sequence[str]:
    """
    Returns a list of all altloc ids which exist in a residue.

    :param res: The residue to check.
    :param backbone_only: Whether to only check backbone atoms.
    """
    atoms = tuple(res.get_atoms() if not backbone_only else residue_backbone_atoms(res))
    return atom_altloc_ids(atoms)


@contextmanager
def altloc_ctx(atom: AltlocAtom, altloc_id: str):
    """
    Context that sets and then restores the selected altloc for an atom.
    :param atom: The atom to set the altloc for.
    :param altloc_id: The altloc id to select.
    """
    if isinstance(atom, DisorderedAtom):
        selected_altloc = atom.get_altloc()
        if atom.disordered_has_id(altloc_id):
            atom.disordered_select(altloc_id)
        yield
        atom.disordered_select(selected_altloc)
    else:
        yield


@contextmanager
def altloc_ctx_all(atoms: Sequence[AltlocAtom], altloc_id: str):
    """
    Context that sets and then restores the selected altloc for a list of atoms.
    :param atoms: The atoms to set the altloc for.
    :param altloc_id: The altloc id to select.
    """
    if not atoms:
        yield None
    else:
        with altloc_ctx(atoms[0], altloc_id):
            with altloc_ctx_all(atoms[1:], altloc_id):
                yield None


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
            assert (
                selected_altloc == altloc_id
            ), f"Atom {a} has {selected_altloc=} but expected {altloc_id=}"
