#!/usr/bin/env python3
"""
evaluate_docking.py
-------------------
Evaluate DiffDock-Pocket docking predictions against ground-truth complexes.

Metrics (per complex, then aggregated across complexes):
  Ligand
    - RMSD (symmetry-corrected via RDKit graph automorphisms)
    - Centroid distance
  Protein
    - Sidechain RMSD (non-backbone heavy atoms, matched by chain/resnum/icode/name)
  Clash
    - Fraction of ligand heavy atoms clashing with protein (VDW overlap)

Aggregate statistics computed for top-K (oracle best-of-K) predictions:
  mean, frac_below_2, frac_below_5, percentile_25/50/75

Result directory layout expected:
  <results_dir>/
    index{N}___{pdb_id}_protein_esmfold_aligned_tr_fix.pdb___{pdb_id}_ligand.mol2/
      rank1_confidence-2.50.sdf
      rank1_confidence-2.50_protein.pdb
      rank2_confidence-2.75.sdf
      ...

Ground-truth directory layout expected:
  <data_dir>/
    {pdb_id}/
      {pdb_id}_ligand.sdf   (or .mol2)
      {pdb_id}_protein_esmfold_aligned_tr_fix.pdb

Usage:
  python evaluate_docking.py \\
      --results_dir DiffDock-Pocket/results/user_inference_2641194.pbs-7 \\
      --data_dir    DiffDock-Pocket/data/PDBBIND_atomCorrected \\
      --output      metrics.csv \\
      --top_k 1 5 10 \\
      --clash_overlap 0.75 \\
      --superimpose_backbone
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# ── BioPython (required for protein parsing) ──────────────────────────────────
try:
    from Bio.PDB import PDBParser
    from Bio.PDB.Structure import Structure
except ImportError:
    print(
        "ERROR: Biopython is required.\n"
        "  pip install biopython\n"
        "  or: conda install -c conda-forge biopython"
    )
    sys.exit(1)

# ── RDKit (required for ligand handling) ──────────────────────────────────────
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolAlign

    RDLogger.DisableLog("rdApp.*")
except ImportError:
    print(
        "ERROR: RDKit is required.\n"
        "  conda install -c conda-forge rdkit\n"
        "  or: pip install rdkit"
    )
    sys.exit(1)

# ── Optional: tqdm progress bar ───────────────────────────────────────────────
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):  # type: ignore[misc]
        return it


# =============================================================================
# Constants
# =============================================================================

# Bondi VDW radii (Angstrom)
VDW_RADII: Dict[str, float] = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "P": 1.80, "S": 1.80, "Cl": 1.75, "Br": 1.85, "I": 1.98,
    "Si": 2.10, "Fe": 1.80, "Zn": 1.39, "Ca": 1.74, "Mg": 1.73,
    "Mn": 1.73, "Cu": 1.40, "Co": 1.63, "Se": 1.90, "B": 1.92,
    "As": 1.85, "Li": 1.82, "Na": 2.27, "K": 2.75,
}
DEFAULT_VDW = 1.70  # fallback for unknown elements

# Backbone atom names — everything else is a sidechain atom
BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}

# Shared parser instance; QUIET=True suppresses harmless PDB format warnings
_PDB_PARSER = PDBParser(QUIET=True)


# =============================================================================
# BioPython structure loading and atom iteration
# =============================================================================

def load_structure(pdb_path: Path, structure_id: str = "s") -> Structure:
    """Parse a PDB file into a BioPython Structure (first model is used throughout)."""
    return _PDB_PARSER.get_structure(structure_id, str(pdb_path))


def _iter_heavy_atoms(structure: Structure, protein_only: bool = False):
    """
    Yield (atom, element_str) for every heavy atom in the first model.

    BioPython residue IDs are (hetflag, resseq, icode):
      - hetflag == ' '      -> standard ATOM residue (protein / nucleic acid)
      - hetflag == 'W'      -> water
      - hetflag == 'H_XXX'  -> HETATM (ligand / ion)

    Disordered atoms: residue.get_unpacked_list() unpacks DisorderedAtom
    objects into plain Atom objects, one per altloc, so we can simply filter
    on get_altloc() instead of navigating the disordered hierarchy ourselves.

    Parameters
    ----------
    protein_only : bool
        If True, skip HETATM and water residues (hetflag != ' ').
    """
    model = structure[0]
    for chain in model:
        for residue in chain:
            hetflag = residue.get_id()[0]
            if protein_only and hetflag.strip() != "":
                continue
            for atom in residue.get_unpacked_list():
                # Keep only the primary altloc (' ' = no disorder, 'A' = first alt)
                if atom.get_altloc() not in (" ", "A"):
                    continue
                element = (atom.element or "").strip().capitalize()
                if not element:
                    # Fall back to first alphabetic character of the atom name
                    name = atom.get_name().lstrip("0123456789")
                    element = name[0].capitalize() if name else "C"
                if element.upper() == "H":
                    continue
                yield atom, element


def _sidechain_coord_dict(structure: Structure) -> Dict[Tuple, np.ndarray]:
    """
    Build a lookup from (chain_id, resseq, icode, atom_name) to coordinates
    for every sidechain heavy atom (ATOM records, non-backbone, non-H).

    Including icode in the key correctly handles insertion-code residues
    (e.g. 100A, 100B) that share the same resseq.
    """
    d: Dict[Tuple, np.ndarray] = {}
    model = structure[0]
    for chain in model:
        for residue in chain:
            # Skip HETATM / water
            if residue.get_id()[0].strip() != "":
                continue
            _, resseq, icode = residue.get_id()
            chain_id = chain.get_id()
            for atom in residue.get_unpacked_list():
                if atom.get_altloc() not in (" ", "A"):
                    continue
                if (atom.element or "").strip().upper() == "H":
                    continue
                if atom.get_name() in BACKBONE_ATOMS:
                    continue
                key = (chain_id, resseq, icode.strip(), atom.get_name())
                d[key] = atom.get_coord()
    return d


def _ca_coord_dict(structure: Structure) -> Dict[Tuple, np.ndarray]:
    """
    Build a lookup from (chain_id, resseq, icode) to Cα coordinates
    for every standard protein residue.
    """
    d: Dict[Tuple, np.ndarray] = {}
    model = structure[0]
    for chain in model:
        for residue in chain:
            if residue.get_id()[0].strip() != "":
                continue
            if not residue.has_id("CA"):
                continue
            ca = residue["CA"]
            # If CA is disordered, select altloc 'A' or the first available
            if ca.is_disordered():
                alt_ids = ca.disordered_get_id_list()
                ca = ca.disordered_get("A" if "A" in alt_ids else alt_ids[0])
            _, resseq, icode = residue.get_id()
            d[(chain.get_id(), resseq, icode.strip())] = ca.get_coord()
    return d


def match_sidechain_atoms(
    pred: Structure, gt: Structure
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pair sidechain heavy atoms between predicted and GT structures by
    (chain_id, resseq, icode, atom_name). Unmatched atoms are silently dropped,
    so missing or extra residues in either structure are handled gracefully.

    Returns
    -------
    pred_coords, gt_coords : ndarray (N, 3) each
    """
    pred_d = _sidechain_coord_dict(pred)
    gt_d   = _sidechain_coord_dict(gt)
    common = sorted(set(pred_d) & set(gt_d))
    if not common:
        return np.empty((0, 3)), np.empty((0, 3))
    return (
        np.array([pred_d[k] for k in common]),
        np.array([gt_d[k]   for k in common]),
    )


def match_ca_atoms(
    pred: Structure, gt: Structure
) -> Tuple[np.ndarray, np.ndarray]:
    """Pair Cα atoms between predicted and GT by (chain_id, resseq, icode)."""
    pred_d = _ca_coord_dict(pred)
    gt_d   = _ca_coord_dict(gt)
    common = sorted(set(pred_d) & set(gt_d))
    if not common:
        return np.empty((0, 3)), np.empty((0, 3))
    return (
        np.array([pred_d[k] for k in common]),
        np.array([gt_d[k]   for k in common]),
    )


# =============================================================================
# Kabsch superimposition
# =============================================================================

def kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Optimal rotation matrix R (3x3) such that P @ R ≈ Q (Kabsch algorithm).
    P and Q must be centred (zero mean). The determinant sign correction
    prevents improper rotations (reflections).
    """
    U, _, Vt = np.linalg.svd(P.T @ Q)
    d = np.linalg.det(Vt.T @ U.T)
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def superimpose_coords(
    mobile: np.ndarray, target: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Align *mobile* onto *target* via centroid shift + Kabsch rotation.

    Returns
    -------
    aligned_mobile : ndarray (N, 3)
    R              : rotation matrix (3, 3)
    t_mob          : centroid of mobile  (needed to apply the same transform
    t_tgt          : centroid of target   to atoms not used in the fit)
    """
    t_mob = mobile.mean(axis=0)
    t_tgt = target.mean(axis=0)
    R     = kabsch_rotation(mobile - t_mob, target - t_tgt)
    return (mobile - t_mob) @ R.T + t_tgt, R, t_mob, t_tgt


def apply_superimposition(
    coords: np.ndarray,
    R: np.ndarray,
    t_mob: np.ndarray,
    t_tgt: np.ndarray,
) -> np.ndarray:
    """Apply a rigid transform computed by superimpose_coords to a new set of coords."""
    return (coords - t_mob) @ R.T + t_tgt


# =============================================================================
# Ligand utilities  (RDKit)
# =============================================================================

def load_ligand(path: Path) -> Optional[Chem.Mol]:
    """Load a ligand from SDF or MOL2 with Hs removed. Returns None on failure."""
    try:
        p = str(path)
        if p.endswith(".sdf"):
            suppl = Chem.SDMolSupplier(p, removeHs=True, sanitize=True)
            mol = suppl[0] if suppl and len(suppl) > 0 else None
        elif p.endswith(".mol2"):
            mol = Chem.MolFromMol2File(p, removeHs=True, sanitize=True)
        else:
            mol = None
    except Exception:
        mol = None
    if mol is None:
        warnings.warn(f"Failed to load ligand: {path}")
    return mol


def get_mol_coords(mol: Chem.Mol) -> np.ndarray:
    """Return heavy-atom coordinates as (N, 3)."""
    return mol.GetConformer().GetPositions()


def compute_ligand_rmsd(mol_pred: Chem.Mol, mol_gt: Chem.Mol) -> float:
    """
    Symmetry-corrected RMSD via RDKit's GetBestRMS, which enumerates graph
    automorphisms to return the minimum over all valid atom mappings.
    Falls back to direct RMSD (no symmetry correction) if atom counts differ
    or the automorphism search fails.
    """
    try:
        return float(rdMolAlign.GetBestRMS(mol_pred, mol_gt))
    except Exception:
        cp, cg = get_mol_coords(mol_pred), get_mol_coords(mol_gt)
        if cp.shape != cg.shape:
            return float("nan")
        return float(np.sqrt(np.mean(np.sum((cp - cg) ** 2, axis=1))))


def compute_centroid_distance(mol_pred: Chem.Mol, mol_gt: Chem.Mol) -> float:
    """Euclidean distance between the geometric centroids of two ligand conformers."""
    return float(np.linalg.norm(
        get_mol_coords(mol_pred).mean(axis=0) - get_mol_coords(mol_gt).mean(axis=0)
    ))


# =============================================================================
# Protein sidechain RMSD
# =============================================================================

def compute_sidechain_rmsd(
    pred: Structure,
    gt: Structure,
    superimpose_backbone: bool = False,
) -> float:
    """
    RMSD of matched sidechain heavy atoms between predicted and GT proteins.

    Atoms are paired by (chain_id, resseq, icode, atom_name), so missing or
    extra residues in either structure are silently dropped.

    Parameters
    ----------
    superimpose_backbone : bool
        If True, first fit the predicted Cα atoms onto the GT Cα atoms via
        Kabsch, then apply that same rigid transform to the sidechain
        coordinates before computing RMSD. Use when the two structures are
        NOT pre-aligned to a common frame.
        If False, RMSD is computed directly in the global frame — appropriate
        for DiffDock-Pocket outputs that are already aligned to the GT frame.
    """
    pred_sc, gt_sc = match_sidechain_atoms(pred, gt)
    if pred_sc.shape[0] == 0:
        return float("nan")

    if superimpose_backbone:
        pred_ca, gt_ca = match_ca_atoms(pred, gt)
        if pred_ca.shape[0] >= 3:
            _, R, t_mob, t_tgt = superimpose_coords(pred_ca, gt_ca)
            pred_sc = apply_superimposition(pred_sc, R, t_mob, t_tgt)

    diff = pred_sc - gt_sc
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


# =============================================================================
# Steric clash detection
# =============================================================================

def compute_steric_clashes(
    mol_pred: Chem.Mol,
    prot_pred: Structure,
    overlap_factor: float = 0.75,
) -> float:
    """
    Fraction of ligand heavy atoms that clash with any protein heavy atom.

    Clash criterion:  dist(lig_i, prot_j) < (vdw_i + vdw_j) * overlap_factor

    Parameters
    ----------
    mol_pred : rdkit.Chem.Mol
        Predicted ligand pose (Hs already removed).
    prot_pred : Bio.PDB.Structure
        Predicted protein structure paired with this ligand pose.
    overlap_factor : float
        0.60 = strict  |  0.75 = standard  |  0.80 = permissive

    Returns
    -------
    float in [0, 1], or NaN if protein has no parseable heavy atoms.
    """
    lig_coords   = get_mol_coords(mol_pred)
    lig_elements = [mol_pred.GetAtomWithIdx(i).GetSymbol()
                    for i in range(mol_pred.GetNumAtoms())]

    prot_coords_list:   List[np.ndarray] = []
    prot_elements_list: List[str]        = []
    for atom, element in _iter_heavy_atoms(prot_pred, protein_only=False):
        prot_coords_list.append(atom.get_coord())
        prot_elements_list.append(element)

    if not prot_coords_list:
        return float("nan")

    prot_coords  = np.array(prot_coords_list)
    dist_matrix  = cdist(lig_coords, prot_coords)          # (n_lig, n_prot)

    lig_radii  = np.array([VDW_RADII.get(e, DEFAULT_VDW) for e in lig_elements ])[:, None]
    prot_radii = np.array([VDW_RADII.get(e, DEFAULT_VDW) for e in prot_elements_list])[None, :]

    clashing = np.any(dist_matrix < (lig_radii + prot_radii) * overlap_factor, axis=1)
    return float(clashing.sum() / max(len(clashing), 1))


# =============================================================================
# Directory / file discovery
# =============================================================================

_RANK_RE = re.compile(r"^rank(\d+)_confidence([-\d.]+)\.sdf$")


def find_rank_files(complex_dir: Path) -> Dict[int, Tuple[Path, Optional[Path]]]:
    """
    Discover ranked prediction files inside a complex directory.

    Only files with a confidence score in the name are included;
    bare rank1.sdf / rank1_protein.pdb (which are symlinks) are skipped.

    Returns
    -------
    dict: rank_int -> (sdf_path, protein_pdb_path | None)
    """
    results: Dict[int, Tuple[Path, Optional[Path]]] = {}
    for f in complex_dir.iterdir():
        m = _RANK_RE.match(f.name)
        if not m:
            continue
        rank = int(m.group(1))
        pdb  = complex_dir / f"{f.stem}_protein.pdb"
        results[rank] = (f, pdb if pdb.exists() else None)
    return results


def extract_pdb_id_and_ligand_ext(dir_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract PDB ID and ground-truth ligand extension from a complex directory name.

    Expected format:
      index{N}___{pdb_id}_protein_esmfold_aligned_tr_fix.pdb___{pdb_id}_ligand.mol2

    The third ___-separated part is the ligand filename and carries the correct
    extension (mol2 or sdf), so we read it directly rather than guessing.
    Falls back to a regex search for the PDB ID if the format doesn't match.

    Returns
    -------
    (pdb_id, ligand_ext)  e.g. ('6d5w', '.mol2')
    Either value may be None if it cannot be determined.
    """
    parts = dir_name.split("___")
    pdb_id: Optional[str] = None
    ligand_ext: Optional[str] = None

    if len(parts) >= 2:
        pdb_id = parts[1].split("_")[0].lower()

    if len(parts) >= 3:
        ligand_filename = parts[2]          # e.g. "6d5w_ligand.mol2"
        suffix = Path(ligand_filename).suffix.lower()   # e.g. ".mol2"
        if suffix in (".mol2", ".sdf"):
            ligand_ext = suffix

    if pdb_id is None:
        m = re.search(r"(?<![a-z0-9])([0-9][a-z0-9]{3})(?![a-z0-9])", dir_name.lower())
        pdb_id = m.group(1) if m else None

    return pdb_id, ligand_ext


def find_gt_ligand(gt_dir: Path, pdb_id: str, ligand_ext: Optional[str] = None) -> Optional[Path]:
    """
    Return the ground-truth ligand path.

    If *ligand_ext* is provided (read from the directory name), it is tried
    first. Falls back to trying both extensions so the function remains
    robust if the directory name is ever in an unexpected format.
    """
    preferred = [ligand_ext] if ligand_ext else []
    fallbacks = [ext for ext in (".mol2", ".sdf") if ext != ligand_ext]
    for ext in preferred + fallbacks:
        p = gt_dir / f"{pdb_id}_ligand{ext}"
        if p.exists():
            return p
    return None


def find_gt_protein(gt_dir: Path, pdb_id: str) -> Optional[Path]:
    """Return ground-truth protein PDB path, preferring ESMFold-aligned version."""
    for suffix in (
        "_protein_esmfold_aligned_tr_fix.pdb",
        "_protein_processed_fix.pdb",
        "_protein.pdb",
    ):
        p = gt_dir / f"{pdb_id}{suffix}"
        if p.exists():
            return p
    return None


# =============================================================================
# Per-complex evaluation
# =============================================================================

def evaluate_complex(
    complex_dir: Path,
    pdb_id: str,
    ligand_ext: Optional[str],
    data_dir: Path,
    top_ks: List[int],
    clash_overlap: float,
    superimpose_backbone: bool,
    verbose: bool,
) -> Optional[Dict]:
    """
    Evaluate all ranked predictions for one complex.

    Returns a flat dict of per-rank and top-K oracle metrics,
    or None if the complex cannot be evaluated (missing GT files, no predictions).
    """
    gt_dir = data_dir / pdb_id

    gt_lig_path  = find_gt_ligand(gt_dir, pdb_id, ligand_ext)
    gt_prot_path = find_gt_protein(gt_dir, pdb_id)

    if gt_lig_path is None:
        if verbose:
            print(f"  [SKIP] No GT ligand for {pdb_id}")
        return None

    gt_mol = load_ligand(gt_lig_path)
    if gt_mol is None:
        if verbose:
            print(f"  [SKIP] Could not load GT ligand for {pdb_id}")
        return None

    # Load GT protein once and share across all rank evaluations
    gt_struct: Optional[Structure] = None
    if gt_prot_path is not None:
        try:
            gt_struct = load_structure(gt_prot_path, structure_id=f"{pdb_id}_gt")
        except Exception as e:
            warnings.warn(f"{pdb_id}: failed to load GT protein — {e}")

    rank_files = find_rank_files(complex_dir)
    if not rank_files:
        if verbose:
            print(f"  [SKIP] No ranked predictions in {complex_dir.name}")
        return None

    # ── Per-rank metrics ──────────────────────────────────────────────────────
    per_rank: Dict[int, Dict[str, float]] = {}

    for rank in sorted(rank_files):
        sdf_path, pdb_path = rank_files[rank]

        pred_mol = load_ligand(sdf_path)
        if pred_mol is None:
            continue

        row: Dict[str, float] = {}

        # Ligand RMSD
        try:
            row["ligand_rmsd"] = compute_ligand_rmsd(pred_mol, gt_mol)
        except Exception as e:
            warnings.warn(f"{pdb_id} rank{rank} ligand RMSD: {e}")
            row["ligand_rmsd"] = float("nan")

        # Centroid distance
        try:
            row["centroid_dist"] = compute_centroid_distance(pred_mol, gt_mol)
        except Exception as e:
            warnings.warn(f"{pdb_id} rank{rank} centroid dist: {e}")
            row["centroid_dist"] = float("nan")

        # Load predicted protein structure (needed for sidechain RMSD + clashes)
        pred_struct: Optional[Structure] = None
        if pdb_path is not None:
            try:
                pred_struct = load_structure(pdb_path, structure_id=f"{pdb_id}_r{rank}")
            except Exception as e:
                warnings.warn(f"{pdb_id} rank{rank} protein load: {e}")

        # Protein sidechain RMSD
        if gt_struct is not None and pred_struct is not None:
            try:
                row["sidechain_rmsd"] = compute_sidechain_rmsd(
                    pred_struct, gt_struct, superimpose_backbone
                )
            except Exception as e:
                warnings.warn(f"{pdb_id} rank{rank} sidechain RMSD: {e}")
                row["sidechain_rmsd"] = float("nan")
        else:
            row["sidechain_rmsd"] = float("nan")

        # Steric clash fraction
        if pred_struct is not None:
            try:
                row["clash_frac"] = compute_steric_clashes(
                    pred_mol, pred_struct, overlap_factor=clash_overlap
                )
            except Exception as e:
                warnings.warn(f"{pdb_id} rank{rank} clashes: {e}")
                row["clash_frac"] = float("nan")
        else:
            row["clash_frac"] = float("nan")

        per_rank[rank] = row

    if not per_rank:
        return None

    # ── Assemble result dict ──────────────────────────────────────────────────
    result: Dict = {
        "pdb_id":     pdb_id,
        "complex_dir": complex_dir.name,
        "n_ranks":    len(per_rank),
    }

    # Raw rank-1 values (useful for inspection / debugging)
    if 1 in per_rank:
        for metric, val in per_rank[1].items():
            result[f"rank1_{metric}"] = val

    # Top-K oracle: best (minimum) value among the top-K confidence-ranked poses
    for k in top_ks:
        ranks_k = sorted(r for r in per_rank if r <= k)
        if not ranks_k:
            for m in ("ligand_rmsd", "centroid_dist", "sidechain_rmsd", "clash_frac"):
                result[f"top{k}_{m}"] = float("nan")
            continue

        for metric in ("ligand_rmsd", "centroid_dist", "sidechain_rmsd"):
            valid = [per_rank[r][metric] for r in ranks_k
                     if not np.isnan(per_rank[r][metric])]
            result[f"top{k}_{metric}"] = min(valid) if valid else float("nan")

        clash_valid = [per_rank[r]["clash_frac"] for r in ranks_k
                       if not np.isnan(per_rank[r]["clash_frac"])]
        result[f"top{k}_clash_frac"] = min(clash_valid) if clash_valid else float("nan")

    return result


# =============================================================================
# Aggregate statistics
# =============================================================================

def aggregate_metric(values: np.ndarray, prefix: str) -> Dict[str, float]:
    """Summary statistics for one scalar metric across all complexes."""
    valid = values[~np.isnan(values)]
    n = len(valid)
    if n == 0:
        return {
            f"{prefix}_mean":        float("nan"),
            f"{prefix}_frac_below_2": float("nan"),
            f"{prefix}_frac_below_5": float("nan"),
            f"{prefix}_p25":         float("nan"),
            f"{prefix}_p50":         float("nan"),
            f"{prefix}_p75":         float("nan"),
            f"{prefix}_n":           0,
        }
    return {
        f"{prefix}_mean":        float(np.mean(valid)),
        f"{prefix}_frac_below_2": float(np.mean(valid < 2.0)),
        f"{prefix}_frac_below_5": float(np.mean(valid < 5.0)),
        f"{prefix}_p25":         float(np.percentile(valid, 25)),
        f"{prefix}_p50":         float(np.percentile(valid, 50)),
        f"{prefix}_p75":         float(np.percentile(valid, 75)),
        f"{prefix}_n":           n,
    }


def compute_aggregate_metrics(rows: List[Dict], top_ks: List[int]) -> Dict:
    """Aggregate per-complex rows into dataset-level summary statistics."""
    df = pd.DataFrame(rows)
    summary: Dict = {"n_complexes": len(df)}

    for k in top_ks:
        for metric in ("ligand_rmsd", "centroid_dist", "sidechain_rmsd"):
            col = f"top{k}_{metric}"
            if col in df.columns:
                summary.update(
                    aggregate_metric(df[col].values.astype(float), col)
                )
        # Clash: mean fraction only (a below-2Å threshold is not meaningful here)
        clash_col = f"top{k}_clash_frac"
        if clash_col in df.columns:
            valid = df[clash_col].dropna().values.astype(float)
            summary[f"{clash_col}_mean"] = float(np.mean(valid)) if len(valid) else float("nan")
            summary[f"{clash_col}_n"]    = len(valid)

    return summary


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate DiffDock-Pocket docking predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--results_dir", type=Path, required=True, nargs="+",
        help=(
            "One or more DiffDock-Pocket result directories. "
            "Each should contain index*___*pdb*___*mol2 subdirectories."
        ),
    )
    p.add_argument(
        "--data_dir", type=Path, required=True,
        help="Root of PDBBIND_atomCorrected (or equivalent) ground-truth directory.",
    )
    p.add_argument(
        "--output", type=Path, default=Path("docking_metrics.csv"),
        help="Path to write per-complex CSV results.",
    )
    p.add_argument(
        "--summary_output", type=Path, default=Path("docking_summary.json"),
        help="Path to write aggregate summary JSON.",
    )
    p.add_argument(
        "--top_k", type=int, nargs="+", default=[1, 5, 10],
        help="Top-K oracle cutoffs to evaluate.",
    )
    p.add_argument(
        "--clash_overlap", type=float, default=0.75,
        help=(
            "VDW overlap factor for steric clash detection. "
            "Clash when dist < (r_i + r_j) * factor. "
            "Range: 0.60 (strict) to 0.80 (permissive)."
        ),
    )
    p.add_argument(
        "--superimpose_backbone", action="store_true", default=False,
        help=(
            "Before computing sidechain RMSD, superimpose predicted protein "
            "onto GT via Cα atoms (Kabsch). Use when structures are NOT "
            "pre-aligned to a common reference frame."
        ),
    )
    p.add_argument(
        "--pdb_id_list", type=Path, default=None,
        help="Optional file with one PDB ID per line; only those complexes are evaluated.",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true", default=False,
        help="Print per-complex progress and skip reasons.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    # ── Validate paths ────────────────────────────────────────────────────────
    for rd in args.results_dir:
        if not rd.exists():
            print(f"ERROR: results_dir not found: {rd}")
            sys.exit(1)
    if not args.data_dir.exists():
        print(f"ERROR: data_dir not found: {args.data_dir}")
        sys.exit(1)

    pdb_whitelist: Optional[set] = None
    if args.pdb_id_list is not None:
        pdb_whitelist = {
            line.strip().lower()
            for line in args.pdb_id_list.read_text().splitlines()
            if line.strip()
        }
        print(f"Whitelist: {len(pdb_whitelist)} PDB IDs.")

    # ── Discover complex directories ──────────────────────────────────────────
    complex_entries: List[Tuple[Path, str, Optional[str]]] = []  # (dir, pdb_id, ligand_ext)
    seen: set = set()

    for results_dir in args.results_dir:
        for subdir in sorted(results_dir.iterdir()):
            if not subdir.is_dir():
                continue
            pdb_id, ligand_ext = extract_pdb_id_and_ligand_ext(subdir.name)
            if pdb_id is None:
                if args.verbose:
                    print(f"  [SKIP] Cannot parse PDB ID: {subdir.name}")
                continue
            if pdb_whitelist is not None and pdb_id not in pdb_whitelist:
                continue
            if pdb_id in seen:
                warnings.warn(f"Duplicate PDB ID {pdb_id} — skipping second occurrence.")
                continue
            seen.add(pdb_id)
            complex_entries.append((subdir, pdb_id, ligand_ext))

    if not complex_entries:
        print("ERROR: No valid complex directories found.")
        sys.exit(1)

    print(f"Found {len(complex_entries)} complexes.")
    print(f"Top-K values:       {args.top_k}")
    print(f"Clash overlap:      {args.clash_overlap}")
    print(f"Superimpose Cα:     {args.superimpose_backbone}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    rows: List[Dict] = []
    n_skipped = 0

    for complex_dir, pdb_id, ligand_ext in tqdm(complex_entries, desc="Evaluating"):
        if args.verbose:
            print(f"\nEvaluating {pdb_id}  ({complex_dir.name})")
        result = evaluate_complex(
            complex_dir=complex_dir,
            pdb_id=pdb_id,
            ligand_ext=ligand_ext,
            data_dir=args.data_dir,
            top_ks=args.top_k,
            clash_overlap=args.clash_overlap,
            superimpose_backbone=args.superimpose_backbone,
            verbose=args.verbose,
        )
        if result is None:
            n_skipped += 1
        else:
            rows.append(result)

    print(f"\nEvaluated {len(rows)} complexes; skipped {n_skipped}.")
    if not rows:
        print("ERROR: No complexes were successfully evaluated.")
        sys.exit(1)

    # ── Save per-complex CSV ──────────────────────────────────────────────────
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Per-complex CSV  -> {args.output}")

    # ── Aggregate and display ─────────────────────────────────────────────────
    summary = compute_aggregate_metrics(rows, args.top_k)
    summary["n_skipped"] = n_skipped

    print("\n" + "=" * 72)
    print("AGGREGATE METRICS")
    print("=" * 72)

    for k in args.top_k:
        print(f"\n-- Top-{k} Oracle --------------------------------------------------")
        for label, key in [
            ("Ligand RMSD (A)",    f"top{k}_ligand_rmsd"),
            ("Centroid Dist (A)",  f"top{k}_centroid_dist"),
            ("Sidechain RMSD (A)", f"top{k}_sidechain_rmsd"),
        ]:
            n    = summary.get(f"{key}_n", 0)
            mean = summary.get(f"{key}_mean", float("nan"))
            fb2  = summary.get(f"{key}_frac_below_2", float("nan"))
            fb5  = summary.get(f"{key}_frac_below_5", float("nan"))
            p25  = summary.get(f"{key}_p25", float("nan"))
            p50  = summary.get(f"{key}_p50", float("nan"))
            p75  = summary.get(f"{key}_p75", float("nan"))
            print(
                f"  {label:22s}  n={n:4d}  mean={mean:6.3f}  "
                f"<2={fb2:.3f}  <5={fb5:.3f}  "
                f"p25={p25:.3f}  p50={p50:.3f}  p75={p75:.3f}"
            )
        clash_mean = summary.get(f"top{k}_clash_frac_mean", float("nan"))
        clash_n    = summary.get(f"top{k}_clash_frac_n", 0)
        print(f"  {'Clash fraction':22s}  n={clash_n:4d}  mean={clash_mean:.4f}")

    print("=" * 72)

    # ── Save summary JSON ─────────────────────────────────────────────────────
    clean: Dict = {}
    for key, val in summary.items():
        if isinstance(val, float) and np.isnan(val):
            clean[key] = None
        elif isinstance(val, (np.integer, np.floating)):
            clean[key] = val.item()
        else:
            clean[key] = val
    with open(args.summary_output, "w") as fh:
        json.dump(clean, fh, indent=2)
    print(f"Summary JSON     -> {args.summary_output}")


if __name__ == "__main__":
    main()