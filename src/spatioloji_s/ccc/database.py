"""
database.py - Ligand-receptor pair database for CCC analysis.

Loads CellChatDB human LR pairs and filters to genes expressed
in your spatioloji object. No external databases beyond CellChatDB
are required.

Signaling types
---------------
juxtacrine : membrane-bound ligand, requires direct cell contact
secreted   : diffusible ligand, paracrine / autocrine
ecm        : ECM-receptor interactions

Multi-subunit complexes
-----------------------
Both ligands and receptors can be multi-subunit complexes in CellChatDB.
These are stored as '|'-joined strings:
  "TGFB1"          → single ligand
  "CXCL5|PPBP"     → heterodimeric ligand complex
  "TGFBR1|TGFBR2"  → receptor complex

Downstream expression scoring uses mean across subunits.
Expression filtering requires ALL subunits to pass min_pct.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spatioloji_s.data.core import spatioloji

from dataclasses import dataclass, field
from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse import issparse


# ── LR pair dataclass ─────────────────────────────────────────────────────────

@dataclass
class LRPair:
    """
    Single ligand-receptor pair, supporting multi-subunit complexes
    on both ligand and receptor sides.

    Attributes
    ----------
    lr_name    : unique identifier, e.g. 'TGFB1_TGFBR1_TGFBR2'
    ligand     : ligand gene symbol, or '|'-joined subunits for complexes
                 e.g. 'CXCL5|PPBP'
    receptor   : receptor gene symbol, or '|'-joined subunits for complexes
                 e.g. 'TGFBR1|TGFBR2'
    pathway    : signaling pathway name from CellChatDB
    lr_type    : 'juxtacrine' | 'secreted' | 'ecm'
    annotation : original annotation string from CellChatDB
    """
    lr_name    : str
    ligand     : str
    receptor   : str
    pathway    : str
    lr_type    : str
    annotation : str = field(default="")

    # ── Ligand properties ─────────────────────────────────────────────────────

    @property
    def ligand_genes(self) -> list[str]:
        """Ligand subunits as a list (always a list, even for single gene)."""
        return self.ligand.split('|')

    @property
    def ligand_is_complex(self) -> bool:
        """True if ligand is a multi-subunit complex."""
        return '|' in self.ligand

    # ── Receptor properties ───────────────────────────────────────────────────

    @property
    def receptor_genes(self) -> list[str]:
        """Receptor subunits as a list (always a list, even for single gene)."""
        return self.receptor.split('|')

    @property
    def receptor_is_complex(self) -> bool:
        """True if receptor is a multi-subunit complex."""
        return '|' in self.receptor

    # ── Combined properties ───────────────────────────────────────────────────

    @property
    def is_complex(self) -> bool:
        """True if either ligand OR receptor is a multi-subunit complex."""
        return self.ligand_is_complex or self.receptor_is_complex

    @property
    def all_genes(self) -> list[str]:
        """All genes involved: all ligand subunits + all receptor subunits."""
        return self.ligand_genes + self.receptor_genes

    @property
    def n_subunits(self) -> tuple[int, int]:
        """(n_ligand_subunits, n_receptor_subunits)."""
        return len(self.ligand_genes), len(self.receptor_genes)

    def __repr__(self) -> str:
        l_str = (self.ligand if not self.ligand_is_complex
                 else f"complex({self.ligand})")
        r_str = (self.receptor if not self.receptor_is_complex
                 else f"complex({self.receptor})")
        return (
            f"LRPair({self.lr_name!r}, "
            f"ligand={l_str!r}, receptor={r_str!r}, "
            f"type={self.lr_type!r}, pathway={self.pathway!r})"
        )


# ── Annotation → lr_type mapping ─────────────────────────────────────────────

_ANNOTATION_TO_TYPE: dict[str, str] = {
    "Secreted Signaling" : "secreted",
    "Cell-Cell Contact"  : "juxtacrine",
    "ECM-Receptor"       : "ecm",
}

# ── Built-in curated subset ───────────────────────────────────────────────────
# (lr_name, ligand, receptor, pathway, lr_type)
# Multi-subunit: use '|' separator

_BUILTIN_RECORDS: list[tuple[str, str, str, str, str]] = [

    # Checkpoint / immune suppression — juxtacrine
    ("CD274-PDCD1",      "CD274",        "PDCD1",           "PD-L1 signaling",    "juxtacrine"),
    ("CD80-CTLA4",       "CD80",         "CTLA4",           "CD80 signaling",     "juxtacrine"),
    ("CD86-CTLA4",       "CD86",         "CTLA4",           "CD86 signaling",     "juxtacrine"),
    ("TIGIT-PVR",        "TIGIT",        "PVR",             "TIGIT signaling",    "juxtacrine"),
    ("LGALS9-HAVCR2",    "LGALS9",       "HAVCR2",          "Galectin signaling", "juxtacrine"),
    ("CD276-TMIGD2",     "CD276",        "TMIGD2",          "B7 signaling",       "juxtacrine"),

    # Notch — juxtacrine
    ("DLL1-NOTCH1",      "DLL1",         "NOTCH1",          "Notch signaling",    "juxtacrine"),
    ("DLL4-NOTCH1",      "DLL4",         "NOTCH1",          "Notch signaling",    "juxtacrine"),
    ("JAG1-NOTCH1",      "JAG1",         "NOTCH1",          "Notch signaling",    "juxtacrine"),

    # Ephrin — juxtacrine
    ("EFNA1-EPHA2",      "EFNA1",        "EPHA2",           "Ephrin signaling",   "juxtacrine"),
    ("EFNB1-EPHB2",      "EFNB1",        "EPHB2",           "Ephrin signaling",   "juxtacrine"),

    # MHC — juxtacrine
    ("HLA-A-CD8A",       "HLA-A",        "CD8A",            "MHC-I signaling",    "juxtacrine"),
    ("HLA-DRA-CD4",      "HLA-DRA",      "CD4",             "MHC-II signaling",   "juxtacrine"),

    # Chemokines — secreted
    ("CXCL12-CXCR4",     "CXCL12",       "CXCR4",           "CXCL12 signaling",   "secreted"),
    ("CXCL10-CXCR3",     "CXCL10",       "CXCR3",           "CXCL10 signaling",   "secreted"),
    ("CCL2-CCR2",         "CCL2",         "CCR2",            "CCL2 signaling",     "secreted"),
    ("CCL5-CCR5",         "CCL5",         "CCR5",            "CCL5 signaling",     "secreted"),
    ("CXCL8-CXCR1",      "CXCL8",        "CXCR1",           "IL8 signaling",      "secreted"),
    ("CXCL1-CXCR2",      "CXCL1",        "CXCR2",           "CXCL1 signaling",    "secreted"),

    # Heterodimeric ligand complexes — secreted
    ("CXCL5_PPBP-CXCR2", "CXCL5|PPBP",  "CXCR2",           "CXCL5 signaling",    "secreted"),
    ("IFNA2_IFNB1-IFNAR", "IFNA2|IFNB1", "IFNAR1|IFNAR2",   "IFN signaling",      "secreted"),

    # Growth factors — secreted
    ("TGFB1-TGFBR1R2",   "TGFB1",        "TGFBR1|TGFBR2",  "TGF-beta signaling", "secreted"),
    ("VEGFA-KDR",         "VEGFA",        "KDR",             "VEGF signaling",     "secreted"),
    ("EGF-EGFR",          "EGF",          "EGFR",            "EGF signaling",      "secreted"),
    ("HGF-MET",           "HGF",          "MET",             "HGF signaling",      "secreted"),
    ("IL6-IL6R",          "IL6",          "IL6R",            "IL6 signaling",      "secreted"),
    ("IL10-IL10RA",       "IL10",         "IL10RA|IL10RB",   "IL10 signaling",     "secreted"),
    ("TNF-TNFRSF1A",      "TNF",          "TNFRSF1A",        "TNF signaling",      "secreted"),
    ("IFNG-IFNGR",        "IFNG",         "IFNGR1|IFNGR2",   "IFN-gamma signaling","secreted"),

    # Wnt — secreted
    ("WNT5A-FZD1",        "WNT5A",        "FZD1",            "Wnt signaling",      "secreted"),
    ("WNT2-FZD4",         "WNT2",         "FZD4",            "Wnt signaling",      "secreted"),

    # Angiopoietin — secreted
    ("ANGPT1-TEK",        "ANGPT1",       "TEK",             "Angiopoietin",       "secreted"),
    ("ANGPT2-TEK",        "ANGPT2",       "TEK",             "Angiopoietin",       "secreted"),

    # ECM — ecm
    ("FN1-ITGB1",         "FN1",          "ITGB1",           "ECM-Integrin",       "ecm"),
    ("COL1A1-ITGB1",      "COL1A1",       "ITGB1",           "ECM-Integrin",       "ecm"),
    ("LAMB1-ITGB1",       "LAMB1",        "ITGB1",           "Laminin signaling",  "ecm"),
    ("THBS1-CD36",        "THBS1",        "CD36",            "TSP signaling",      "ecm"),
    ("SPP1-CD44",         "SPP1",         "CD44",            "OPN signaling",      "ecm"),
    ("MMP9-CD44",         "MMP9",         "CD44",            "MMP signaling",      "ecm"),
]


# ── Public functions ──────────────────────────────────────────────────────────

def load_lr_database(
    source: str = 'builtin',
    lr_types: list[str] | None = None,
    custom_df: pd.DataFrame | None = None,
) -> list[LRPair]:
    """
    Load LR pair database from built-in list or a custom DataFrame.

    Parameters
    ----------
    source : str
        'builtin' uses the curated built-in list.
        'custom' requires custom_df with columns:
        lr_name, ligand, receptor, pathway, lr_type.
        Both ligand and receptor may use '|' notation for complexes.
    lr_types : list[str] | None
        Filter to specific types. Options: 'juxtacrine', 'secreted', 'ecm'.
        None = all types.
    custom_df : pd.DataFrame | None
        Required when source='custom'.

    Returns
    -------
    list[LRPair]
        Unfiltered by expression — call filter_to_expressed() next.

    Examples
    --------
    >>> pairs = load_lr_database()
    >>> pairs = load_lr_database(lr_types=['juxtacrine'])
    """
    if source == 'builtin':
        pairs = [
            LRPair(
                lr_name  = rec[0],
                ligand   = rec[1],
                receptor = rec[2],
                pathway  = rec[3],
                lr_type  = rec[4],
            )
            for rec in _BUILTIN_RECORDS
        ]

    elif source == 'custom':
        if custom_df is None:
            raise ValueError("custom_df is required when source='custom'")
        required = {'lr_name', 'ligand', 'receptor', 'pathway', 'lr_type'}
        missing  = required - set(custom_df.columns)
        if missing:
            raise ValueError(f"custom_df missing columns: {missing}")
        pairs = [
            LRPair(
                lr_name    = str(row['lr_name']),
                ligand     = str(row['ligand']),
                receptor   = str(row['receptor']),
                pathway    = str(row['pathway']),
                lr_type    = str(row['lr_type']),
                annotation = str(row.get('annotation', '')),
            )
            for _, row in custom_df.iterrows()
        ]

    else:
        raise ValueError(
            f"Unknown source: {source!r}. Use 'builtin' or 'custom'."
        )

    pairs = _filter_by_type(pairs, lr_types)
    _print_summary(pairs, skipped=0, source_label=source)
    return pairs


def load_from_cellchatdb_csv(
    csv_path: str,
    lr_types: list[str] | None = None,
) -> list[LRPair]:
    """
    Load LR pairs from a CellChatDB CSV file.

    Both ligand.symbol and receptor.symbol are parsed for
    multi-subunit complexes (comma-separated values).

    Parameters
    ----------
    csv_path : str
        Path to CellChatDB CSV file.
    lr_types : list[str] | None
        Filter to specific types after loading. None = all.

    Returns
    -------
    list[LRPair]
        Unfiltered by expression — call filter_to_expressed() next.

    Examples
    --------
    >>> pairs = load_from_cellchatdb_csv("CellChatDB.csv")
    >>> pairs = load_from_cellchatdb_csv("CellChatDB.csv",
    ...                                   lr_types=['juxtacrine', 'secreted'])
    """
    df = pd.read_csv(csv_path, index_col=0)

    required = {
        'interaction_name', 'pathway_name',
        'ligand.symbol', 'receptor.symbol', 'annotation',
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CellChatDB CSV missing required columns: {missing}\n"
            f"Available: {list(df.columns)}"
        )

    pairs:   list[LRPair] = []
    skipped: int          = 0

    for _, row in df.iterrows():

        # Map annotation → lr_type
        annotation = str(row.get('annotation', '')).strip()
        lr_type    = _ANNOTATION_TO_TYPE.get(annotation)
        if lr_type is None:
            skipped += 1
            continue

        # Parse ligand (may be complex)
        ligand = _parse_gene_field(str(row['ligand.symbol']))
        if ligand is None:
            skipped += 1
            continue

        # Parse receptor (may be complex)
        receptor = _parse_gene_field(str(row['receptor.symbol']))
        if receptor is None:
            skipped += 1
            continue

        pairs.append(LRPair(
            lr_name    = str(row['interaction_name']).strip(),
            ligand     = ligand,
            receptor   = receptor,
            pathway    = str(row['pathway_name']).strip(),
            lr_type    = lr_type,
            annotation = annotation,
        ))

    pairs = _filter_by_type(pairs, lr_types)
    _print_summary(pairs, skipped, csv_path)
    return pairs


def filter_to_expressed(
    lr_pairs: list[LRPair],
    sp: 'spatioloji',
    min_pct: float = 0.05,
    layer: str | None = 'log_normalized',
) -> list[LRPair]:
    """
    Filter LR pairs to those where ALL ligand subunits AND ALL
    receptor subunits are expressed in at least min_pct of cells.

    For single genes: standard pct_expressed check.
    For complexes:    every subunit must independently pass min_pct.
                      This enforces co-expression required for a
                      functional complex.

    Parameters
    ----------
    lr_pairs : list[LRPair]
        Output of load_lr_database() or load_from_cellchatdb_csv().
    sp : spatioloji
        spatioloji object (typically already subset to one FOV).
    min_pct : float
        Minimum fraction of cells with expression > 0. Default 0.05.
    layer : str | None
        Which layer to use. None → use sp.expression (raw counts).

    Returns
    -------
    list[LRPair]

    Examples
    --------
    >>> expressed = filter_to_expressed(pairs, sp_fov, min_pct=0.05)
    >>> print(f"Expressed: {len(expressed)} / {len(pairs)}")
    """
    # Get expression matrix
    if layer is not None and layer in sp.layers:
        expr = sp.layers[layer]
    else:
        expr = sp.expression.to_dense()

    if issparse(expr):
        expr = expr.toarray()

    gene_names = np.array(sp.gene_index)
    n_cells    = expr.shape[0]
    gene_set   = set(gene_names)
    gene2idx   = {g: i for i, g in enumerate(gene_names)}

    # Cache pct_expressed per gene — computed once per unique gene
    pct_cache: dict[str, float] = {}

    def _pct(gene: str) -> float:
        if gene not in pct_cache:
            if gene not in gene2idx:
                pct_cache[gene] = 0.0
            else:
                col = expr[:, gene2idx[gene]]
                pct_cache[gene] = float((col > 0).sum()) / n_cells
        return pct_cache[gene]

    def _all_pass(genes: list[str]) -> tuple[bool, str]:
        """
        Check all subunits pass min_pct.
        Returns (passed, reason) where reason is '' if passed.
        """
        for g in genes:
            if g not in gene_set:
                return False, 'not_in_panel'
            if _pct(g) < min_pct:
                return False, 'low_pct'
        return True, ''

    kept:           list[LRPair] = []
    n_not_in_panel: int          = 0
    n_low_pct:      int          = 0

    for pair in lr_pairs:

        # Check all ligand subunits
        passed, reason = _all_pass(pair.ligand_genes)
        if not passed:
            if reason == 'not_in_panel':
                n_not_in_panel += 1
            else:
                n_low_pct += 1
            continue

        # Check all receptor subunits
        passed, reason = _all_pass(pair.receptor_genes)
        if not passed:
            if reason == 'not_in_panel':
                n_not_in_panel += 1
            else:
                n_low_pct += 1
            continue

        kept.append(pair)

    print(f"[CCC filter] {len(kept)} / {len(lr_pairs)} LR pairs pass "
          f"(min_pct={min_pct:.0%})")
    print(f"  Removed: {n_not_in_panel} not in gene panel, "
          f"{n_low_pct} below min_pct")

    return kept


def lr_pairs_to_dataframe(lr_pairs: list[LRPair]) -> pd.DataFrame:
    """
    Convert a list of LRPair objects to a summary DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: lr_name, ligand, receptor, pathway, lr_type,
                 annotation, ligand_is_complex, receptor_is_complex,
                 n_ligand_subunits, n_receptor_subunits
    """
    return pd.DataFrame([
        {
            'lr_name'             : p.lr_name,
            'ligand'              : p.ligand,
            'receptor'            : p.receptor,
            'pathway'             : p.pathway,
            'lr_type'             : p.lr_type,
            'annotation'          : p.annotation,
            'ligand_is_complex'   : p.ligand_is_complex,
            'receptor_is_complex' : p.receptor_is_complex,
            'n_ligand_subunits'   : len(p.ligand_genes),
            'n_receptor_subunits' : len(p.receptor_genes),
        }
        for p in lr_pairs
    ])


# ── Private helpers ───────────────────────────────────────────────────────────

def _parse_gene_field(raw: str) -> str | None:
    """
    Parse a gene field from CellChatDB (ligand.symbol or receptor.symbol).

    Handles:
      Single gene:  "TGFB1"         → "TGFB1"
      Complex:      "TGFBR2, TGFBR1"→ "TGFBR2|TGFBR1"
      Semicolons:   "TGFBR2; TGFBR1"→ "TGFBR2|TGFBR1"
      Missing:      "nan" or ""     → None

    Returns None if field is empty or unparseable.
    """
    raw = raw.strip()
    if not raw or raw == 'nan':
        return None

    genes = [
        g.strip()
        for g in raw.replace(';', ',').split(',')
        if g.strip() and g.strip() != 'nan'
    ]
    if not genes:
        return None

    return '|'.join(genes)


def _filter_by_type(
    pairs: list[LRPair],
    lr_types: list[str] | None,
) -> list[LRPair]:
    """Filter LRPair list by lr_type. No-op if lr_types is None."""
    if lr_types is None:
        return pairs
    valid = {'juxtacrine', 'secreted', 'ecm'}
    bad   = set(lr_types) - valid
    if bad:
        raise ValueError(f"Unknown lr_types: {bad}. Valid: {valid}")
    return [p for p in pairs if p.lr_type in lr_types]


def _print_summary(
    pairs: list[LRPair],
    skipped: int,
    source_label: str,
) -> None:
    """Print loading summary with type and complex breakdown."""
    print(f"[CCC database] Loaded {len(pairs)} LR pairs "
          f"from {source_label}")
    if skipped:
        print(f"  Skipped {skipped} rows "
              f"(unknown annotation or missing genes)")
    if pairs:
        for t, n in sorted(Counter(p.lr_type for p in pairs).items()):
            n_ligand_cx  = sum(1 for p in pairs
                               if p.lr_type == t and p.ligand_is_complex)
            n_receptor_cx = sum(1 for p in pairs
                                if p.lr_type == t and p.receptor_is_complex)
            print(f"  {t:12s}: {n:4d}  "
                  f"(ligand complexes: {n_ligand_cx}, "
                  f"receptor complexes: {n_receptor_cx})")
