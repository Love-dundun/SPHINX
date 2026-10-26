"""Canonical protein-association prior for SPHINX.

This module is the **single source of truth** for the documented protein
co-expression and mutual-exclusion relationships used anywhere in the model:

* ``ProteinGAT`` (``sphinx/model.py``) turns them into additive attention
  biases on the 18 x 18 protein graph.
* ``bio_prior_loss`` (``sphinx/losses.py``) applies the same relationships as a
  regularisation term on the predicted protein channels.

The pairs are defined once here and imported by both consumers, so the loss and
the graph attention always operate on the same relationships.

Provenance
----------
The pairs were identified from pairwise protein co-expression analysis of the
training cohort and curated against published marker biology; the curated list
below is the one reported in the manuscript (18 co-expression + 26
mutual-exclusion pairs).  Pair order follows the canonical 18-protein ordering
used by ``PROTEIN_COLS`` in ``sphinx/dataset.py``.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

# ── Canonical protein ordering (must match PROTEIN_COLS in dataset.py) ──
PROTEIN_ORDER: List[str] = [
    'ATPase', 'CD11c', 'CD20', 'CD3', 'CD31', 'CD4',
    'CD45RA', 'CD56', 'CD68', 'CD8', 'CTLA4', 'FOXP3',
    'HLA-DR', 'KI67', 'PD1', 'PDL1', 'PanCK', 'aSMA',
]

# ── Documented co-expression pairs (positive prior) ──
POSITIVE_PAIRS: List[Tuple[str, str]] = [
    ('CD3', 'CD4'), ('CD3', 'CD8'), ('CD3', 'FOXP3'), ('CD3', 'PD1'),
    ('CD4', 'FOXP3'), ('CD4', 'CTLA4'), ('FOXP3', 'CTLA4'), ('FOXP3', 'PD1'),
    ('CD8', 'PD1'), ('CD4', 'PD1'),
    ('CD20', 'HLA-DR'), ('CD11c', 'HLA-DR'), ('CD68', 'HLA-DR'), ('CD68', 'CD11c'),
    ('PanCK', 'KI67'), ('PanCK', 'PDL1'),
    ('CD31', 'aSMA'),
    ('PD1', 'CTLA4'),
]

# ── Documented mutual-exclusion pairs (negative prior) ──
NEGATIVE_PAIRS: List[Tuple[str, str]] = [
    ('CD4', 'CD8'),
    ('CD3', 'CD20'), ('CD4', 'CD20'), ('CD8', 'CD20'), ('FOXP3', 'CD20'),
    ('CD3', 'CD68'), ('CD3', 'CD11c'), ('CD4', 'CD68'), ('CD8', 'CD68'),
    ('CD20', 'CD68'), ('CD20', 'CD11c'),
    ('PanCK', 'CD3'), ('PanCK', 'CD4'), ('PanCK', 'CD8'),
    ('PanCK', 'CD20'), ('PanCK', 'CD68'), ('PanCK', 'CD11c'),
    ('PanCK', 'CD56'), ('PanCK', 'FOXP3'), ('PanCK', 'CTLA4'),
    ('aSMA', 'PanCK'), ('aSMA', 'CD3'), ('aSMA', 'CD20'),
    ('CD56', 'CD20'), ('CD56', 'CD68'),
    ('FOXP3', 'CD8'),
]

# ── Pair set used by the biology-prior loss (io_prior_loss) ──
# This is deliberately NOT identical to POSITIVE_PAIRS / NEGATIVE_PAIRS above.
# The two sets were produced at different stages of the project: the set below is
# the earlier data-driven extraction, the set above is the version curated
# against the literature.  They share 30 pairs; 14 pairs are only in the
# attention set and 8 only in the loss set.  The models
# reported in the manuscript were trained with these pairs, so they are kept
# as-is; unifying them requires retraining and is tracked as future work.
LOSS_POSITIVE_PAIRS: List[Tuple[str, str]] = [
    ('CD3', 'CD4'), ('CD3', 'CD8'), ('CD3', 'CD45RA'), ('CD3', 'FOXP3'),
    ('CD4', 'FOXP3'), ('PD1', 'CTLA4'),
    ('CD20', 'HLA-DR'), ('CD11c', 'HLA-DR'), ('CD68', 'CD11c'), ('CD68', 'HLA-DR'),
    ('PanCK', 'PDL1'), ('CD31', 'ATPase'), ('aSMA', 'CD31'),
    ('CD3', 'PD1'), ('CD8', 'PD1'),
]

LOSS_NEGATIVE_PAIRS: List[Tuple[str, str]] = [
    ('PanCK', 'CD3'), ('PanCK', 'CD20'), ('PanCK', 'CD68'), ('PanCK', 'CD11c'),
    ('PanCK', 'CD4'), ('PanCK', 'CD8'), ('PanCK', 'FOXP3'), ('PanCK', 'PD1'),
    ('aSMA', 'CD3'), ('aSMA', 'CD20'), ('aSMA', 'CD68'),
    ('CD31', 'CD3'), ('CD31', 'CD20'),
    ('CD3', 'CD20'), ('CD8', 'CD20'), ('CD4', 'CD20'), ('CD68', 'CD20'),
    ('CD68', 'CD3'), ('CD11c', 'CD20'), ('CD11c', 'CD3'),
    ('CD56', 'CD3'), ('CD56', 'CD4'),
    ('CD4', 'CD8'),
]

# ── Three-level prior values added to the attention logits ──
BIAS_CO_EXPRESSION: float = 0.0     # documented co-expression
BIAS_MUTUAL_EXCLUSION: float = -5.0  # documented mutual exclusion
BIAS_UNSPECIFIED: float = -10.0      # no documented relationship

NAME_TO_INDEX: Dict[str, int] = {n: i for i, n in enumerate(PROTEIN_ORDER)}


def to_index_pairs(pairs: Iterable[Sequence[str]]) -> List[Tuple[int, int]]:
    """Map named pairs to index pairs, validating every name."""
    out: List[Tuple[int, int]] = []
    for a, b in pairs:
        if a not in NAME_TO_INDEX:
            raise KeyError(f"unknown protein {a!r}; expected one of {PROTEIN_ORDER}")
        if b not in NAME_TO_INDEX:
            raise KeyError(f"unknown protein {b!r}; expected one of {PROTEIN_ORDER}")
        i, j = NAME_TO_INDEX[a], NAME_TO_INDEX[b]
        if i == j:
            raise ValueError(f"self-pair {a}-{b} is not a valid prior entry")
        out.append((i, j))
    return out


POSITIVE_INDEX_PAIRS: List[Tuple[int, int]] = to_index_pairs(POSITIVE_PAIRS)
NEGATIVE_INDEX_PAIRS: List[Tuple[int, int]] = to_index_pairs(NEGATIVE_PAIRS)
LOSS_POSITIVE_INDEX_PAIRS: List[Tuple[int, int]] = to_index_pairs(LOSS_POSITIVE_PAIRS)
LOSS_NEGATIVE_INDEX_PAIRS: List[Tuple[int, int]] = to_index_pairs(LOSS_NEGATIVE_PAIRS)


def build_bias_matrix(num_proteins: int | None = None) -> List[List[float]]:
    """Return the additive attention bias implied by the pairs.

    Args:
        num_proteins: optional prefix length; defaults to all 18 proteins.
    """
    n = len(PROTEIN_ORDER) if num_proteins is None else int(num_proteins)
    if not 1 <= n <= len(PROTEIN_ORDER):
        raise ValueError(f"num_proteins must be in 1..{len(PROTEIN_ORDER)}")
    bias = [[BIAS_UNSPECIFIED] * n for _ in range(n)]
    for i, j in POSITIVE_INDEX_PAIRS:
        bias[i][j] = bias[j][i] = BIAS_CO_EXPRESSION
    for i, j in NEGATIVE_INDEX_PAIRS:
        bias[i][j] = bias[j][i] = BIAS_MUTUAL_EXCLUSION
    for i in range(n):
        bias[i][i] = BIAS_CO_EXPRESSION
    return bias


def summary() -> str:
    return (f"{len(PROTEIN_ORDER)} proteins; attention prior: "
            f"{len(POSITIVE_PAIRS)} co-expression + {len(NEGATIVE_PAIRS)} mutual-exclusion pairs; "
            f"prior-loss set: {len(LOSS_POSITIVE_PAIRS)} + {len(LOSS_NEGATIVE_PAIRS)} pairs")


if __name__ == "__main__":
    print(summary())
    bias = build_bias_matrix()
    for row in bias:
        print(" ".join(f"{v:5.1f}" for v in row))
