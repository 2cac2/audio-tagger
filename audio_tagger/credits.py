"""Credit normalization for multi-artist Indian film albums.

Two jobs, both in service of *faithful* accounting of who did what:

1. ``canonical_acts`` folds the individual members of well-known Bollywood
   composer duos/trios back onto the act they are actually credited as — a
   soundtrack scored by "Vishal Dadlani" and "Shekhar Ravjiani" is credited to
   the act **Vishal-Shekhar**, not two separate composers. It is deliberately
   conservative: only a full-name member match (or the act's own name) folds,
   so a lone "Vishal Mishra" or "Vishal Bhardwaj" is never mistaken for
   Vishal-Shekhar.

2. ``merge_credits`` unions the role-tagged credits from several sources into a
   single deduplicated list, tolerant of transliteration spelling drift, while
   preserving the order in which singers were first seen (playback-singer order
   is meaningful).

Standard library only; transliteration comparison is delegated to ``translit``.
"""

from __future__ import annotations

from .models import ArtistCredit, Candidate
from .translit import normalize_roman_hindi, sim

# ---------------------------------------------------------------------------
# Composer-act alias table
# ---------------------------------------------------------------------------

# (canonical display name, [full member names]). The canonical name is the
# conventional hyphenated act name; matching is done on translit-normalized
# full member names and on the act name itself (spaced / unspaced), never on a
# bare first name — that guard keeps unrelated "Vishal ..."s out of the act.
_ACT_TABLE: list[tuple[str, list[str]]] = [
    ("Vishal-Shekhar", ["Vishal Dadlani", "Shekhar Ravjiani"]),
    ("Shankar-Ehsaan-Loy", ["Shankar Mahadevan", "Ehsaan Noorani", "Loy Mendonsa"]),
    ("Sachin-Jigar", ["Sachin Sanghvi", "Jigar Saraiya"]),
    ("Salim-Sulaiman", ["Salim Merchant", "Sulaiman Merchant"]),
    ("Ajay-Atul", ["Ajay Gogavale", "Atul Gogavale"]),
    ("Laxmikant-Pyarelal", ["Laxmikant Shantaram Kudalkar", "Pyarelal Ramprasad Sharma"]),
    ("Kalyanji-Anandji", ["Kalyanji Virji Shah", "Anandji Virji Shah"]),
    ("Jatin-Lalit", ["Jatin Pandit", "Lalit Pandit"]),
    ("Nadeem-Shravan", ["Nadeem Saifi", "Shravan Rathod"]),
    ("Anand-Milind", ["Anand Chitragupta", "Milind Chitragupta"]),
    ("Sajid-Wajid", ["Sajid Khan", "Wajid Khan"]),
]


def _build_index() -> list[tuple[str, set[str], set[str]]]:
    """Precompute (display, member-norm-set, act-alias-set) for each act."""
    index: list[tuple[str, set[str], set[str]]] = []
    for display, members in _ACT_TABLE:
        member_norms = {normalize_roman_hindi(m) for m in members}
        act_norm = normalize_roman_hindi(display)             # e.g. "visal sekar"
        aliases = {act_norm, act_norm.replace(" ", "")}       # spaced + unspaced
        index.append((display, member_norms, aliases))
    return index


_ACT_INDEX = _build_index()


def _resolve_act(name: str) -> str:
    """Map a single credited name onto its canonical act, or return it as-is."""
    n = normalize_roman_hindi(name)
    if not n:
        return name.strip()
    for display, member_norms, aliases in _ACT_INDEX:
        if n in member_norms or n in aliases or n.replace(" ", "") in aliases:
            return display
    return name.strip()


def canonical_acts(names: list[str]) -> list[str]:
    """Fold composer-act members to their credited act; dedupe; keep order.

    ``["Vishal Dadlani", "Shekhar Ravjiani"]`` -> ``["Vishal-Shekhar"]``.
    First-seen order is preserved, and duplicates (including two members that
    fold to the same act) collapse to a single entry.
    """
    out: list[str] = []
    seen: set[str] = set()
    for name in names or []:
        if not name or not name.strip():
            continue
        act = _resolve_act(name)
        key = normalize_roman_hindi(act)
        if key in seen:
            continue
        seen.add(key)
        out.append(act)
    return out


# ---------------------------------------------------------------------------
# Cross-source credit merge
# ---------------------------------------------------------------------------

def merge_credits(candidates: list[Candidate]) -> list[ArtistCredit]:
    """Union role-tagged credits across candidates into one deduped list.

    Two credits collapse when they share a role and their names are
    transliteration-similar (``translit.sim >= 0.9``). The first spelling seen
    wins, so singer order from the earliest/best source is preserved; a later
    duplicate only contributes its MBID if the first had none.
    """
    merged: list[ArtistCredit] = []
    for cand in candidates or []:
        for cr in getattr(cand, "credits", []) or []:
            name = (cr.name or "").strip()
            if not name:
                continue
            dup = None
            for existing in merged:
                if existing.role == cr.role and sim(existing.name, name) >= 0.9:
                    dup = existing
                    break
            if dup is None:
                merged.append(ArtistCredit(name=name, role=cr.role, mbid=cr.mbid))
            elif dup.mbid is None and cr.mbid:
                dup.mbid = cr.mbid
    return merged
