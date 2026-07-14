"""Album-level reconciliation.

Music players group tracks into an album by (album name + album-artist),
and some also by album id. If album-artist is chosen per-track, a film with
several singers/composers explodes into many one-track "albums". This pass
runs AFTER per-track resolution: it groups tracks that belong to the same
release and rewrites the album-level fields to be *identical* across the
group, so the film stays as one album.

Rules applied per group (faithful multi-composer accounting):
  * ALBUM        : one normalized string for every track.
  * ALBUMARTIST  : composer names are folded to their credited acts
                   (``credits.canonical_acts``), then
                     - 1 act  -> that act
                     - 2..3 acts -> ALL of them as a multi-value list, in
                       first-seen order (NEVER "Various Artists" for a film
                       whose composers are known)
                     - >3 or none known -> fall back (sole singer, else
                       "Various Artists").
  * comp (TCMP)  : 1 ONLY when the release itself is a compilation
                   (release_type == COMPILATION or looks_like_compilation).
                   Varying playback singers across a film is normal and MUST
                   NOT set comp — album cohesion comes from identical ALBUM +
                   ALBUMARTIST + album_key, not the compilation flag.
  * MUSICBRAINZ_ALBUMID / album key: shared, for id-based grouping.
Per-track fields (ARTIST=singers, COMPOSER, LYRICIST, TITLE) are untouched.
"""

from __future__ import annotations

from collections import defaultdict

from .credits import canonical_acts
from .heuristics import looks_like_compilation
from .models import Candidate, ReleaseType, Resolution


def _album_key(c: Candidate) -> str:
    """Stable grouping key: prefer a real release id, else film, else album text."""
    if c.release_mbid:
        return f"mbid:{c.release_mbid}"
    if c.film:
        return f"film:{c.film.strip().lower()}"
    return f"album:{(c.release_title or c.title).strip().lower()}"


def _canonical_album_name(members: list[Resolution]) -> str:
    # Use the longest existing 'album' tag (usually the fully-suffixed OST name).
    names = [m.tags.get("album", "") for m in members if m.tags.get("album")]
    return max(names, key=len) if names else ""


def _dedupe_ordered(names: list[str]) -> list[str]:
    """Case-insensitively dedupe while preserving first-seen order."""
    out: list[str] = []
    seen: set[str] = set()
    for name in names:
        if not name or not name.strip():
            continue
        key = name.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(name.strip())
    return out


def _album_artist_for_group(members: list[Resolution]) -> tuple[list[str], list[str]]:
    """Decide the shared ALBUMARTIST for a group.

    Returns ``(album_artist, composer_acts)`` where ``composer_acts`` is the
    full canonicalized composer-act list (kept for the reasons accounting).

    Composers are gathered in first-seen order across the group and folded to
    their credited acts. One act -> that act; two or three acts -> all of them
    (a multi-composer film credits every composer, never "Various Artists");
    more than three or none known -> fall back to the sole singer if there is
    exactly one, else "Various Artists".
    """
    composer_names: list[str] = []
    singer_names: list[str] = []
    for m in members:
        c = m.chosen
        if not c:
            continue
        composer_names.extend(c.composers)
        singer_names.extend(c.singers)

    acts = canonical_acts(composer_names)  # folded + deduped, first-seen order
    n = len(acts)

    if n == 1:
        return [acts[0]], acts
    if 2 <= n <= 3:
        # Faithful accounting: credit every known composer act.
        return list(acts), acts

    # n == 0 (no composers known) or n > 3 (anthology beyond our fidelity) ->
    # fall back. A single distinct singer becomes the album-artist; otherwise
    # the album is genuinely various-artists.
    singers = _dedupe_ordered(singer_names)
    if len(singers) == 1:
        return [singers[0]], acts
    return ["Various Artists"], acts


def _is_compilation_release(members: list[Resolution]) -> bool:
    """True only when the release itself is a compilation.

    Either a member's chosen candidate declares ``COMPILATION`` or its title
    text trips ``heuristics.looks_like_compilation`` ("Greatest Hits", etc.).
    A film soundtrack with many singers is NOT a compilation.
    """
    for m in members:
        c = m.chosen
        if not c:
            continue
        if c.release_type == ReleaseType.COMPILATION:
            return True
        if looks_like_compilation(c):
            return True
    return False


def reconcile_albums(resolutions: list[Resolution]) -> list[Resolution]:
    """Rewrite album-level tags in place so grouped tracks share one album."""
    groups: dict[str, list[Resolution]] = defaultdict(list)
    for r in resolutions:
        if r.chosen:
            groups[_album_key(r.chosen)].append(r)

    for key, members in groups.items():
        if len(members) < 1:
            continue
        album_name = _canonical_album_name(members)
        album_artist, acts = _album_artist_for_group(members)
        comp = _is_compilation_release(members)

        # Human-readable accounting of the composer folding + comp decision.
        if acts:
            acts_desc = ", ".join(acts)
        else:
            acts_desc = "none known"
        if len(acts) == 1:
            aa_desc = f"single composer act -> albumartist={album_artist[0]}"
        elif 2 <= len(acts) <= 3:
            aa_desc = (
                f"{len(acts)} composer acts, all credited -> "
                f"albumartist={album_artist}"
            )
        elif album_artist == ["Various Artists"]:
            aa_desc = "no single act -> albumartist=Various Artists"
        else:
            aa_desc = f"fallback to sole singer -> albumartist={album_artist[0]}"
        if comp:
            comp_desc = "comp=1 (release is a compilation)"
        else:
            comp_desc = (
                "comp=0 (film soundtrack; varying singers is normal, "
                "not a compilation)"
            )
        reason = (
            f"album '{album_name}': composer acts=[{acts_desc}]; "
            f"{aa_desc}; {comp_desc}"
        )

        for m in members:
            if album_name:
                m.tags["album"] = album_name
            m.tags["albumartist"] = album_artist
            m.tags["comp"] = 1 if comp else 0            # beets/iTunes compilation flag
            m.tags["album_key"] = key                     # shared grouping id
            if m.chosen and m.chosen.release_mbid:
                m.tags["musicbrainz_albumid"] = m.chosen.release_mbid
            m.reasons.append(reason)
    return resolutions
