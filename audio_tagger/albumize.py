"""Album-level reconciliation.

Music players group tracks into an album by (album name + album-artist),
and some also by album id. If album-artist is chosen per-track, a film with
several singers/composers explodes into many one-track "albums". This pass
runs AFTER per-track resolution: it groups tracks that belong to the same
release and rewrites the album-level fields to be *identical* across the
group, so the film stays as one album.

Rules applied per group:
  * ALBUM        : one normalized string for every track.
  * ALBUMARTIST  : the single music director if the film has exactly one;
                   otherwise "Various Artists".
  * comp (TCMP)  : 1 when the album spans multiple track artists — the flag
                   players use to keep a multi-artist album together.
  * MUSICBRAINZ_ALBUMID / album key: shared, for id-based grouping.
  * disc/track numbers preserved if present.
Per-track fields (ARTIST=singers, COMPOSER, LYRICIST, TITLE) are untouched.
"""

from __future__ import annotations

from collections import defaultdict

from .models import Candidate, Resolution


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


def _album_artist_for_group(members: list[Resolution]) -> tuple[list[str], bool]:
    """Return (albumartist, is_multi_artist).

    One music director across the whole film -> that director.
    Multiple (or none) -> Various Artists + compilation flag.
    """
    composer_sets = []
    singer_names = set()
    for m in members:
        c = m.chosen
        if not c:
            continue
        if c.composers:
            composer_sets.append(tuple(sorted({x.strip() for x in c.composers})))
        singer_names.update(s.strip() for s in c.singers)

    distinct_composers = set()
    for cs in composer_sets:
        distinct_composers.update(cs)

    multi_artist = len(singer_names) > 1

    if len(distinct_composers) == 1:
        return [next(iter(distinct_composers))], multi_artist
    if distinct_composers:  # several music directors across the film
        return ["Various Artists"], True
    # No composer info (indipop/punjabi single-artist album): keep the sole
    # album-artist if there's exactly one; else Various Artists.
    if len(singer_names) == 1:
        return [next(iter(singer_names))], False
    return ["Various Artists"], multi_artist


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
        album_artist, multi = _album_artist_for_group(members)

        for m in members:
            if album_name:
                m.tags["album"] = album_name
            m.tags["albumartist"] = album_artist
            m.tags["comp"] = 1 if multi else 0          # beets/iTunes compilation flag
            m.tags["album_key"] = key                    # shared grouping id
            if m.chosen and m.chosen.release_mbid:
                m.tags["musicbrainz_albumid"] = m.chosen.release_mbid
            if multi:
                m.reasons.append(
                    f"album '{album_name}': multi-artist -> albumartist={album_artist[0]}, "
                    f"comp=1 (keeps the film as one album)"
                )
    return resolutions
