"""
Altius-Code Codec — Prototype 1 (adressage geographique 2D)

Bijection entre coordonnees GPS et codes alphanumeriques compacts
via les courbes de Hilbert.

Theorie : Paul Guindo, Altius Academy SNC.
"""

import math

from constants import DOMAINS

# ---------------------------------------------------------------------------
# Alphabet base-29 (sans O/I/L/0/1/U/Z pour eviter les ambiguites)
# ---------------------------------------------------------------------------

ALPHABET = "23456789ABCDEFGHJKMNPQRSTVWXY"
BASE = len(ALPHABET)  # 29

CHAR_TO_VAL = {c: i for i, c in enumerate(ALPHABET)}

# ---------------------------------------------------------------------------
# Precision canonique
# ---------------------------------------------------------------------------
# La longueur du code k = ceil(2p * ln2 / ln29) n'est pas injective sur p.
# Pour garantir la bijectivite du round-trip encode/decode, on utilise la
# precision canonique : pour un k donne, p_canon = floor(k * ln29 / (2*ln2)),
# c'est-a-dire le plus grand p dont le code tient en k symboles.
# ---------------------------------------------------------------------------


def _code_length(p, n=2):
    """Nombre de symboles base-29 necessaires pour encoder n*p bits.

    k = ceil(n * p * ln(2) / ln(29))
    """
    return math.ceil(n * p * math.log(2) / math.log(BASE))


def _canonical_precision(k):
    """Plus grand ordre p tel que _code_length(p) <= k.

    C'est l'inverse de _code_length : p = floor(k * ln(29) / (2 * ln(2))).
    """
    return int(k * math.log(BASE) / (2 * math.log(2)))


def _snap_precision(p):
    """Renvoie la precision canonique effective pour un p demande.

    Calcule k = _code_length(p), puis retourne _canonical_precision(k).
    """
    k = _code_length(p)
    return _canonical_precision(k)


# ---------------------------------------------------------------------------
# Courbe de Hilbert 2D — Algorithmes 1, 2, 3 du papier
# ---------------------------------------------------------------------------


def _rot2d(s, x, y, rx, ry):
    """Rotation de quadrant (Algorithm 3)."""
    if ry == 0:
        if rx == 1:
            x = s - 1 - x
            y = s - 1 - y
        x, y = y, x
    return x, y


def _xy2d(p, x, y):
    """Encodage Hilbert 2D : (x, y) -> index d (Algorithm 1).

    p : ordre de la courbe (grille 2^p x 2^p)
    """
    n = 1 << p
    d = 0
    s = n >> 1
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        x, y = _rot2d(s, x, y, rx, ry)
        s >>= 1
    return d


def _d2xy(p, d):
    """Decodage Hilbert 2D : index d -> (x, y) (Algorithm 2).

    p : ordre de la courbe (grille 2^p x 2^p)
    """
    n = 1 << p
    x = y = 0
    s = 1
    while s < n:
        rx = 1 if (d & 2) > 0 else 0
        ry = 1 & (d ^ rx)
        x, y = _rot2d(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        d >>= 2
        s <<= 1
    return x, y


# ---------------------------------------------------------------------------
# Conversion base-29
# ---------------------------------------------------------------------------


def _int_to_base29(d, length):
    """Conversion entier -> chaine base 29, avec padding a `length` caracteres."""
    if d == 0:
        return ALPHABET[0] * length
    chars = []
    val = d
    while val > 0:
        chars.append(ALPHABET[val % BASE])
        val //= BASE
    while len(chars) < length:
        chars.append(ALPHABET[0])
    chars.reverse()
    return "".join(chars)


def _base29_to_int(code):
    """Conversion chaine base 29 -> entier."""
    d = 0
    for c in code:
        c_upper = c.upper()
        if c_upper not in CHAR_TO_VAL:
            raise ValueError(f"Caractere invalide dans le code: '{c}'")
        d = d * BASE + CHAR_TO_VAL[c_upper]
    return d


# ---------------------------------------------------------------------------
# Resolution d'une precision donnee
# ---------------------------------------------------------------------------


def resolution(p, domain="NE"):
    """Retourne la resolution approximative en metres pour une precision p."""
    dom = DOMAINS[domain]
    lat_range = dom["lat_max"] - dom["lat_min"]
    lon_range = dom["lon_max"] - dom["lon_min"]
    m = 1 << p
    dlat = lat_range / m
    dlon = lon_range / m
    lat_m = dlat * 111_000
    lon_m = (
        dlon * 111_000 * math.cos(math.radians((dom["lat_min"] + dom["lat_max"]) / 2))
    )
    return max(lat_m, lon_m)


# ---------------------------------------------------------------------------
# Fonctions principales
# ---------------------------------------------------------------------------


def encode(lat, lon, precision=12, domain="NE"):
    """Encode des coordonnees GPS en code Altius-Code.

    La precision demandee est ajustee a la precision canonique (le plus grand
    p pour la longueur de code calculee) afin de garantir le round-trip.

    Args:
        lat: latitude (degres decimaux)
        lon: longitude (degres decimaux)
        precision: ordre p souhaite de la courbe de Hilbert (3..20)
        domain: prefixe du domaine geographique

    Returns:
        str: code au format "{PREFIX}-{BASE29}" (ex: "NE-4H7A3B")
    """
    if domain not in DOMAINS:
        raise ValueError(
            f"Domaine inconnu: '{domain}'. Disponibles: {list(DOMAINS.keys())}"
        )

    dom = DOMAINS[domain]

    # Snap a la precision canonique pour garantir la bijectivite
    k = _code_length(precision)
    p = _canonical_precision(k)
    m = 1 << p

    # Normalisation (Def. 5.1) : lat -> y, lon -> x
    x = min(m - 1, int((lon - dom["lon_min"]) / (dom["lon_max"] - dom["lon_min"]) * m))
    y = min(m - 1, int((lat - dom["lat_min"]) / (dom["lat_max"] - dom["lat_min"]) * m))
    x = max(0, x)
    y = max(0, y)

    # Index Hilbert
    d = _xy2d(p, x, y)

    # Encodage base-29
    code = _int_to_base29(d, k)

    return f"{domain}-{code}"


def decode(code):
    """Decode un code Altius-Code en coordonnees GPS.

    Args:
        code: code au format "{PREFIX}-{BASE29}" (ex: "NE-4H7A3B")

    Returns:
        dict: {"lat", "lon", "precision", "domain", "bounds"}
    """
    if "-" not in code:
        raise ValueError(
            "Format invalide: le code doit contenir un tiret (ex: 'NE-4H7A3B')"
        )

    prefix, base29_code = code.split("-", 1)
    prefix = prefix.upper()

    if prefix not in DOMAINS:
        raise ValueError(f"Prefixe de domaine inconnu: '{prefix}'")

    dom = DOMAINS[prefix]

    # Precision canonique a partir de la longueur du code
    k = len(base29_code)
    p = _canonical_precision(k)
    m = 1 << p

    # Decodage base-29 -> index Hilbert -> (x, y)
    d = _base29_to_int(base29_code)
    x, y = _d2xy(p, d)

    # Denormalisation (centre de la cellule)
    lon = dom["lon_min"] + (x + 0.5) * (dom["lon_max"] - dom["lon_min"]) / m
    lat = dom["lat_min"] + (y + 0.5) * (dom["lat_max"] - dom["lat_min"]) / m

    bounds = _cell_bounds(x, y, p, dom)

    return {
        "lat": round(lat, 8),
        "lon": round(lon, 8),
        "precision": p,
        "domain": prefix,
        "bounds": bounds,
    }


def get_bounds(code):
    """Retourne les bornes de la cellule correspondant au code.

    Returns:
        dict: {"lat_min", "lat_max", "lon_min", "lon_max"}
    """
    return decode(code)["bounds"]


def _cell_bounds(x, y, p, dom):
    """Calcule les bornes geographiques d'une cellule (x, y) dans le domaine."""
    m = 1 << p
    lat_step = (dom["lat_max"] - dom["lat_min"]) / m
    lon_step = (dom["lon_max"] - dom["lon_min"]) / m
    return {
        "lat_min": round(dom["lat_min"] + y * lat_step, 8),
        "lat_max": round(dom["lat_min"] + (y + 1) * lat_step, 8),
        "lon_min": round(dom["lon_min"] + x * lon_step, 8),
        "lon_max": round(dom["lon_min"] + (x + 1) * lon_step, 8),
    }


def neighbors(code):
    """Retourne les 8 codes voisins de la cellule.

    Returns:
        list[str]: liste de codes voisins (peut en contenir moins de 8 aux bords)
    """
    if "-" not in code:
        raise ValueError("Format invalide")

    prefix, base29_code = code.split("-", 1)
    prefix = prefix.upper()

    if prefix not in DOMAINS:
        raise ValueError(f"Prefixe de domaine inconnu: '{prefix}'")

    k = len(base29_code)
    p = _canonical_precision(k)
    m = 1 << p

    d = _base29_to_int(base29_code)
    cx, cy = _d2xy(p, d)

    result = []
    for dx, dy in [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]:
        nx, ny = cx + dx, cy + dy
        if 0 <= nx < m and 0 <= ny < m:
            nd = _xy2d(p, nx, ny)
            ncode = _int_to_base29(nd, k)
            result.append(f"{prefix}-{ncode}")

    return result
