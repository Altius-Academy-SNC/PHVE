"""
Altius-Code 3D — Encodage bijectif de volumes anatomiques

Bijection entre coordonnées spatiales (x, y, z) et codes alphanumériques
compacts via la courbe de Hilbert 3D.

Auteur : Paul Guindo, Altius Academy SNC
"""

import math

# ===================================================================
# Alphabet base-29 (sans O/I/L/0/1/U/Z)
# ===================================================================

ALPHABET = "23456789ABCDEFGHJKMNPQRSTVWXY"
BASE = len(ALPHABET)  # 29
CHAR_TO_VAL = {c: i for i, c in enumerate(ALPHABET)}

# ===================================================================
# Volumes anatomiques de référence
# ===================================================================

VOLUMES = {
    "CR": {"name": "Crâne",       "dims": (300, 250, 250)},
    "TX": {"name": "Thorax",      "dims": (400, 350, 300)},
    "CA": {"name": "Coeur",       "dims": (150, 120, 100)},
    "AB": {"name": "Abdomen",     "dims": (400, 350, 300)},
    "MB": {"name": "Membre",      "dims": (700, 200, 200)},
    "FB": {"name": "Corps entier","dims": (2000, 600, 400)},
}

# ===================================================================
# Courbe de Hilbert 3D — Algorithme de Skilling (2004)
# ===================================================================
# "Programming the Hilbert curve" — John Skilling, AIP 2004
# Fonctionne pour toute dimension. Ici n=3 (3D).

def _coords_to_hilbert(coords, p):
    """Transpose-to-Hilbert: transforme les coordonnées en index Hilbert.

    coords: liste mutable [x, y, z] (modifiée en place)
    p: nombre de bits par coordonnée
    """
    n = len(coords)
    M = 1 << (p - 1)

    # Inverse undo
    Q = M
    while Q > 1:
        P = Q - 1
        for i in range(n):
            if coords[i] & Q:
                coords[0] ^= P
            else:
                t = (coords[0] ^ coords[i]) & P
                coords[0] ^= t
                coords[i] ^= t
        Q >>= 1

    # Gray encode
    for i in range(1, n):
        coords[i] ^= coords[i - 1]

    t = 0
    Q = M
    while Q > 1:
        if coords[n - 1] & Q:
            t ^= Q - 1
        Q >>= 1
    for i in range(n):
        coords[i] ^= t


def _hilbert_to_coords(coords, p):
    """Hilbert-to-Transpose: transforme l'index Hilbert en coordonnées.

    coords: liste mutable [x, y, z] (modifiée en place)
    p: nombre de bits par coordonnée
    """
    n = len(coords)
    N = 2 << (p - 1)

    # Gray decode
    t = coords[n - 1] >> 1
    for i in range(n - 1, 0, -1):
        coords[i] ^= coords[i - 1]
    coords[0] ^= t

    # Undo excess work
    M = 2
    while M != N:
        P = M - 1
        for i in range(n - 1, -1, -1):
            if coords[i] & M:
                coords[0] ^= P
            else:
                t = (coords[0] ^ coords[i]) & P
                coords[0] ^= t
                coords[i] ^= t
        M <<= 1


def _interleave_3d(x, y, z, p):
    """Entrelace 3 coordonnées de p bits en un index de 3p bits."""
    d = 0
    for i in range(p):
        bit_x = (x >> i) & 1
        bit_y = (y >> i) & 1
        bit_z = (z >> i) & 1
        d |= (bit_x << (3 * i + 2)) | (bit_y << (3 * i + 1)) | (bit_z << (3 * i))
    return d


def _deinterleave_3d(d, p):
    """Désentralace un index de 3p bits en 3 coordonnées de p bits."""
    x = y = z = 0
    for i in range(p):
        z |= ((d >> (3 * i)) & 1) << i
        y |= ((d >> (3 * i + 1)) & 1) << i
        x |= ((d >> (3 * i + 2)) & 1) << i
    return x, y, z


def xyz2d(p, x, y, z):
    """Encodage Hilbert 3D : (x, y, z) -> index d.

    p : ordre de la courbe (grille 2^p × 2^p × 2^p)
    Retourne d ∈ [0, 8^p - 1]
    """
    coords = [x, y, z]
    _coords_to_hilbert(coords, p)
    return _interleave_3d(coords[0], coords[1], coords[2], p)


def d2xyz(p, d):
    """Décodage Hilbert 3D : index d -> (x, y, z).

    p : ordre de la courbe
    Retourne (x, y, z) ∈ [0, 2^p - 1]^3
    """
    x, y, z = _deinterleave_3d(d, p)
    coords = [x, y, z]
    _hilbert_to_coords(coords, p)
    return coords[0], coords[1], coords[2]


# ===================================================================
# Vérification de bijectivité
# ===================================================================

def verify_bijectivity(p):
    """Vérifie la bijectivité de xyz2d/d2xyz pour l'ordre p."""
    n = 1 << p
    total = n * n * n
    seen = set()
    for x in range(n):
        for y in range(n):
            for z in range(n):
                d = xyz2d(p, x, y, z)
                if d in seen:
                    return False, f"Doublon: d={d} pour ({x},{y},{z})"
                seen.add(d)
                x2, y2, z2 = d2xyz(p, d)
                if (x2, y2, z2) != (x, y, z):
                    return False, f"Round-trip échoué: ({x},{y},{z})->d={d}->({x2},{y2},{z2})"
    if len(seen) != total:
        return False, f"Couverture incomplète: {len(seen)}/{total}"
    return True, f"OK: {total} points bijectifs"


# ===================================================================
# Conversion base-29
# ===================================================================

def _code_length_3d(p):
    """Nombre de caractères base-29 pour un code 3D d'ordre p."""
    return math.ceil(3 * p * math.log(2) / math.log(BASE))


def _canonical_precision_3d(k):
    """Plus grand ordre p tel que _code_length_3d(p) <= k."""
    return int(k * math.log(BASE) / (3 * math.log(2)))


def _int_to_base29(d, length):
    """Entier -> chaîne base 29, paddée à `length` caractères."""
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
    """Chaîne base 29 -> entier."""
    d = 0
    for c in code.upper():
        if c not in CHAR_TO_VAL:
            raise ValueError(f"Caractère invalide: '{c}'")
        d = d * BASE + CHAR_TO_VAL[c]
    return d


# ===================================================================
# Encodage / Décodage de volumes anatomiques
# ===================================================================

def encode(x_mm, y_mm, z_mm, p=8, volume="CR"):
    """Encode des coordonnées spatiales (mm) en code Altius-Code 3D.

    Args:
        x_mm, y_mm, z_mm: coordonnées en mm dans le volume de référence
        p: ordre de la courbe de Hilbert
        volume: préfixe du volume anatomique ("CR", "TX", "CA", ...)

    Returns:
        str: code au format "VOL:CCC-CCC-CC"
    """
    if volume not in VOLUMES:
        raise ValueError(f"Volume inconnu: '{volume}'. Disponibles: {list(VOLUMES.keys())}")

    dims = VOLUMES[volume]["dims"]
    n = 1 << p

    # Normaliser les coordonnées sur [0, 2^p - 1]
    ix = max(0, min(n - 1, int(x_mm / dims[0] * n)))
    iy = max(0, min(n - 1, int(y_mm / dims[1] * n)))
    iz = max(0, min(n - 1, int(z_mm / dims[2] * n)))

    d = xyz2d(p, ix, iy, iz)

    k = _code_length_3d(p)
    code = _int_to_base29(d, k)

    # Formater avec tirets tous les 3 caractères
    parts = [code[i:i+3] for i in range(0, len(code), 3)]
    formatted = "-".join(parts)

    return f"{volume}:{formatted}"


def decode(code):
    """Décode un code Altius-Code 3D en coordonnées spatiales (mm).

    Args:
        code: code au format "VOL:CCC-CCC-CC"

    Returns:
        dict: {"x_mm", "y_mm", "z_mm", "p", "volume", "resolution_mm"}
    """
    if ":" not in code:
        raise ValueError("Format invalide: le code doit contenir ':' (ex: 'CR:4H7-KBR')")

    vol_prefix, base29_part = code.split(":", 1)
    vol_prefix = vol_prefix.upper()

    if vol_prefix not in VOLUMES:
        raise ValueError(f"Volume inconnu: '{vol_prefix}'")

    # Retirer les tirets
    base29_code = base29_part.replace("-", "")
    k = len(base29_code)
    p = _canonical_precision_3d(k)
    n = 1 << p

    dims = VOLUMES[vol_prefix]["dims"]

    d = _base29_to_int(base29_code)
    ix, iy, iz = d2xyz(p, d)

    # Dénormaliser (centre du voxel)
    x_mm = (ix + 0.5) * dims[0] / n
    y_mm = (iy + 0.5) * dims[1] / n
    z_mm = (iz + 0.5) * dims[2] / n

    # Résolution
    res = max(dims[0] / n, dims[1] / n, dims[2] / n)

    return {
        "x_mm": round(x_mm, 3),
        "y_mm": round(y_mm, 3),
        "z_mm": round(z_mm, 3),
        "p": p,
        "volume": vol_prefix,
        "resolution_mm": round(res, 3),
    }


def truncate(code, n_chars=1):
    """Tronque un code de n caractères (zoom arrière).

    Retourne un code de précision inférieure désignant un volume englobant.
    """
    if ":" not in code:
        raise ValueError("Format invalide")

    vol_prefix, base29_part = code.split(":", 1)
    base29_code = base29_part.replace("-", "")

    if n_chars >= len(base29_code):
        return f"{vol_prefix}:"

    truncated = base29_code[:-n_chars]

    # Reformater avec tirets
    parts = [truncated[i:i+3] for i in range(0, len(truncated), 3)]
    formatted = "-".join(parts)

    return f"{vol_prefix}:{formatted}"


def encode_grid(p, volume="CR"):
    """Encode tous les indices de grille et retourne le mapping.

    Utile pour la visualisation : retourne un dict {(ix,iy,iz): code}.
    Pour les grandes grilles, utiliser encode() directement.
    """
    n = 1 << p
    k = _code_length_3d(p)
    mapping = {}
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                d = xyz2d(p, ix, iy, iz)
                code = _int_to_base29(d, k)
                mapping[(ix, iy, iz)] = code
    return mapping


# ===================================================================
# Fonctions utilitaires pour volumes NIfTI
# ===================================================================

def voxel_to_code(voxel_ijk, affine, p=8, volume="CR"):
    """Convertit un voxel (i, j, k) d'un NIfTI en code Altius-Code 3D.

    Args:
        voxel_ijk: tuple (i, j, k) indices du voxel dans le volume NIfTI
        affine: matrice affine 4x4 du NIfTI (voxel -> mm)
        p: ordre de la courbe
        volume: préfixe du volume anatomique

    Returns:
        str: code Altius-Code 3D
    """
    import numpy as np
    ijk = np.array([voxel_ijk[0], voxel_ijk[1], voxel_ijk[2], 1.0])
    xyz_mm = affine @ ijk
    # Les coordonnées NIfTI peuvent être négatives (RAS/LPS)
    # On décale vers les positifs en utilisant les dimensions du volume
    dims = VOLUMES[volume]["dims"]
    x = xyz_mm[0] + dims[0] / 2
    y = xyz_mm[1] + dims[1] / 2
    z = xyz_mm[2] + dims[2] / 2
    return encode(x, y, z, p=p, volume=volume)


def code_to_voxel(code, affine):
    """Convertit un code Altius-Code 3D en indices voxel NIfTI.

    Args:
        code: code Altius-Code 3D
        affine: matrice affine 4x4 du NIfTI

    Returns:
        tuple: (i, j, k) indices voxel
    """
    import numpy as np
    result = decode(code)
    dims = VOLUMES[result["volume"]]["dims"]
    # Coordonnées mm (centrées sur l'origine du volume de référence)
    x_mm = result["x_mm"] - dims[0] / 2
    y_mm = result["y_mm"] - dims[1] / 2
    z_mm = result["z_mm"] - dims[2] / 2
    # Inverse de la matrice affine
    inv_affine = np.linalg.inv(affine)
    xyz = np.array([x_mm, y_mm, z_mm, 1.0])
    ijk = inv_affine @ xyz
    return (int(round(ijk[0])), int(round(ijk[1])), int(round(ijk[2])))


# ===================================================================
# Main — Test rapide
# ===================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Altius-Code 3D — Test de bijectivité")
    print("=" * 60)

    # Test bijectivité p=3
    ok, msg = verify_bijectivity(3)
    print(f"\n  p=3 : {msg}")

    # Test encode/decode
    print(f"\n  Test encode/decode :")
    for vol in ["CR", "CA", "FB"]:
        dims = VOLUMES[vol]["dims"]
        x, y, z = dims[0] / 2, dims[1] / 2, dims[2] / 2  # centre
        code = encode(x, y, z, p=6, volume=vol)
        result = decode(code)
        print(f"    {vol} centre ({x:.0f}, {y:.0f}, {z:.0f}) mm")
        print(f"      -> {code}")
        print(f"      -> décodé: ({result['x_mm']:.1f}, {result['y_mm']:.1f}, {result['z_mm']:.1f}) mm")
        print(f"      -> résolution: {result['resolution_mm']:.1f} mm")

    # Test hiérarchie par troncature
    code = encode(150, 125, 125, p=8, volume="CR")
    print(f"\n  Hiérarchie par troncature :")
    print(f"    Code complet : {code}")
    for i in range(1, 5):
        t = truncate(code, i)
        print(f"    Tronqué -{i}   : {t}")
