"""
PHVE 3D codec -- bijective encoding of anatomical volumes

Implementation of the encoding map F_p^{(3),alpha} of the paper:

    F_p^{(3),alpha}(x) = Enc o Hil_p^{(3)} o Nor_p^{(3),alpha}(x),

where Nor normalises (x, y, z) into the integer grid [0, 2^p - 1]^3 of
the chosen anatomical volume alpha (cranium, thorax, etc.), Hil is the
Skilling 3D Hilbert kernel (Algorithm 1 of the paper), and Enc encodes
the index in base 29 with an unambiguous alphabet.

Author: Paul Guindo, Altius Academy SNC.
"""

import math

# ===================================================================
# Base-29 alphabet (no O / I / L / 0 / 1 / U / Z to avoid ambiguity)
# ===================================================================

ALPHABET = "23456789ABCDEFGHJKMNPQRSTVWXY"
BASE = len(ALPHABET)  # 29
CHAR_TO_VAL = {c: i for i, c in enumerate(ALPHABET)}

# ===================================================================
# Reference anatomical volumes (PHVE Reference Family, d=3)
# ===================================================================

VOLUMES = {
    "CR": {"name": "Cranium",     "dims": (300, 250, 250)},
    "TX": {"name": "Thorax",      "dims": (400, 350, 300)},
    "CA": {"name": "Heart",       "dims": (150, 120, 100)},
    "AB": {"name": "Abdomen",     "dims": (400, 350, 300)},
    "MB": {"name": "Limb",        "dims": (700, 200, 200)},
    "FB": {"name": "Full body",   "dims": (2000, 600, 400)},
}

# ===================================================================
# 3D Hilbert curve -- Skilling (2004) algorithm
# ===================================================================
# "Programming the Hilbert curve", J. Skilling, AIP 2004.
# Works for any dimension; here n = 3.

def _coords_to_hilbert(coords, p):
    """Transpose-to-Hilbert: in-place transform of [x, y, z] integer
    coordinates into the Skilling-transposed Hilbert representation.
    """
    n = len(coords)
    M = 1 << (p - 1)

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
    """Hilbert-to-Transpose: inverse of _coords_to_hilbert."""
    n = len(coords)
    N = 2 << (p - 1)

    t = coords[n - 1] >> 1
    for i in range(n - 1, 0, -1):
        coords[i] ^= coords[i - 1]
    coords[0] ^= t

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
    """Interleave three p-bit coordinates into a single 3p-bit index."""
    d = 0
    for i in range(p):
        bit_x = (x >> i) & 1
        bit_y = (y >> i) & 1
        bit_z = (z >> i) & 1
        d |= (bit_x << (3 * i + 2)) | (bit_y << (3 * i + 1)) | (bit_z << (3 * i))
    return d


def _deinterleave_3d(d, p):
    """Split a 3p-bit interleaved index back into three p-bit coordinates."""
    x = y = z = 0
    for i in range(p):
        z |= ((d >> (3 * i)) & 1) << i
        y |= ((d >> (3 * i + 1)) & 1) << i
        x |= ((d >> (3 * i + 2)) & 1) << i
    return x, y, z


def xyz2d(p, x, y, z):
    """3D Hilbert encoding Hil_p^{(3)}: (x, y, z) -> index d in [0, 8^p - 1].

    p : order of the curve (grid 2^p x 2^p x 2^p).
    """
    coords = [x, y, z]
    _coords_to_hilbert(coords, p)
    return _interleave_3d(coords[0], coords[1], coords[2], p)


def d2xyz(p, d):
    """Inverse 3D Hilbert decoding (Hil_p^{(3)})^{-1}: d -> (x, y, z)."""
    x, y, z = _deinterleave_3d(d, p)
    coords = [x, y, z]
    _hilbert_to_coords(coords, p)
    return coords[0], coords[1], coords[2]


# ===================================================================
# Bijectivity verification
# ===================================================================

def verify_bijectivity(p):
    """Exhaustively verify bijectivity of xyz2d / d2xyz at order p."""
    n = 1 << p
    total = n * n * n
    seen = set()
    for x in range(n):
        for y in range(n):
            for z in range(n):
                d = xyz2d(p, x, y, z)
                if d in seen:
                    return False, f"Collision: d={d} for ({x},{y},{z})"
                seen.add(d)
                x2, y2, z2 = d2xyz(p, d)
                if (x2, y2, z2) != (x, y, z):
                    return False, f"Round-trip failed: ({x},{y},{z})->d={d}->({x2},{y2},{z2})"
    if len(seen) != total:
        return False, f"Incomplete coverage: {len(seen)}/{total}"
    return True, f"OK: {total} bijective points"


# ===================================================================
# Base-29 conversion
# ===================================================================

def _code_length_3d(p):
    """Length of a 3D code at order p (ceil(3p * log_29(2))).
    Matches Definition 2.3 of the paper for d=3, B=29.
    """
    return math.ceil(3 * p * math.log(2) / math.log(BASE))


def _canonical_precision_3d(k):
    """Largest order p such that _code_length_3d(p) <= k (round-trip safe)."""
    return int(k * math.log(BASE) / (3 * math.log(2)))


def _int_to_base29(d, length):
    """Integer -> base-29 string padded to `length` characters."""
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
    """Base-29 string -> integer."""
    d = 0
    for c in code.upper():
        if c not in CHAR_TO_VAL:
            raise ValueError(f"Invalid character: '{c}'")
        d = d * BASE + CHAR_TO_VAL[c]
    return d


# ===================================================================
# PHVE encode / decode
# ===================================================================

def encode(x_mm, y_mm, z_mm, p=8, volume="CR"):
    """Encode spatial coordinates (mm) into a PHVE 3D code.

    Args:
        x_mm, y_mm, z_mm: coordinates in mm inside the reference volume.
        p: Hilbert curve order.
        volume: anatomical volume prefix ("CR", "TX", "CA", ...).

    Returns:
        str: code formatted as "VOL:CCC-CCC-CC".
    """
    if volume not in VOLUMES:
        raise ValueError(f"Unknown volume: '{volume}'. Available: {list(VOLUMES.keys())}")

    dims = VOLUMES[volume]["dims"]
    n = 1 << p

    ix = max(0, min(n - 1, int(x_mm / dims[0] * n)))
    iy = max(0, min(n - 1, int(y_mm / dims[1] * n)))
    iz = max(0, min(n - 1, int(z_mm / dims[2] * n)))

    d = xyz2d(p, ix, iy, iz)

    k = _code_length_3d(p)
    code = _int_to_base29(d, k)

    parts = [code[i:i+3] for i in range(0, len(code), 3)]
    formatted = "-".join(parts)

    return f"{volume}:{formatted}"


def decode(code):
    """Decode a PHVE 3D code back to spatial coordinates (mm).

    Args:
        code: code formatted as "VOL:CCC-CCC-CC".

    Returns:
        dict: {"x_mm", "y_mm", "z_mm", "p", "volume", "resolution_mm"}.
    """
    if ":" not in code:
        raise ValueError("Invalid format: code must contain ':' (e.g. 'CR:4H7-KBR')")

    vol_prefix, base29_part = code.split(":", 1)
    vol_prefix = vol_prefix.upper()

    if vol_prefix not in VOLUMES:
        raise ValueError(f"Unknown volume: '{vol_prefix}'")

    base29_code = base29_part.replace("-", "")
    k = len(base29_code)
    p = _canonical_precision_3d(k)
    n = 1 << p

    dims = VOLUMES[vol_prefix]["dims"]

    d = _base29_to_int(base29_code)
    ix, iy, iz = d2xyz(p, d)

    x_mm = (ix + 0.5) * dims[0] / n
    y_mm = (iy + 0.5) * dims[1] / n
    z_mm = (iz + 0.5) * dims[2] / n

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
    """Truncate a code by n characters (zoom out -- Theorem 6.1).

    Returns a coarser code denoting an enclosing super-cell.
    """
    if ":" not in code:
        raise ValueError("Invalid format")

    vol_prefix, base29_part = code.split(":", 1)
    base29_code = base29_part.replace("-", "")

    if n_chars >= len(base29_code):
        return f"{vol_prefix}:"

    truncated = base29_code[:-n_chars]

    parts = [truncated[i:i+3] for i in range(0, len(truncated), 3)]
    formatted = "-".join(parts)

    return f"{vol_prefix}:{formatted}"


def encode_grid(p, volume="CR"):
    """Encode every grid index and return the {(ix, iy, iz): code} mapping.

    Useful for visualisation; for large grids, call encode() directly.
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
# NIfTI helpers
# ===================================================================

def voxel_to_code(voxel_ijk, affine, p=8, volume="CR"):
    """Voxel (i, j, k) of a NIfTI -> PHVE 3D code.

    Args:
        voxel_ijk: tuple of voxel indices in the NIfTI volume.
        affine: 4x4 NIfTI affine matrix (voxel -> mm).
        p: Hilbert curve order.
        volume: anatomical volume prefix.
    """
    import numpy as np
    ijk = np.array([voxel_ijk[0], voxel_ijk[1], voxel_ijk[2], 1.0])
    xyz_mm = affine @ ijk
    dims = VOLUMES[volume]["dims"]
    x = xyz_mm[0] + dims[0] / 2
    y = xyz_mm[1] + dims[1] / 2
    z = xyz_mm[2] + dims[2] / 2
    return encode(x, y, z, p=p, volume=volume)


def code_to_voxel(code, affine):
    """PHVE 3D code -> voxel (i, j, k) of a NIfTI volume."""
    import numpy as np
    result = decode(code)
    dims = VOLUMES[result["volume"]]["dims"]
    x_mm = result["x_mm"] - dims[0] / 2
    y_mm = result["y_mm"] - dims[1] / 2
    z_mm = result["z_mm"] - dims[2] / 2
    inv_affine = np.linalg.inv(affine)
    xyz = np.array([x_mm, y_mm, z_mm, 1.0])
    ijk = inv_affine @ xyz
    return (int(round(ijk[0])), int(round(ijk[1])), int(round(ijk[2])))


# ===================================================================
# Standalone smoke test
# ===================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  PHVE 3D codec -- bijectivity smoke test")
    print("=" * 60)

    ok, msg = verify_bijectivity(3)
    print(f"\n  p=3 : {msg}")

    print(f"\n  encode/decode test:")
    for vol in ["CR", "CA", "FB"]:
        dims = VOLUMES[vol]["dims"]
        x, y, z = dims[0] / 2, dims[1] / 2, dims[2] / 2  # centre
        code = encode(x, y, z, p=6, volume=vol)
        result = decode(code)
        print(f"    {vol} centre ({x:.0f}, {y:.0f}, {z:.0f}) mm")
        print(f"      -> {code}")
        print(f"      -> decoded: ({result['x_mm']:.1f}, {result['y_mm']:.1f}, {result['z_mm']:.1f}) mm")
        print(f"      -> resolution: {result['resolution_mm']:.1f} mm")

    code = encode(150, 125, 125, p=8, volume="CR")
    print(f"\n  Truncation hierarchy:")
    print(f"    Full code   : {code}")
    for i in range(1, 5):
        t = truncate(code, i)
        print(f"    Truncated -{i}: {t}")
