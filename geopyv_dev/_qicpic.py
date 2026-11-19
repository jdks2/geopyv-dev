"""Grain-outline statistics from a QicPic particle report (.docx), for ``gp.Speckle``'s ``outlines`` setting.

The report holds one picture per particle: a 1-bit silhouette and its measured values (EQPC, Feret widths,
sphericity, aspect ratio). Each single grain's outline is written as radius waves about its centre of area,
r(t) = r0 E(t) (1 + sum_n a_n cos(n t + ph_n)), with E the ellipse of the grain's second moments of area
(its elongation); the statistics the generator needs are:

- ``irregularity``, ``irregularity_sd``: median and log-spread of S, the rms of a_3..a_8;
- ``profile``, ``profile_sd``: median and log-spread of a_n / S for n = 3..WAVES;
- ``dent``: quantiles (0, 5, ..., 100 %) of the deepest dent of the WAVES-wave outline (largest distance from
  the outline to its convex hull, over the equal-area diameter), measured as ``src/soil.rs`` measures it;
- ``aspect_ratio``, ``aspect_ratio_sd``: mean and standard deviation of the reported aspect ratios.

Particles whose traced outline has a dent deeper than ``PINCH`` x the equal-area diameter are two grains
overlapping at a pinch (a figure of eight) and are left out. Sphericity is not used: QicPic's value is set mostly
by fine roughness and pixel steps (a traced perfect circle scores 0.948), not by the shape a camera sees.
"""
import functools
import os
import re
import struct
import zipfile

import contourpy
import numpy as np
from scipy import ndimage

WAVES = 24          # must match soil::WAVES
PINCH = 0.12
N_ANG = 512
N_PTS = 360         # outline points for the dent, as soil::Shape::polygon


def _read_emf(b):
    """(silhouette bool array or None, {label: text}) from one QicPic EMF picture."""
    i, texts, mask = 0, [], None
    while i + 8 <= len(b):
        t, n = struct.unpack_from("<II", b, i)
        if t == 84:                                   # EMR_EXTTEXTOUTW
            nch, off = struct.unpack_from("<II", b, i + 44)
            texts.append(b[i + off:i + off + 2 * nch].decode("utf-16le"))
        elif t == 81:                                 # EMR_STRETCHDIBITS
            off_bmi, _, off_bits = struct.unpack_from("<III", b, i + 48)
            _, w, h, _, bpp = struct.unpack_from("<IiiHH", b, i + off_bmi)
            if bpp == 1:
                stride = ((w + 31) // 32) * 4
                raw = np.frombuffer(b, np.uint8, stride * abs(h), i + off_bits).reshape(abs(h), stride)
                bits = np.unpackbits(raw, axis=1)[:, :w]
                mask = bits[::-1] if h > 0 else bits  # bottom-up rows
        if n == 0:
            break
        i += n
    return mask, dict(zip(texts[0::2], texts[1::2]))


def read_report(path):
    """Silhouettes (largest blob, bool) and reported aspect ratios from a QicPic .docx report."""
    masks, aspect = [], []
    with zipfile.ZipFile(path) as z:
        for name in sorted(n for n in z.namelist() if n.startswith("word/media/") and n.lower().endswith(".emf")):
            mask, vals = _read_emf(z.read(name))
            if mask is None or "Aspect ratio" not in vals:
                continue
            sil = mask == 0 if mask.mean() > 0.5 else mask == 1   # the particle is the minority colour
            lab, k = ndimage.label(sil)
            if k == 0:
                continue
            big = lab == 1 + np.argmax(ndimage.sum(sil, lab, range(1, k + 1)))
            masks.append(big)
            aspect.append(float(re.sub(r"[^\d.]", "", vals["Aspect ratio"])))
    if not masks:
        raise ValueError(f"no QicPic particle pictures found in {path}")
    return masks, np.array(aspect)


def _outline(mask):
    """Sub-pixel outline (x, y) of a silhouette, counter-clockwise."""
    z = np.pad(mask, 1).astype(float)
    lines = contourpy.contour_generator(z=z).lines(0.5)
    p = max(lines, key=len)
    p = p[:-1] if np.allclose(p[0], p[-1]) else p
    x, y = p[:, 0], p[:, 1]
    if np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)) < 0:
        p = p[::-1]
    return p


def _hull_dent(p):
    """Deepest dent of a closed polygon (largest distance from its points to the convex hull's sides) and its
    equal-area diameter."""
    from scipy.spatial import ConvexHull
    x, y = p[:, 0], p[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    v = np.sort(ConvexHull(p).vertices)
    deep = 0.0
    for a, b in zip(v, np.roll(v, -1)):
        idx = np.arange(a + 1, b) if b > a else np.r_[np.arange(a + 1, len(p)), np.arange(0, b)]
        if len(idx) == 0:
            continue
        e = p[b] - p[a]
        dist = np.abs(e[0] * (p[idx, 1] - p[a, 1]) - e[1] * (p[idx, 0] - p[a, 0])) / np.hypot(*e)
        deep = max(deep, dist.max())
    return deep, 2 * np.sqrt(area / np.pi)


def _moments(p):
    """Area, centre of area and second moments (about the centre) of a closed polygon."""
    x, y = p[:, 0], p[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cr = x * yn - xn * y
    a = cr.sum() / 2
    cx, cy = ((x + xn) * cr).sum() / (6 * a), ((y + yn) * cr).sum() / (6 * a)
    ixx = ((x * x + x * xn + xn * xn) * cr).sum() / 12 / a - cx * cx
    iyy = ((y * y + y * yn + yn * yn) * cr).sum() / 12 / a - cy * cy
    ixy = ((x * yn + 2 * x * y + 2 * xn * yn + xn * y) * cr).sum() / 24 / a - cx * cy
    return a, np.array([cx, cy]), ixx, iyy, ixy


def _ellipse(t, q, phi):
    """Radius of an ellipse with semi-axes 1 along phi and q across it."""
    return 1 / np.sqrt(np.cos(t - phi) ** 2 + (np.sin(t - phi) / q) ** 2)


def _radius(p):
    """r(theta) at N_ANG even angles about the centre of area (outermost point in each angle bin)."""
    _, c, _, _, _ = _moments(p)
    # densify so every angle bin holds points
    seg = np.roll(p, -1, 0) - p
    k = np.maximum(1, np.ceil(np.hypot(*seg.T) * 4).astype(int))
    t = np.concatenate([np.arange(n) / n for n in k])
    q = np.repeat(p, k, 0) + np.repeat(seg, k, 0) * t[:, None] - c
    th = np.arctan2(q[:, 1], q[:, 0]) % (2 * np.pi)
    b = np.minimum((th / (2 * np.pi) * N_ANG).astype(int), N_ANG - 1)
    r = np.zeros(N_ANG)
    np.maximum.at(r, b, np.hypot(q[:, 0], q[:, 1]))
    return r


def _wave_polygon(a, ph):
    """N_PTS-point outline of r = 1 + sum_{n>=2} a_n cos(n t + ph_n) (a, ph indexed from n = 2)."""
    t = 2 * np.pi * np.arange(N_PTS) / N_PTS
    n = np.arange(2, 2 + len(a))
    r = 1 + (a[None] * np.cos(np.outer(t, n) + ph[None])).sum(1)
    return np.c_[r * np.cos(t), r * np.sin(t)]


def statistics(masks, aspect):
    """Outline statistics (dict) from silhouettes and their reported aspect ratios."""
    S, prof, dents, keep = [], [], [], []
    t = 2 * np.pi * np.arange(N_ANG) / N_ANG
    for m, ar in zip(masks, aspect):
        p = _outline(m)
        deep, d_eq = _hull_dent(p)
        if deep > PINCH * d_eq:
            continue
        r = _radius(p)
        # the grain's elongation: the ellipse of its second moments; the
        # irregularity is what the outline adds to it
        _, _, ixx, iyy, ixy = _moments(p)
        l1, l2 = np.linalg.eigvalsh([[ixx, ixy], [ixy, iyy]])[::-1]
        phi = 0.5 * np.arctan2(2 * ixy, ixx - iyy)
        f = np.fft.rfft(r / _ellipse(t, np.sqrt(l2 / l1), phi)) / N_ANG
        a = 2 * np.abs(f[1:WAVES + 1]) / f[0].real            # waves 1..WAVES
        s = np.sqrt((a[2:8] ** 2).sum())
        # the dent of the real outline smoothed to WAVES waves
        g = np.fft.rfft(r) / N_ANG
        deep, d_eq = _hull_dent(_wave_polygon(2 * np.abs(g[2:WAVES + 1]) / g[0].real, np.angle(g[2:WAVES + 1])))
        S.append(s)
        prof.append(a[2:] / s)
        dents.append(deep / d_eq)
        keep.append(ar)
    if len(S) < 10:
        raise ValueError(f"only {len(S)} single grains in the report: too few for outline statistics")
    lS, lp = np.log(S), np.log(np.array(prof))
    return {"grains": len(S), "irregularity": float(np.exp(np.median(lS))), "irregularity_sd": float(lS.std()),
            "profile": np.exp(np.median(lp, 0)).tolist(), "profile_sd": lp.std(0).tolist(),
            "dent": np.percentile(dents, np.arange(0, 101, 5)).tolist(),
            "aspect_ratio": float(np.mean(keep)), "aspect_ratio_sd": float(np.std(keep))}


@functools.lru_cache(maxsize=8)
def _calibrate(path, mtime):
    return statistics(*read_report(path))


def calibrate(path):
    """Outline statistics of the single grains in a QicPic .docx report (cached while the file is unchanged)."""
    path = os.path.abspath(os.fspath(path))
    if not path.lower().endswith(".docx"):
        raise ValueError(f"outlines must be a QicPic particle report (.docx), got {path}")
    return dict(_calibrate(path, os.path.getmtime(path)))
