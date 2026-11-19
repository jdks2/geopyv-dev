"""Regenerate images/shear/: synthetic Leighton Buzzard sand under shear.

Grain sizes (QicPic size curve, um) and aspect ratio are measured on Leighton
Buzzard sand; the brightness and blur are those fitted to real photographs of
it (geopyv_dev_pub piv_compare/match.py, Leighton candidate 2). The median
grain is drawn 8 px across (the sand imaged at about 0.12 mm/px). No camera
noise, so the demos show the method's own error.
"""

from pathlib import Path

import numpy as np

import geopyv_dev as gp

OUT = Path(__file__).resolve().parent / "shear"
OUT.mkdir(exist_ok=True)

PSD = np.array([
    [11.79, 0.0], [15.2, 0.01], [19.6, 0.01], [25.27, 0.013], [32.58, 0.02], [42.01, 0.033],
    [54.17, 0.05], [69.85, 0.083], [90.06, 0.15], [116.12, 0.263], [149.72, 0.483],
    [193.04, 0.857], [248.9, 1.46], [320.93, 2.29], [413.79, 3.303], [533.53, 4.537],
    [687.92, 6.973], [886.98, 32.047], [1143.65, 76.383], [1474.59, 96.197],
    [1901.29, 99.863], [2451.47, 100.0],
])

speckle = gp.Speckle(
    image_cfg={"image_dir": str(OUT), "name": "shear", "image_size": (1001, 1001), "file_format": ".png"},
    speckle_cfg={
        "soil": "sand",
        "grains": "distribution", "psd": PSD, "d_p": 8.0,
        "aspect_ratio": 0.697, "aspect_ratio_sd": 0.117,
        "matrix_brightness": 122, "matrix_spread": 0.3,
        "seed_brightness": 60, "seed_spread": 0.3, "asr": 0.14,
        "blur": 0.4,
    },
    progression="deformation",
    # du/dy = dv/dx = 0.01 about the image centre, reached over 5 images
    deformation_cfg={"comp": [0.0, 0.0, 0.0, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
    noise_cfg=(0.0, 0.0),
    scale_cfg={"scale": "lin", "n": 5},
)
speckle.solve(seed=42)
gp.save(str(OUT / "speckle_shear.pyv"), speckle)
print(f"wrote {sorted(p.name for p in OUT.iterdir())}")
