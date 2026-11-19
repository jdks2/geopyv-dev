# geopyv_dev demos

Short, runnable scripts that walk through the package in order. Each one can be
run from any folder:

```bash
pip install geopyv_dev
python demo/0_images.py
```

| Script | Shows | Needs |
|---|---|---|
| `0_images.py` | Images, local/global masks, regions, generating synthetic speckle | - |
| `1_subset.py` | Correlating one subset; inspect, convergence, save/load | - |
| `2_mesh.py` | Full-field correlation of one image pair; contours, rates | - |
| `3_sequence.py` | Correlating a whole image series; writes `output/sequence.pyv` | - |
| `4_particle.py` | Following one material point; histories and traces | `3_sequence.py` |
| `5_field.py` | A field of particles; strain contours; writes `output/field.pyv` | `3_sequence.py` |
| `6_validation.py` | Measured against the known applied deformation | `5_field.py` |
| `7_calibration.py` | ChArUco camera calibration; pixels to mm | `pip install geopyv_dev[calibration]` |

`images/shear/` holds five synthetic images under 1 % simple shear, made with
`gp.Speckle`; `speckle_shear.pyv` holds their true deformation, which
`6_validation.py` compares against. `images/calibration/` holds 25 photographs
of a ChArUco board (29 x 18 squares, 10 mm squares, 8 mm markers) for
`7_calibration.py`; replace them with your own board images for a real test.

Results go in `output/` (not tracked by git). Plots open in their own windows;
close each to continue.

The same steps are explained in more detail in the tutorials:
<https://geopyv-dev.readthedocs.io/en/latest/>.

## GUI

Everything here can also be done from the desktop GUI, installed with the
package:

```bash
geopyv-gui
```
