# Data and analysis scripts — XR vs. desktop shape-form estimation

Supporting material for:

> Danni Liu, Rui Wang, Gary Delaney, Bruce H. Thomas, and Jun Zhang.
> *Comparing XR and Desktop Interfaces for Shape-Form Estimation and Manipulation in a Scientific
> Context.* Submitted to IEEE Transactions on Visualization and Computer Graphics.

Three between-subjects experiments (N = 64) comparing an immersive XR interface against a
conventional desktop interface for interactive superquadric shape estimation.

---

## Contents

| Path | What it is |
|---|---|
| `user_study_data_deidentified.csv` | de-identified per-participant dataset, 64 × 112 |
| `interaction_analysis.py` | aligned rank transform analysis of expertise × interface |
| `barr_recompute.py` | independent reimplementation of the Barr normalised implicit distance |
| `fbx_reader.py` | minimal FBX mesh reader used by `barr_recompute.py` |
| `validate_all.py` | validates the Barr reimplementation against meshes of known parameters |
| `barr_validation_results.xlsx` | output of that validation |
| `study_data/` | study stimuli (superquadric and rock meshes) and derived result workbooks |

## The dataset

64 participants, one row each, 112 columns. Participants were randomly assigned to the XR or
desktop condition at registration.

Columns cover: condition and expertise group; demographics; a pre-study survey; per-experiment
performance; the four superquadric parameters (`a`, `b`, `c`, `m`) each participant settled on for
each of three target meshes; NASA-TLX responses for all three sessions; and gripper-task measures.

**De-identification.** Participants are identified only by a pseudonymous ID. Age is released in
five-year-plus bands rather than exact years, because exact age combined with gender and expertise
made 23 of the 64 participants individually distinguishable. Ten free-text response columns were
removed; none of them are analysed in the paper.

## Reproducing the analysis

```bash
pip install pandas numpy scipy
python interaction_analysis.py
```

That runs standalone from the CSV and reproduces the expertise × interface interaction tests
reported in §4.1.2 — an aligned rank transform (Wobbrock et al., 2011) for each of the three target
mesh types.

## Reproducing the Barr metric validation

```bash
pip install trimesh numpy scipy openpyxl pandas
python validate_all.py
```

`barr_recompute.py` implements the Barr normalised implicit distance

```
F(x, y, z) = (|x/a|^m + |y/b|^m + |z/c|^m)^(1/m) − 1
```

from its definition, and scores a fit as `100 × mean|F|` over points sampled on the target mesh,
minimised over the six permutations of the semi-axes `(a, b, c)`. `validate_all.py` checks that
implementation against `ValidateMesh_1..5`, whose generating parameters are known.

**One part will not run from this repository.** `barr_recompute.main()` reads the raw study workbook
`study_data/User_Study_Data_clean.xlsx`, which is not published here because it holds the
un-de-identified participant records. Everything downstream of it is included: the per-participant
results it produces are in `study_data/barr_analysis/` and `study_data/trial12_analysis/`, and the
mesh validation runs without it.

The rock stimulus is included as geometry plus its texture map. The XR source `.blend` and a
duplicate copy of the texture were left out of this repository for size.

---

## Licence

- **Code** (`*.py`) — MIT, see [`LICENSE`](LICENSE).
- **Data** (`*.csv`, `*.xlsx`, meshes and textures under `study_data/`) — CC BY 4.0, see
  [`LICENSE-DATA`](LICENSE-DATA).

If you use this material, please cite the paper above. Machine-readable citation metadata is in
[`CITATION.cff`](CITATION.cff); Zenodo deposition metadata is in `.zenodo.json`.

## DOI

<!-- After the first Zenodo release, replace this block with the badge Zenodo provides:
     [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
     Use the *concept* DOI (the one that always resolves to the latest version) in the paper. -->

A DOI is being minted via Zenodo; it will appear here and in the paper once issued.

## Ethics

Ethics approval was granted by the human research ethics committees of Swinburne University of
Technology and CSIRO prior to recruitment and data collection. All participants gave written
informed consent; participation was voluntary and participants could withdraw at any time without
penalty.
