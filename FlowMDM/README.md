# FlowMDM Benchmark Release

This repository is a HumanML3D-focused benchmark release based on FlowMDM. It is prepared for external distribution and includes:

- code for generation / evaluation / training
- HumanML3D checkpoints under `results/humanml`
- custom anaphora evaluation support

This release does not package a full HumanML3D dataset. You are expected to prepare the dataset separately and expose it through a local symlink.

## Environment Setup

The current repository was run with `uv 0.10.9` and Python `3.10.12`. PyTorch is pinned to `1.13.0`, so stay on Python 3.10 unless you want to re-resolve the environment yourself.

1. Install system packages:

```bash
sudo apt update
sudo apt install -y ffmpeg git unzip
```

2. Create the virtual environment with `uv`:

```bash
bash scripts/setup_uv_env.sh
source .venv/bin/activate
```

This script creates `.venv`, preinstalls the build tools required for `chumpy`, installs [`requirements-uv.txt`](requirements-uv.txt), and downloads `en_core_web_sm`.

Manual equivalent:

```bash
uv python install 3.10
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python pip setuptools==65.5.0 wheel
uv pip install --python .venv/bin/python --no-build-isolation chumpy==0.70
uv pip install --python .venv/bin/python -r requirements-uv.txt
uv pip install --python .venv/bin/python setuptools==65.5.0
.venv/bin/python -m spacy download en_core_web_sm
```

The dependency list in [`requirements-uv.txt`](requirements-uv.txt) is pinned against the current `uv` environment. Core runtime libraries include `torch==1.13.0`, `torchvision==0.14.0`, `transformers==4.25.1`, `numpy==1.23.4`, `spacy==3.4.3`, `trimesh==3.18.1`, `pyrender==0.1.45`, `matplotlib==3.1.3`, `scipy==1.9.3`, `gdown==4.6.0`, `blobfile==2.0.0`, `wandb==0.26.0`, and pinned Git dependencies for `CLIP` and `smplx`. The spaCy model `en_core_web_sm` is installed in the separate command above.

## Dataset Setup

Create a symlink so that `dataset/HumanML3D` points at your prepared HumanML3D root:

```bash
ln -s /path/to/HumanML3D ./dataset/HumanML3D
```

The target directory is expected to contain at least:

- `Mean.npy`
- `Std.npy`
- `train.txt`
- `val.txt`
- `test.txt`
- `humanml_test_set_anaphora.json`
- `humanml_test_set_anaphora_fullgen.json`
- `texts/`
- `new_joint_vecs/`
- `new_joints/`
- `t2m/`

## Download Required Assets

### 1. SMPL files

These files are **not redistributed** as part of this benchmark release or the shared GDrive archives.  
Users are expected to prepare `body_models/` separately, for example by running:

```bash
bash runners/prepare/download_smpl_files.sh
```

### 2. GloVe

```bash
bash runners/prepare/download_glove.sh
```

### 3. HumanML evaluator / normalization support files

This release expects a zip archive that expands into the current repository layout:

- `dataset/t2m_mean.npy`
- `dataset/t2m_std.npy`
- `dataset/HML_Mean_Gen.npy`
- `dataset/HML_Std_Gen.npy`
- `t2m/humanml/text_mot_match/model/finest.tar`

This archive is expected to contain only the evaluator / normalization support files above. It should **not** contain `body_models/`.

Placeholder GDrive link:

```text
https://drive.google.com/file/d/REPLACE_WITH_SUPPORT_FILES_ID/view?usp=sharing
```

Download and extract with:

```bash
FLOWMDM_SUPPORT_FILES_URL="https://drive.google.com/file/d/REPLACE_WITH_SUPPORT_FILES_ID/view?usp=sharing" \
  bash runners/prepare/download_t2m_evaluators.sh
```

Manual fallback:

```bash
gdown --fuzzy "https://drive.google.com/file/d/REPLACE_WITH_SUPPORT_FILES_ID/view?usp=sharing" -O flowmdm_support_files.zip
unzip -o flowmdm_support_files.zip -d .
rm flowmdm_support_files.zip
```

### 4. Pretrained HumanML checkpoints

This release ships HumanML checkpoints under `results/humanml`:

- `results/humanml/FlowMDM`
- `results/humanml/ELMAv3_anap_hml3d`
- `results/humanml/ELMAv3_anap_hml3d_hist`

This archive is expected to contain only the pretrained checkpoints above. It should **not** contain `body_models/`.

Placeholder GDrive link:

```text
https://drive.google.com/file/d/REPLACE_WITH_RESULTS_HUMANML_ID/view?usp=sharing
```

Download and extract with:

```bash
FLOWMDM_PRETRAINED_MODELS_URL="https://drive.google.com/file/d/REPLACE_WITH_RESULTS_HUMANML_ID/view?usp=sharing" \
  bash runners/prepare/download_pretrained_models.sh
```

Manual fallback:

```bash
gdown --fuzzy "https://drive.google.com/file/d/REPLACE_WITH_RESULTS_HUMANML_ID/view?usp=sharing" -O results_humanml_pretrained_models.zip
unzip -o results_humanml_pretrained_models.zip -d .
rm results_humanml_pretrained_models.zip
```

## Usage

The main generation / evaluation commands are documented in [`runners/README.md`](runners/README.md).

## Notes

- This benchmark release is centered on HumanML3D. Babel code is still present in the repository, but Babel checkpoints and evaluator assets are not bundled here.
- The `args.json` files under `results/humanml` are kept relative-path-safe for this repository layout.
- Release archive creation for maintainers is scripted in [`scripts/package_release_archives.sh`](scripts/package_release_archives.sh).
