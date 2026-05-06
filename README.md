![ELMABench](Image/ELMABench.png)

# ELMABench_dev

`benchmark-code` is a HumanML3D anaphora benchmark repository built around `FlowMDM`.  
The actual generation and evaluation code lives in [`FlowMDM/`](./FlowMDM), while `benchmark-code/scripts/` provides small wrapper scripts for setup and `test_anaphora` evaluation.

## Main Files

- [`FlowMDM/README.md`](./FlowMDM/README.md): detailed `FlowMDM` setup instructions
- [`FlowMDM/runners/README.md`](./FlowMDM/runners/README.md): detailed generation and evaluation commands
- [`scripts/setup_flowmdm_env.sh`](./scripts/setup_flowmdm_env.sh): creates the `uv` environment
- [`scripts/prepare_flowmdm_assets.sh`](./scripts/prepare_flowmdm_assets.sh): downloads GloVe, evaluator support files, and pretrained models
- [`scripts/eval_test_anaphora_hist.sh`](./scripts/eval_test_anaphora_hist.sh): runs `test_anaphora` evaluation with the history-text model
- [`scripts/eval_test_anaphora_nohist.sh`](./scripts/eval_test_anaphora_nohist.sh): runs `test_anaphora` evaluation with the no-history model

## Environment Setup

1. Install the required system packages.

```bash
sudo apt update
sudo apt install -y ffmpeg git unzip
```

2. Create `FlowMDM/.venv`.

```bash
cd /path/to/benchmark-code
bash scripts/setup_flowmdm_env.sh
```

3. Link HumanML3D into `FlowMDM/dataset/HumanML3D`.

```bash
ln -s /path/to/HumanML3D ./FlowMDM/dataset/HumanML3D
```

At minimum, the linked dataset root must contain:

- `Mean.npy`
- `Std.npy`
- `train.txt`
- `val.txt`
- `test.txt`
- `test_anaphora.txt`
- `humanml_test_set_anaphora.json`
- `humanml_test_set_anaphora_fullgen.json`
- `texts/`
- `new_joint_vecs/`
- `new_joints/`
- `t2m/`

4. Prepare the SMPL body model files.

These files are **not redistributed** in our release archives.  
The expected setup is that each user prepares `body_models/` separately, for example by running the upstream download script:

```bash
cd /path/to/benchmark-code/FlowMDM
bash runners/prepare/download_smpl_files.sh
```

5. Download the remaining support files and pretrained models.

```bash
export FLOWMDM_SUPPORT_FILES_URL="https://drive.google.com/file/d/REPLACE_WITH_SUPPORT_FILES_ID/view?usp=sharing"
export FLOWMDM_PRETRAINED_MODELS_URL="https://drive.google.com/file/d/REPLACE_WITH_RESULTS_HUMANML_ID/view?usp=sharing"

cd /path/to/benchmark-code
bash scripts/prepare_flowmdm_assets.sh
```

`FLOWMDM_SUPPORT_FILES_URL` should point to a zip that expands to `dataset/t2m_mean.npy`, `dataset/t2m_std.npy`, `dataset/HML_Mean_Gen.npy`, `dataset/HML_Std_Gen.npy`, and `t2m/humanml/text_mot_match/model/finest.tar`.  
`FLOWMDM_PRETRAINED_MODELS_URL` should point to a zip that expands to the pretrained checkpoints under `FlowMDM/results/humanml`.  
Neither archive is expected to contain `body_models/`.

## test_anaphora Evaluation

In this benchmark, `test_anaphora` refers to evaluation using `FlowMDM/dataset/HumanML3D/test_anaphora.txt` together with `FlowMDM/dataset/HumanML3D/humanml_test_set_anaphora.json`.  
The wrapper scripts are fixed to the following checkpoints:

- history-text enabled:
  `FlowMDM/results/humanml/ELMAv3_anap_hml3d_hist/model000550000.pt`
- history-text disabled:
  `FlowMDM/results/humanml/ELMAv3_anap_hml3d/model000550000.pt`

### History-text Model

```bash
cd /path/to/benchmark-code
GPU_ID=0 EVAL_MODE=final bash scripts/eval_test_anaphora_hist.sh
```

### No-History Model

```bash
cd /path/to/benchmark-code
GPU_ID=0 EVAL_MODE=final bash scripts/eval_test_anaphora_nohist.sh
```

For a shorter smoke run, use `EVAL_MODE=fast`.

```bash
GPU_ID=0 EVAL_MODE=fast bash scripts/eval_test_anaphora_hist.sh
GPU_ID=0 EVAL_MODE=fast bash scripts/eval_test_anaphora_nohist.sh
```

Outputs:

- history-text model:
  `FlowMDM/results/humanml/ELMAv3_anap_hml3d_hist/evaluation/`
- no-history model:
  `FlowMDM/results/humanml/ELMAv3_anap_hml3d/evaluation/`

Summary JSON files are written under each model directory's `evaluations_summary/`.

## Script Summary

- [`scripts/setup_flowmdm_env.sh`](./scripts/setup_flowmdm_env.sh)
  Calls `FlowMDM/scripts/setup_uv_env.sh` to create `FlowMDM/.venv`.
- [`scripts/prepare_flowmdm_assets.sh`](./scripts/prepare_flowmdm_assets.sh)
  Runs `download_glove.sh`, `download_t2m_evaluators.sh`, and `download_pretrained_models.sh` in sequence. It does not prepare `body_models/`; SMPL files must be handled separately.
- [`scripts/eval_test_anaphora_hist.sh`](./scripts/eval_test_anaphora_hist.sh)
  Evaluates `test_anaphora` with `ELMAv3_anap_hml3d_hist/model000550000.pt`, using `--scenario anaphora` together with `--use_history_text --history_current_weight 0.5`.
- [`scripts/eval_test_anaphora_nohist.sh`](./scripts/eval_test_anaphora_nohist.sh)
  Evaluates `test_anaphora` with `ELMAv3_anap_hml3d/model000550000.pt` without history-text conditioning.

Both evaluation scripts accept `GPU_ID` and `EVAL_MODE` as environment variables, and any extra `runners.eval` arguments can be appended at the end.
