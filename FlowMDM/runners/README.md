# Running FlowMDM (Benchmark Release)

This benchmark release is configured around HumanML3D. Before running any command, finish the setup steps in the top-level [`README.md`](../README.md):

- create the `uv` environment
- link `dataset/HumanML3D`
- download SMPL / GloVe / support files / pretrained checkpoints

## Generation

### Baseline FlowMDM

```bash
./.venv/bin/python -m runners.generate \
  --model_path ./results/humanml/FlowMDM/model000500000.pt \
  --dataset humanml \
  --num_repetitions 1 \
  --guidance_param 2.5 \
  --instructions_file ./runners/jsons/composition_humanml.json \
  --use_chunked_att
```

### ELMAv3 anaphora model

```bash
./.venv/bin/python -m runners.generate \
  --model_path ./results/humanml/ELMAv3_anap_hml3d/model000550000.pt \
  --dataset humanml \
  --num_repetitions 1 \
  --guidance_param 2.5 \
  --instructions_file ./runners/jsons/composition_humanml.json \
  --use_chunked_att
```

### ELMAv3 anaphora + history-text model

```bash
./.venv/bin/python -m runners.generate \
  --model_path ./results/humanml/ELMAv3_anap_hml3d_hist/model000550000.pt \
  --dataset humanml \
  --num_repetitions 1 \
  --guidance_param 2.5 \
  --instructions_file ./runners/jsons/composition_humanml.json \
  --use_history_text \
  --history_current_weight 0.5 \
  --use_chunked_att
```

If you want to sample from the linked HumanML3D test split instead of a JSON instruction file, replace `--instructions_file ...` with `--num_samples N`.

## Evaluation

### HumanML3D test_anaphora evaluation

The `test_anaphora` benchmark uses `./dataset/HumanML3D/humanml_test_set_anaphora.json`.

History-text model:

```bash
./.venv/bin/python -m runners.eval \
  --model_path ./results/humanml/ELMAv3_anap_hml3d_hist/model000550000.pt \
  --dataset humanml \
  --eval_mode final \
  --guidance_param 2.5 \
  --transition_length 60 \
  --scenario anaphora \
  --eval_file ./dataset/HumanML3D/humanml_test_set_anaphora.json \
  --use_history_text \
  --history_current_weight 0.5
```

No-history model:

```bash
./.venv/bin/python -m runners.eval \
  --model_path ./results/humanml/ELMAv3_anap_hml3d/model000550000.pt \
  --dataset humanml \
  --eval_mode final \
  --guidance_param 2.5 \
  --transition_length 60 \
  --scenario anaphora \
  --eval_file ./dataset/HumanML3D/humanml_test_set_anaphora.json
```

If you are using the outer `benchmark-code` repository, wrapper scripts are available at `../../scripts/eval_test_anaphora_hist.sh` and `../../scripts/eval_test_anaphora_nohist.sh`.

### HumanML3D anaphora full-generation evaluation

History-text model:

```bash
./.venv/bin/python -m runners.eval \
  --model_path ./results/humanml/ELMAv3_anap_hml3d_hist/model000550000.pt \
  --dataset humanml \
  --eval_mode final \
  --guidance_param 2.5 \
  --transition_length 60 \
  --scenario anaphora_fullgen \
  --eval_file ./dataset/HumanML3D/humanml_test_set_anaphora_fullgen.json \
  --use_history_text \
  --history_current_weight 0.5
```

No-history model:

```bash
./.venv/bin/python -m runners.eval \
  --model_path ./results/humanml/ELMAv3_anap_hml3d/model000550000.pt \
  --dataset humanml \
  --eval_mode final \
  --guidance_param 2.5 \
  --transition_length 60 \
  --scenario anaphora_fullgen \
  --eval_file ./dataset/HumanML3D/humanml_test_set_anaphora_fullgen.json
```

Baseline FlowMDM:

```bash
./.venv/bin/python -m runners.eval \
  --model_path ./results/humanml/FlowMDM/model000500000.pt \
  --dataset humanml \
  --eval_mode final \
  --guidance_param 2.5 \
  --transition_length 60 \
  --scenario anaphora_fullgen \
  --eval_file ./dataset/HumanML3D/humanml_test_set_anaphora_fullgen.json
```

`--eval_mode fast` is useful for a shorter smoke run. `--use_chunked_att` is recommended for long generations.

## Mesh Rendering

```bash
./.venv/bin/python -m runners.render_mesh --input_path /path/to/sample_rep00.mp4
```

This produces:

- `sample_rep##_smpl_params.npy`
- `sample_rep##_obj/`

## Training

Training code is kept in the repository, but this release primarily targets generation and evaluation from distributed checkpoints. If you retrain, make sure your local dataset layout matches the symlink setup described in the top-level README.
