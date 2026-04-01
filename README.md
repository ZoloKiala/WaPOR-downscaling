# WaPOR 20 m Downscaler

This repository contains the CatBoost-based workflow used to downscale WaPOR ETa data to 20 m resolution, evaluate model outputs, summarize results, and interact with the deployed Gradio application.

## Quick Start

1. Request an API key or access token if the hosted inference service requires authentication.
2. Install the Gradio client:

```bash
pip install gradio_client
```

3. Run inference through the Gradio app:

```python
from gradio_client import Client, handle_file

client = Client("IWMIHQ/WaPOR_20_m_Downscaler")

result = client.predict(
    aoi_file=handle_file(
        "https://github.com/gradio-app/gradio/raw/main/test/test_files/sample_file.pdf"
    ),
    date_str="2026-03-11",
    model_choice="WaPOR-AETI-DownScaler v1.0",
    api_name="/run_inference_gradio",
)

print(result)
```

## Repository Contents

- `wapor_downscale_catboost_datesplit_both.py`
  Main CatBoost training script. Supports CPU/GPU execution, feature subset search, optional W&B logging, optional TPOT, and optional upload of local datasets as W&B artifacts.
- `per_file_eval_catboost.py`
  Runs per-file evaluation using a saved CatBoost model bundle.
- `model_output.py`
  Generates summary tables and plots from CatBoost outputs.
- `plot_lamego_best_map.py`
  Produces the best-date visualization map for Lamego or Baixo.

## Data Layout

Expected local data folders:

- `wapor_downscale_data/BAIXO_STACK_S2_MATCH_L3_20M_FULL_1`
- `wapor_downscale_data/LAMEGO_STACK_S2_MATCH_L3_20M_FULL_1`

## Environment

Use the base conda environment:

```bash
conda activate base
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

## Train the CatBoost Model

Run the model on CPU:

```bash
python wapor_downscale_catboost_datesplit_both.py \
  --data-dir wapor_downscale_data/BAIXO_STACK_S2_MATCH_L3_20M_FULL_1 \
  --eval-dir wapor_downscale_data/LAMEGO_STACK_S2_MATCH_L3_20M_FULL_1 \
  --out-dir outputs_catboost_datesplit \
  --task-type CPU \
  --n-samples-total 300000 \
  --subset-trials 40 \
  --val-frac 0.15 \
  --per-file-min 500 \
  --per-file-max 4000 \
  --max-slope-deg 25 \
  --veg-sample-frac 0.7 \
  --veg-ndvi-thr 0.35 \
  --veg-fvc-thr 0.2 \
  --min-groups 2 \
  --n-jobs -1 \
  --seed 7
```

Run the model on CPU with W&B logging and local dataset artifact upload:

```bash
python wapor_downscale_catboost_datesplit_both.py \
  --data-dir wapor_downscale_data/BAIXO_STACK_S2_MATCH_L3_20M_FULL_1 \
  --eval-dir wapor_downscale_data/LAMEGO_STACK_S2_MATCH_L3_20M_FULL_1 \
  --out-dir outputs_catboost_datesplit \
  --task-type CPU \
  --n-samples-total 300000 \
  --subset-trials 40 \
  --val-frac 0.15 \
  --per-file-min 500 \
  --per-file-max 4000 \
  --max-slope-deg 25 \
  --veg-sample-frac 0.7 \
  --veg-ndvi-thr 0.35 \
  --veg-fvc-thr 0.2 \
  --min-groups 2 \
  --n-jobs -1 \
  --seed 7 \
  --wandb \
  --wandb-entity zolokiala-iwmi \
  --wandb-project wapor-downscale-catboost \
  --wandb-job-type train_catboost \
  --wandb-log-local-datasets \
  --wandb-train-artifact-name baixo_stack_s2_match_l3_20m_full_1 \
  --wandb-eval-artifact-name lamego_stack_s2_match_l3_20m_full_1
```

Primary trained model output:

```text
outputs_catboost_datesplit/catboost_best.joblib
```

## Evaluate the Saved Model

```bash
python per_file_eval_catboost.py \
  --model-bundle outputs_catboost_datesplit/catboost_best.joblib \
  --lamego-eval-dir wapor_downscale_data/LAMEGO_STACK_S2_MATCH_L3_20M_FULL_1 \
  --baixo-eval-dir wapor_downscale_data/BAIXO_STACK_S2_MATCH_L3_20M_FULL_1 \
  --out-dir outputs_perfile_eval \
  --save-preds
```

## Generate Summary Outputs

```bash
python model_output.py
```

## Plot Best-Date Maps

Lamego:

```bash
python plot_lamego_best_map.py \
  --metrics-csv outputs_perfile_eval/LAMEGO/per_file_metrics_lamego.csv \
  --site lamego
```

Baixo:

```bash
python plot_lamego_best_map.py \
  --metrics-csv outputs_perfile_eval/BAIXO/per_file_metrics_baixo.csv \
  --site baixo
```

## W&B Dataset Sharing Notes

- Do not share your personal W&B API key with other users.
- If you want anyone to download the dataset artifacts, make the target W&B project public.
- Artifact access follows project visibility.
- After logging local datasets as artifacts, users can download them with the artifact path:
  `entity/project/artifact_name:latest`

Example artifact paths:

- `zolokiala-iwmi/wapor-downscale-catboost/baixo_stack_s2_match_l3_20m_full_1:latest`
- `zolokiala-iwmi/wapor-downscale-catboost/lamego_stack_s2_match_l3_20m_full_1:latest`

## Gradio Client Usage

Install the client:

```bash
pip install gradio_client
```

The deployed app is:

```text
IWMIHQ/WaPOR_20_m_Downscaler
```

### 1. Run Inference Over an AOI

Use `handle_file(...)` for the AOI input file. Replace the sample file URL with your actual AOI file if needed.

```python
from gradio_client import Client, handle_file

client = Client("IWMIHQ/WaPOR_20_m_Downscaler")

result = client.predict(
    aoi_file=handle_file(
        "https://github.com/gradio-app/gradio/raw/main/test/test_files/sample_file.pdf"
    ),
    date_str="2026-03-11",
    model_choice="WaPOR-AETI-DownScaler v1.0",
    api_name="/run_inference_gradio",
)

print(result)
```

Notes:

- `date_str` must be passed as a string.
- `model_choice` must be quoted as a string.
- Replace the sample file with the correct AOI file format expected by the app.

### 2. Update the Drawer Map Search

```python
from gradio_client import Client

client = Client("IWMIHQ/WaPOR_20_m_Downscaler")

result = client.predict(
    search_query="Hello!!",
    api_name="/update_drawer_map",
)

print(result)
```

### 3. Load the Example AOI

```python
from gradio_client import Client

client = Client("IWMIHQ/WaPOR_20_m_Downscaler")

result = client.predict(
    api_name="/load_example_aoi",
)

print(result)
```

### 4. Request WaPOR Time Series and Forecast Over an AOI

```python
from gradio_client import Client, handle_file

client = Client("IWMIHQ/WaPOR_20_m_Downscaler")

result = client.predict(
    aoi_file=handle_file(
        "https://github.com/gradio-app/gradio/raw/main/test/test_files/sample_file.pdf"
    ),
    start_str="2025-07-11",
    end_str="2026-03-11",
    do_forecast=True,
    forecast_method="Statistical (Holt)",
    forecast_horizon=6,
    api_name="/wapor_and_pred_timeseries_over_aoi",
)

print(result)
```

Notes:

- Use Python boolean `True`, not `true`.
- `forecast_method` must be quoted as a string.
- `forecast_horizon` is passed as an integer.

## Gradio Client Tips

- If authentication is required for the hosted app, run `huggingface-cli login` first or configure the required token in your environment.
- If the app interface changes, inspect the latest endpoint schema before reusing old client calls.
- For production use, replace sample URLs and placeholder search queries with real inputs.

## Outputs

Typical generated outputs include:

- `outputs_catboost_datesplit/catboost_best.joblib`
- `outputs_catboost_datesplit/best_subset.json`
- `outputs_catboost_datesplit/subset_trials.csv`
- `outputs_catboost_datesplit/feature_importance_selected.csv`
- `outputs_perfile_eval/...`
- `agg300m_rmse_bootstrap_and_tests.csv`
- `feature_group_selection_frequency.png`
- `top15_feature_importance.png`

## Notes

- `catboost_info/` is a normal CatBoost runtime output directory.
- `wandb/` contains W&B run metadata and artifacts cached locally.
- The repository was cleaned to keep CatBoost-related scripts and outputs only.
