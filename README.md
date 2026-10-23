# Benchmarking LLMs Against Clinicians on Psychopathological Assessment: Local vLLM Refactor

> **Source:** This repository is a refactor of [42elenz/LLMs_psychopathologies](https://github.com/42elenz/LLMs_psychopathologies)
> (imported at commit [`c89a3ea`](https://github.com/42elenz/LLMs_psychopathologies/commit/c89a3ea)).
> All credit for the study design, data, prompts and analysis belongs to the original authors.
> For the study background, experimental design, data description, published results and the
> interactive visualizations, see the [original README](https://github.com/42elenz/LLMs_psychopathologies#readme).

## What this fork changes

The original pipeline rates simulated psychiatric interview transcripts with commercial LLM APIs
(OpenAI, Gemini, Anthropic, Mistral, Together) using API keys. This fork runs the **same pipeline against a
self-hosted model served by [vLLM](https://github.com/vllm-project/vllm)** on a remote GPU desktop,
reached through an SSH tunnel. No API keys or cloud accounts are needed.

**Unchanged:** prompt construction, definition chunking, the JSON schema, response parsing, chunk merging,
the output CSV format, and all statistics in the notebook, `src/utils/helper.py` and `src/R_analyses/`.
Commit `a52ef43` is a byte-identical import of upstream and serves as the reference for checking this.

| Area | Change |
|---|---|
| Inference backends | All cloud-provider code removed; `vllm` is the only supported `api_family` |
| Credentials | `api_keys/` and `.env` loading removed |
| Dependencies | Cloud SDKs (and dependencies used only by them) removed from `src/requirements.txt` |
| Model config | `src/utils/model_config.py` lists the vLLM-served model (currently `gemma-4-31B-it`) |
| vLLM call | Updated to the current vLLM structured-output API (`guided_json` was removed in vLLM 0.12); startup check that the server is reachable; configurable output length, seed and timeout |
| Timing appendix | `time_one_call.py` times the configured vLLM model instead of the paper's cloud models |

Plot colors for the paper's cloud models are kept in `site_config_colors.py` so the notebook can still
reproduce the published figures. The full step-by-step plan and its status are in [REFACTOR_PLAN.md](REFACTOR_PLAN.md).

## How inference connects

```
this computer                                   remote desktop (GPU)
00_call_models.py ──► localhost:11436 ══ SSH tunnel ══► vLLM server :11436 ──► gemma-4-31B-it
```

1. Open the tunnel (keep this terminal open while running):

   ```bash
   ssh -N <user>@<host> -p <ssh-port> -L 11436:localhost:11436
   ```

2. Check the model is reachable:

   ```bash
   curl http://localhost:11436/v1/models
   ```

3. Run the pipeline from `src/`. Arguments are the same as upstream, plus the vLLM options below:

   ```bash
   cd src
   python 00_call_models.py --outer_runs 1 --number_of_definitions 10 --prompt_language german --transcript_dir ../data/02_transcripts_example/
   ```

| vLLM option | Default | Purpose |
|---|---|---|
| `--vllm_url` | `http://localhost:11436/v1` | OpenAI-compatible endpoint (the tunnel) |
| `--vllm_max_tokens` | `8192` | Max output tokens per chunk (same as upstream) |
| `--vllm_seed` | none | Sampling seed, for reproducibility |
| `--vllm_timeout` | `900` | Request timeout in seconds |
| `VLLM_API_KEY` (env var) | none | Only needed if the server was started with `--api-key` |

### Install

Python 3.11+ (the pinned `numpy`/`scikit-learn` versions require it). From the repository root:

```bash
pip install -r src/requirements.txt
```

The appendix scripts in `src/appendix_code/` also need `tiktoken` (and `transformers` for `estimate_tokens.py`),
which upstream doesn't list in the requirements.

### Outputs

| Path | Contents |
|---|---|
| `outputs/AI_ratings/all_runs_combined_definitions_<n>_<timestamp>_<temperature>.csv` | One row per transcript × run, same columns as upstream |
| `src/raw_responses/` | Raw model output per chunk |
| `src/parsing_errors/` | Output of any chunk that failed; the remaining chunks and runs for that transcript are skipped (upstream behaviour) |

The `model_name` column holds the model's folder name (e.g. `gemma-4-31B-it`) rather than the server's full path.
All three output locations are gitignored.

## Analysing new ratings

`src/01_comparision_and_statistics.ipynb` loads the publication data by default and reproduces the paper unchanged.
To add your own runs, list the combined CSVs in `new_ratings_csvs` (cell `vllm-new-ratings`, right after the data is loaded).
Their sites are relabelled to the publication's naming convention:

| Run | Site key |
|---|---|
| temperature 0, 10 definitions per chunk | `gemma-4-31B-it_temp_0_10` |
| temperature 0, no definitions | `gemma-4-31B-it_temp_0_0` |
| other temperature | `gemma-4-31B-it_10` / `gemma-4-31B-it_0` |

These keys have colors in `src/utils/site_config_colors.py`. To use the local model as the reference AI in the
later comparisons, set `best_ai` in the next cell to one of these keys. Everything else (study design, data
description, and details of the analyses) is documented in the
[original README](https://github.com/42elenz/LLMs_psychopathologies#readme).

## Inference server

| Setting | Value |
|---|---|
| Host | Remote GPU desktop, reached via SSH tunnel |
| Endpoint (local end of tunnel) | `http://localhost:11436/v1` |
| Served model id | `/vllm/models/gemma-4-31B-it` |
| Max context (`max_model_len`) | 262,144 tokens |
| vLLM version | not recorded yet (`curl http://localhost:11436/version`) |
| Server launch flags | not recorded yet |

Record the version and flags here once the tunnel is working, since results can't be reproduced without them.

## Local test server (optional)

For testing the pipeline without the remote desktop, a small model can be served on a laptop GPU.
Tested on an RTX 4060 Laptop (8 GB) under WSL2 Ubuntu 24.04 with vLLM 0.32.0 and
[`Qwen/Qwen2.5-7B-Instruct-AWQ`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ) (Apache-2.0, revision `b250375`).
Its ratings are for checking the pipeline end to end and aren't comparable to the remote model.

**One-time setup** (inside WSL):

```bash
sudo apt update && sudo apt install -y build-essential     # gcc, needed by Triton's runtime compiler
curl -LsSf https://astral.sh/uv/install.sh | sh            # uv package manager (user-local)
uv venv --managed-python --python 3.12 ~/venvs/vllm        # uv's Python ships Python.h; Ubuntu's doesn't
source ~/venvs/vllm/bin/activate && uv pip install vllm==0.32.0
```

**Start the server** (downloads the ~5.5 GB model on first start, then serves on `127.0.0.1:8000`):

```bash
bash scripts/start_vllm.sh
```

**Run the pipeline against it:** in `src/utils/model_config.py`, use the commented-out Qwen entry instead of Gemma, then:

```bash
cd src
python 00_call_models.py --vllm_url http://localhost:8000/v1 --vllm_max_tokens 3072 --prompt_language german --transcript_dir ../data/02_transcripts_example/
```

**Limits on 8 GB:**

| Item | Detail |
|---|---|
| Context | 22,016 tokens (fp16 KV cache). The longest prompt is 18,775 tokens (German, 10 definitions per chunk). |
| German runs | need `--vllm_max_tokens 3072` so prompt + reply fit; a 10-item reply is about 500 tokens |
| English runs | fit the default `--vllm_max_tokens 8192` (longest prompt 13,303 tokens) |
| `--number_of_definitions 0` | all 100 items in one reply; may not fit in the space left for German (untested) |
| Speed | about 50 s per German 10-definition chunk, so roughly 25 min per full German run |

`scripts/start_vllm.sh` explains each server flag. In short: GPU memory use 0.86 (WSL leaves ~6.9 of 8 GiB free);
Triton attention and the PyTorch sampler (FlashInfer would need the CUDA toolkit); no CUDA graphs and 2k-token
prefill chunks to free memory; compact JSON output (stops runs of whitespace); and an fp16 KV cache, because fp8
garbled this model's output.

## Status

The code refactor is complete. Live runs against the remote model haven't been done yet;
see the "Deferred" section of [REFACTOR_PLAN.md](REFACTOR_PLAN.md).
