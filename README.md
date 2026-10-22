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

For installing dependencies and running the analysis notebook, follow the
[original setup instructions](https://github.com/42elenz/LLMs_psychopathologies#setup-instructions);
skip the API-key section.

## Status

Code refactor in progress. Live runs against the remote model haven't been done yet;
see the "Deferred" section of [REFACTOR_PLAN.md](REFACTOR_PLAN.md).
