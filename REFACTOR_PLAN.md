# Refactor Plan: Local vLLM Inference for LLMs_psychopathologies

**Source:** `42elenz/LLMs_psychopathologies` @ `c89a3ea` (cloned in `./LLMs_psychopathologies`)
**Target repo:** `matthewsjzhang/llm-benchmarking-lenz-refactor`
**Complexity:** Medium. Most of the risk is in environment setup; the code changes are small.

## Goal

Run the rating pipeline against a local LLM served by vLLM instead of cloud APIs with API keys.
**Every algorithm and calculation stays unchanged:** prompt construction, definition chunking,
JSON parsing, chunk merging, CSV layout, and all analysis in the notebook, `helper.py` and R.
Only the code that sends the prompt to a model and gets text back is touched.

## What the code does today (grounding)

| Area | Location | Notes |
|---|---|---|
| Entry point | `src/00_call_models.py` | CLI args; already has `--vllm_url`, which it exports as `VLLM_BASE_URL` |
| Model selection | `src/utils/model_config.py` | list of `(api_family, model_name)` tuples; a `("vllm", ...)` example is commented out |
| Client setup | `src/utils/helper_inference.py:109` `initialize_clients` | builds a vLLM `OpenAI(base_url=..., api_key="EMPTY")` client; no key is needed |
| LLM call | `src/utils/helper_inference.py:624` `call_api` (vllm branch) | `chat.completions.create`; system = `chunk_prompt + schema_instruction`; `max_tokens=8192`; `extra_body={"guided_json": schema}` |
| Retry | `helper_inference.py:653` | 5 attempts, 20s exponential backoff; "connection refused" counts as retryable |
| Key loading | `helper_inference.py:22`, `appendix_code/time_one_call.py:14` | `load_dotenv("../api_keys/.env")` at import time |
| Cloud SDK imports | `helper_inference.py:11-18` | google-genai, together, anthropic, mistralai are imported at module load, so they must be installed even if unused |
| Analysis keying | `helper.py:343` | each AI rating set is looked up by its `site_key` in `site_config_colors.py`; a key that isn't listed there won't be plotted |
| Paths | everywhere | relative (`../data`, `./raw_responses`), so scripts **must run from `src/`** |

### Problems found

1. **`guided_json` was removed in vLLM v0.12.0.** The existing vLLM branch therefore fails on any current vLLM.
   The replacements are `response_format={"type":"json_schema", ...}` or `extra_body={"structured_outputs":{"json": ...}}`.
2. **Hard-coded `max_tokens=8192`.** A German transcript (~39 KB, about 12k tokens) plus a 10-definition chunk (about 5k tokens) plus 8k output
   comes to roughly 25k tokens. vLLM rejects any request where prompt + `max_tokens` exceeds the server's `--max-model-len`.
3. **Cloud SDKs load at import time**, and so does `load_dotenv`. The pipeline can't run without those packages installed, even when it only uses vLLM.
4. **Hardware.** The GPU is an RTX 4060 Laptop with 8 GB VRAM, alongside 31 GB RAM, on Windows with no WSL and no Python installed.
   vLLM only runs on Linux, so this setup needs **WSL2 + Ubuntu**. 8 GB of VRAM limits us to roughly a **7–8B model quantized to 4 bits (AWQ/GPTQ)**
   with a context of about 32k tokens.
5. **One upstream file is named `src/__init__.py ` (trailing space)**, which Windows can't create (handled in Step 0).
6. **Upstream `.gitignore` ignores `*.png`.** Copying it into your repo would silently drop `data/03_ressources/*.png` and other images.

## Open decisions (please answer before Step 1)

| # | Decision | Recommendation |
|---|---|---|
| D1 | Which local model? | **`Qwen/Qwen2.5-7B-Instruct-AWQ`**: non-reasoning, strong at JSON and German, 32k native context, fits in 8 GB. Alternative: `Qwen/Qwen3-8B-AWQ` with thinking disabled. |
| D2 | What happens to the cloud providers (OpenAI, Gemini, ...)? | **Keep them dormant:** move their imports inside their branches (lazy imports) and drop `load_dotenv`. The diff stays small, no keys or SDKs are required, and the original code stays recognisable. Alternative: delete the branches entirely. |
| D3 | Where does the Python client run? | **Inside WSL**, next to the vLLM server: one Linux environment and no Windows/WSL networking issues. The notebook can still be opened from Windows VS Code via the WSL extension. |
| D4 | How does upstream code get into your repo? | **Snapshot copy** (files only, no upstream history) in a single commit "Import upstream @ c89a3ea". This avoids the trailing-space index problem. Alternative: keep upstream history via a remote plus merge, which needs the one-off `core.protectNTFS` workaround. |
| D5 | Where does vLLM run? | **Remote desktop over an SSH tunnel when available, otherwise a port on this laptop** (see Step 1b). The pipeline only ever talks to `localhost`, so the code is identical either way. If the remote GPU is bigger, you may pick a larger model there (D1 per host). |

## Working agreement for every step

- One step = one task. I finish it, run its **Validate** checks, and then **stop at a checkpoint**.
- At each checkpoint I report what changed, paste the validation output, and suggest a commit message.
- **You review, commit and push manually.** I don't commit or push.
- The next step starts only when you say go.

---

## Steps

### Step 0: Bring the upstream code into your repo
- **Action:** Copy everything from `LLMs_psychopathologies/` (except `.git/`) into the root of `llm-benchmarking-lenz-refactor`. Create the empty `src/__init__.py` under its correct name.
  Merge upstream's `.gitignore` into the repo-root `.gitignore`, which already excludes Claude working files, envs, caches, generated output and model weights.
  Do **not** carry over upstream's blanket `*.png` rule, so the tracked images are kept.
  Delete or move away the nested `LLMs_psychopathologies/` clone once the copy is verified.
- **No code changes.**
- **Validate:** file count matches upstream (113 minus the trailing-space file plus `src/__init__.py`); `git status` shows only additions; spot-check hashes of the notebook and the CSVs.
- **Checkpoint:** you commit "Import upstream LLMs_psychopathologies @ c89a3ea" and push.

### Step 1: Set up vLLM completely (single step)
- **Action:**
  1. Install WSL2 + Ubuntu 24.04 (`wsl --install -d Ubuntu-24.04`). **You run this yourself**: it needs admin rights and probably a reboot.
  2. In WSL, confirm the GPU is visible (`nvidia-smi`). The Windows driver (CUDA 13.4) supplies CUDA to WSL; **no Linux NVIDIA driver is installed.**
  3. Install `uv`, create a dedicated **server venv** (`~/venvs/vllm`, Python 3.12) and `pip install vllm` (pinned to the current stable release, recorded in the repo).
  4. Download the model chosen in D1 into the Hugging Face cache.
  5. Add `scripts/start_vllm.sh` to the repo. It holds the exact `vllm serve` command and its tuned flags:
     `--served-model-name` (a stable name used in the CSVs), `--max-model-len 32768`, `--gpu-memory-utilization ~0.90`,
     `--kv-cache-dtype fp8` if needed to fit, `--max-num-seqs` small (we send one request at a time), `--seed 0`, port 8000.
  6. Tune for 8 GB: if the server fails to start with an out-of-memory (OOM) error, step down in this order: `kv-cache-dtype fp8`, then lower `gpu-memory-utilization`, then `--enforce-eager`, then reduce `max-model-len` to 24576.
- **Validate:**
  - `curl localhost:8000/v1/models` lists the model, both from inside WSL and from Windows PowerShell.
  - One structured-output request (`response_format` json_schema with a toy schema) returns valid JSON.
  - One **real-sized** request (the longest German transcript plus a 10-definition prompt, `max_tokens` 8192) completes without a context-length error. Record its latency.
- **Checkpoint:** you commit `scripts/start_vllm.sh` and `docs/VLLM_SETUP.md` (setup notes, versions, and the final tuned flags) and push.

### Step 1b (optional): Run vLLM on a remote desktop over an SSH tunnel
The local setup from Step 1 stays the **default and fallback**. This step adds a remote desktop with a stronger GPU as the preferred host.
- **Remote host requirements:** Linux, or Windows with WSL2; an NVIDIA GPU; an OpenSSH server reachable from this laptop (LAN, VPN or Tailscale);
  **key-based SSH login** (no passwords stored anywhere).
- **Action:**
  1. On the remote host, repeat Step 1 items 3–6 (venv, vLLM, model, `start_vllm.sh`). vLLM binds to **`127.0.0.1:8000` only**, never `0.0.0.0`,
     so it can't be reached from the network except through SSH. Optionally add `--api-key` as a second layer.
  2. On the laptop, add `config/inference.local.env`, which is gitignored and holds the connection settings:
     `REMOTE_SSH_HOST`, `REMOTE_SSH_USER`, `REMOTE_SSH_PORT` (default 22), `REMOTE_VLLM_PORT` (default 8000),
     `TUNNEL_LOCAL_PORT` (default **8001**), `LOCAL_VLLM_PORT` (default **8000**). Commit a placeholder `config/inference.env.example`.
     The tunnel and the local server get separate ports so both can exist at once without clashing.
  3. Add `scripts/connect_remote.sh` (with a `.ps1` twin for Windows). It opens the tunnel
     (`ssh -N -L 8001:127.0.0.1:8000 user@host`, with `-o ExitOnForwardFailure=yes -o ServerAliveInterval=30`),
     waits for `http://localhost:8001/v1/models` to answer, and prints which endpoint is active.
     If SSH or the health check fails within a timeout, it says so and tells you to start the local server instead (`start_vllm.sh`, port 8000).
  4. Endpoint selection in code is covered in Step 5. It's the only code that depends on this step.
- **Validate:**
  - With the tunnel up: `curl localhost:8001/v1/models` lists the remote model, and one real-sized request succeeds. Record its latency next to the local one.
  - With the tunnel down: the script reports the fallback cleanly and `localhost:8000` (local vLLM) still works.
- **Checkpoint:** you commit `scripts/connect_remote.*`, `config/inference.env.example` and the docs section, then push.
  Never commit `inference.local.env` or SSH keys.

### Step 2: Set up the project's Python environment
- **Action:** In WSL, create a separate **project venv** (`.venv`, Python 3.12; the pins `numpy 2.4` / `scikit-learn 1.8` need ≥3.11) and install `src/requirements.txt`.
  It stays separate from the vLLM venv so the two sets of torch/version pins can't conflict. Add `.venv/` to `.gitignore`.
- **No code changes.**
- **Validate:** `python -c "import pandas, openai, pydantic"` works; the notebook kernel starts; `01_comparision_and_statistics.ipynb` runs top to bottom on the publication data **unmodified**.
- **Checkpoint:** you commit the `.gitignore` tweak (and any install notes) and push.

### Step 3: Capture a baseline (golden reference) before any code edits
- **Action:** Add `tests/` with a small pytest suite that pins the current behaviour of the functions that must not change.
  It needs no LLM and no network:
  - `generate_instructions_prompts` (0 and 10 definitions, German and English): the exact prompt strings are hashed and stored.
  - `get_schema_instruction`: the exact schema instruction text and custom schema.
  - `json_to_dataframe` → `merge_partial_rows` → `finalize_merged_row` on fixed sample JSON responses (including wrapped/unwrapped lists and missing keys): expected DataFrames stored as CSV fixtures.
  - Copy the publication output tables produced in Step 2 (`outputs/tables/*`) to `tests/golden/` so they can be compared later.
- **Validate:** `pytest` passes on the **unmodified** code.
- **Checkpoint:** you commit `tests/` and push. Every later step must keep this suite green.

### Step 4: Decouple from API keys and cloud SDKs
- **Action** (as decided in D2):
  - Remove `load_dotenv("../api_keys/.env")` from `helper_inference.py` (and from `time_one_call.py` in Step 7).
  - Move each cloud SDK import into its own branch in `initialize_clients` / `call_api`, so importing the module needs only `openai`.
  - Set `model_config.py` to `models_to_use = [("vllm", "<served-model-name>")]`.
  - Trim `requirements.txt` into a minimal runtime set plus an optional `requirements-cloud.txt`, or leave it as-is (your choice at the checkpoint).
- **No change** to prompts, parsing, merging, retries, file naming, or output columns.
- **Validate:** `pytest` stays green; `python -c "from utils import helper_inference"` works in a venv **without** the cloud SDKs; `api_keys/` is no longer read anywhere (`grep load_dotenv` returns nothing).
- **Checkpoint:** you commit and push.

### Step 5: Fix the vLLM call itself
- **Action** (only inside the `vllm` branch of `call_api` and the CLI):
  - Replace `extra_body={"guided_json": ...}` with `response_format={"type":"json_schema","json_schema":{"name":"psychopathologies","schema": json_schema}}`. It's the OpenAI-standard form and stable across vLLM releases.
  - Keep the system message exactly `chunk_prompt + schema_instruction` and the user message exactly `transcript`, the same as today.
  - Make `max_tokens` configurable (`--vllm_max_tokens`, default 8192). If prompt + max_tokens would exceed the server's context, clamp it with a clear warning rather than failing.
    The limit is read from `/v1/models` (`max_model_len`).
  - **Endpoint selection** (Step 1b), checked once at startup:
    1. if `--vllm_url` or `VLLM_BASE_URL` is set explicitly, use only that;
    2. otherwise, if the remote tunnel answers on `localhost:<TUNNEL_LOCAL_PORT>` (8001), use it;
    3. otherwise, fall back to local vLLM on `localhost:<LOCAL_VLLM_PORT>` (8000).

    The chosen endpoint and served model are printed and written to the run log, so every CSV can be traced to the host that produced it.
  - Add a preflight check at startup: if no endpoint answers, or the configured model isn't being served, fail fast with a readable message.
  - Optional `--seed`, passed through for reproducibility.
  - If a Qwen3-style reasoning model was chosen in D1, pass `chat_template_kwargs={"enable_thinking": False}` so the output contains only the JSON.
  - Keep the existing markdown-fence stripping and the retry logic unchanged.
- **Validate:** `pytest` green; one live call on the example transcript, chunk 1, which `json.loads` parses and `json_to_dataframe` accepts with all 10 numbers filled.
- **Checkpoint:** you commit and push.

### Step 6: End-to-end smoke run
- **Action:** No code changes; this step runs the pipeline as-is.
  From `src/`: `python 00_call_models.py --outer_runs 1 --number_of_definitions 10 --prompt_language german --transcript_dir ../data/02_transcripts_example/`.
  Then repeat with `--number_of_definitions 0`, the basic prompt that rates all 100 items in one response; this tests the output-length ceiling.
- **Validate:** the CSV appears in `outputs/AI_ratings/` with the same column set and order as the reference schema; `src/parsing_errors/` is empty; the number of NaNs is reported.
  Record time per chunk and per run.
- **Checkpoint:** you review a sample of the raw responses for sanity (German reasoning, plausible severities), then commit any doc notes and push.
  Generated CSVs stay gitignored unless you want one committed as an example.

### Step 7: Point the appendix timing script at the local model
- **Action:** In `appendix_code/time_one_call.py`, replace the hard-coded cloud `ALL_MODELS` list with the model(s) from `model_config.py` and drop `load_dotenv`.
  Token counting (tiktoken approximation) and the extrapolation maths stay unchanged.
  Leave `calc_costs.py` / `estimate_tokens.py` alone: they report the paper's cloud costs, and local inference costs nothing.
- **Validate:** the script runs against the vLLM server and writes `outputs/tables/inference_timing.csv`.
- **Checkpoint:** you commit and push.

### Step 8: Connect the new ratings to the analysis (configuration only)
- **Action:** Add `site_config_colors.py` entries for the new site keys (e.g. `vllm_0`, `vllm_10`, or whatever key the notebook derives) so the local model appears in the plots.
  Add a single clearly marked cell or variable in the notebook that selects the new CSV. The analysis code itself is untouched.
- **Validate:** with the publication data, the notebook still reproduces `tests/golden/` tables exactly. With the new CSV, it runs to the end and includes the local model.
- **Checkpoint:** you commit and push.

### Step 9: Documentation cleanup
- **Action:** Rewrite the README setup section for the vLLM workflow (WSL → start a local server **or** connect the remote tunnel → run pipeline → notebook). Remove or replace `api_keys/.your_env`.
  Record the exact vLLM version, model, quantization and server flags, since a reader can't reproduce results without them.
- **Validate:** follow the README from a fresh WSL shell; it works as written.
- **Checkpoint:** you commit and push. The refactor is complete.

### Step 10 (optional): Full production run
- All 3 transcripts, `--outer_runs` matching the paper's majority-voting setup (10), with and without definitions, in both languages if wanted.
  Expect several hours on a laptop GPU; it runs in the background with progress logs.

---

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| WSL/CUDA setup problems (virtualisation off in BIOS, driver mismatch) | Medium | Step 1 isolates this; the Windows driver is already recent (CUDA 13.4) |
| 8 GB VRAM can't hold ~25k-token context plus model | Medium | AWQ 4-bit, fp8 KV cache, single sequence; fall back to `max-model-len` 24k |
| `--number_of_definitions 0` (100 items in one reply) runs past `max_tokens` | Medium-High | Measured in Step 6; clamp/raise `max_tokens` within context; the existing code stops the model run on a parse error |
| A small model produces lower-quality or invalid JSON → parse failure stops the run (existing behaviour) | Medium | Structured outputs guarantee valid JSON; behaviour unchanged by design |
| Results differ from the paper | Certain | Expected: a different model, not a changed calculation. Golden tests prove the maths is identical |
| Temperature 0 is not bit-exact across vLLM versions/batching | Low-Medium | `--seed`, single-sequence serving, pinned versions recorded |
| Laptop thermal throttling / sleep during long runs | Medium | Plugged in, sleep disabled during Step 10 |
| `*.png` gitignore drops tracked images on import | Certain if ignored | Handled in Step 0 |
| SSH tunnel drops mid-run | Medium | Existing retry (connection errors are retryable, up to 5 attempts); `ServerAliveInterval`; rerun the script to reconnect |
| Remote vLLM exposed to the network | Low | Bind to `127.0.0.1` only; reachable solely through SSH; key auth; optional `--api-key` |
| Remote and local runs use different models, so CSVs get mixed up | Medium | Distinct `--served-model-name` per host; endpoint and model logged per run |

## Acceptance

- [ ] Pipeline runs end-to-end with only a local vLLM server and no API keys
- [ ] `tests/` golden suite green after every step (proves calculations are unchanged)
- [ ] Notebook reproduces the publication tables exactly
- [ ] Setup is reproducible from the README and `scripts/start_vllm.sh`
- [ ] (Optional) Remote tunnel is used when it's available and the pipeline falls back to local `localhost:8000` when it isn't
- [ ] Only program-essential files are tracked; Claude working files, envs, caches, outputs and weights are gitignored
