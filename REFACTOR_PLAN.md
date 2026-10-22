# Refactor Plan: Local vLLM Inference for LLMs_psychopathologies

**Source:** [`42elenz/LLMs_psychopathologies`](https://github.com/42elenz/LLMs_psychopathologies) @ `c89a3ea`
**Target repo:** `matthewsjzhang/llm-benchmarking-lenz-refactor`
**Complexity:** Medium. The code changes are small; most of the risk is in connectivity and model behaviour.

## Goal

Run the rating pipeline against a self-hosted LLM served by vLLM instead of cloud APIs with API keys.
**Every algorithm and calculation stays unchanged:** prompt construction, definition chunking,
JSON parsing, chunk merging, CSV layout, and all analysis in the notebook, `helper.py` and R.
Only the code that sends the prompt to a model and gets text back is touched.

## Decisions (confirmed 2026-10-22)

| # | Decision | Consequence for the plan |
|---|---|---|
| D1 | Model: **`gemma-4-31B-it`**, already served by vLLM on the remote desktop as `/vllm/models/gemma-4-31B-it` (`max_model_len` 262144) | Context size is not a constraint. The served model id is a filesystem path, so the code needs a clean display label for the CSVs (Step 3). |
| D2 | **Delete all cloud-provider branches.** Only local/self-hosted vLLM models can run. | OpenAI-cloud, Gemini, Anthropic, Mistral and Together code, imports, keys and SDK dependencies are removed (Step 2). |
| D3 | Client environment: **deferred**. Nothing is run against the LLM yet. | Steps that need Python or a live model are grouped under "Deferred" at the end. |
| D4 | **Bring the upstream code into this repo and modify it in place**, with a README credit to the original repo. | Step 0 imports upstream verbatim as its own commit, which doubles as the unmodified baseline for later equivalence tests. |
| D5 | **vLLM runs only on the remote desktop, reached through an SSH tunnel.** Nothing runs locally. | No local vLLM, WSL or GPU setup is needed. Step 1 sets up the tunnel; the pipeline talks to `http://localhost:<port>/v1`. |

## What the code does today (grounding)

| Area | Location | Notes |
|---|---|---|
| Entry point | `src/00_call_models.py` | CLI args; already has `--vllm_url`, which it exports as `VLLM_BASE_URL` |
| Model selection | `src/utils/model_config.py` | list of `(api_family, model_name)` tuples |
| Client setup | `src/utils/helper_inference.py:109` `initialize_clients` | builds a vLLM `OpenAI(base_url=..., api_key="EMPTY")` client |
| LLM call | `src/utils/helper_inference.py:624` `call_api` (vllm branch) | `chat.completions.create`; system = `chunk_prompt + schema_instruction`; `max_tokens=8192`; `extra_body={"guided_json": schema}` |
| Retry | `helper_inference.py:653` | 5 attempts, 20s exponential backoff; "connection refused/error" and "timeout" count as retryable |
| Key loading | `helper_inference.py:22`, `appendix_code/time_one_call.py:14` | `load_dotenv("../api_keys/.env")` |
| Cloud SDK imports | `helper_inference.py:11-18` | google-genai, together, anthropic, mistralai, openai |
| Analysis keying | `helper.py:343` | each AI rating set is looked up by its `site_key` in `site_config_colors.py`; a key that isn't listed there won't be plotted |
| Paths | everywhere | relative (`../data`, `./raw_responses`), so scripts **must run from `src/`** |

### Problems found

1. **`guided_json` was removed in vLLM v0.12.0.** The existing vLLM branch fails on current vLLM.
   The replacement is `response_format={"type":"json_schema", ...}` or `extra_body={"structured_outputs":{"json": ...}}`.
2. **The served model id is a path** (`/vllm/models/gemma-4-31B-it`). The code writes `model_name.replace('/', '_')`
   into CSVs and file names, which would give `_mnt_DataDrive_vllm_models_gemma-4-31B-it`.
3. **Gemma chat templates and system prompts:** Gemma-family templates have historically folded the system message into the first user turn.
   The prompt text stays identical, but the role placement may differ from the paper's models. This needs checking at the first live call.
4. **One upstream file is named `src/__init__.py ` (trailing space)**, which Windows can't create. It's recreated as `src/__init__.py`.
5. **Upstream `.gitignore` ignores `*.png`, `outputs/figs/` and `outputs/tables/`** even though upstream tracks files there. Copying those rules would silently drop them.
6. **Upstream has no LICENSE file.** Redistribution rights are unclear; it's worth asking the authors, or keeping this repo private, before making it public.

## Working agreement for every step

- One step = one task. I finish it, run its **Validate** checks, and then **stop at a checkpoint** with a summary and suggested commit message.
- Commits use your account only, with no Claude attribution. You review and push, or tell me to push.
- The next step starts only when you say go.

---

## Steps

### Step 0: Import upstream code (verbatim baseline) (done, `a52ef43`)
- **Action:** Copy everything from the temporary clone `LLMs_psychopathologies/` (except `.git/`) into the repo root. Recreate `src/__init__.py` (empty).
  Merge upstream's `.gitignore` into the root one **without** the `*.png`, `outputs/figs/` and `outputs/tables/` rules, so the publication figures and tables stay tracked.
  Add a "Source" credit at the top of `README.md`. Remove the temporary clone once the import is verified.
- **No code changes.** This commit is the reference point for "calculations unchanged".
- **Validate:** every imported file's git blob hash equals upstream's (`git ls-files -s` vs upstream `git ls-tree -r HEAD`), apart from the README credit.
- **Checkpoint:** commit "Import upstream LLMs_psychopathologies @ c89a3ea" and push.

### Step 1: Connect to the remote vLLM server via SSH tunnel
- **Needed from you:** the remote desktop's address (LAN IP, VPN or Tailscale name), SSH username and port,
  the port vLLM listens on there (11436, per the command you were given), and whether SSH login already works with a password.
- **Action:**
  1. On this laptop (Windows has the OpenSSH client built in): generate a key (`ssh-keygen -t ed25519`) and install it on the remote host's `~/.ssh/authorized_keys`.
     **You run the key-install step**, because it needs your remote password once.
  2. Add a `Host vllm-remote` entry to `~/.ssh/config` (outside the repo) with `HostName`, `User`, `Port`, `IdentityFile`,
     `LocalForward 11436 localhost:11436`, `ServerAliveInterval 30` and `ExitOnForwardFailure yes`.
  3. Add `scripts/connect_remote.ps1` (and a `.sh` twin) to the repo. It runs `ssh -N vllm-remote`, waits until `http://localhost:11436/v1/models` answers,
     and prints the served model, or a clear error naming which part failed (SSH auth, port forward, or vLLM not running).
     It reads host settings only from `~/.ssh/config`, so no addresses or usernames are committed.
  4. Check how vLLM is bound on the remote host. If it listens on `0.0.0.0`, recommend restarting it on `127.0.0.1` (and/or adding `--api-key`) so it's reachable only through SSH.
     Optionally restart it with `--served-model-name gemma-4-31B-it` for a clean model id.
- **Validate:** with the tunnel up, `curl http://localhost:11436/v1/models` on the laptop returns the Gemma model. One tiny chat request returns text.
  With the tunnel down, the script reports the failure cleanly.
- **Checkpoint:** commit `scripts/connect_remote.*` and the setup notes in `docs/REMOTE_VLLM.md`. Never commit SSH keys or host details.

### Step 2: Remove cloud providers (D2) (done, `52f23fe`)
- **Action:**
  - `helper_inference.py`: delete the `openai`, `gemini`, `anthropic`, `mistral` and `open_source` branches from `initialize_clients`, `call_api` and `get_api_provider`.
    Delete `get_model_max_tokens` (Together-only), the cloud SDK imports and `load_dotenv`. Keep the `openai` package: it is the client library for vLLM's OpenAI-compatible API, not the cloud service.
  - `model_config.py`: only `("vllm", "<served model id>")` entries.
  - `appendix_code/time_one_call.py`: replace the cloud `ALL_MODELS` list with `models_to_use`; drop `load_dotenv`. Timing and token maths unchanged.
  - Delete `api_keys/`. Remove `anthropic`, `google-genai`, `google-auth`, `mistralai` and `together` (plus deps that only they need) from `requirements.txt`.
  - **Untouched:** `estimate_tokens.py` and `calc_costs.py`, which compute the paper's token and cost figures without calling any LLM, and `site_config_colors.py` cloud entries,
    which the notebook needs to reproduce the publication analysis.
- **Validate:** `grep` for `anthropic|genai|mistral|together|load_dotenv|api_keys` in `src/` finds no live code.
  A structural diff against the Step 0 commit shows that only the LLM-call layer, config, requirements and `api_keys/` changed.
- **Checkpoint:** commit and push.

### Step 3: Fix and harden the vLLM call (done, `9e9ec79`)
- **Action** (only inside the `vllm` branch of `call_api`, `initialize_clients` and the CLI):
  - Replace `extra_body={"guided_json": ...}` with `response_format={"type":"json_schema","json_schema":{"name":"psychopathologies","schema": json_schema}}`.
  - Keep the system message exactly `chunk_prompt + schema_instruction` and the user message exactly `transcript`.
  - Endpoint: `--vllm_url` / `VLLM_BASE_URL`, default `http://localhost:11436/v1` (the tunnel). Optional `VLLM_API_KEY`, read only from the environment and never stored in the repo.
  - Preflight at startup: call `/v1/models`. If the tunnel or server is down, or the configured model isn't served, fail fast with "is the SSH tunnel up? run scripts/connect_remote".
  - **Model label:** a `display_name` per model in `model_config.py` (e.g. `gemma-4-31B-it`), used for CSV `model_name` and file names instead of the raw path.
    This is metadata only and changes no values.
  - `--vllm_max_tokens` (default 8192, unchanged) and an optional `--seed`. Increase the client timeout for a 31B model over SSH.
  - Keep the markdown-fence stripping and the retry logic unchanged.
  - Error path: reset `rating_text` per chunk. Upstream raised `UnboundLocalError` when the first chunk's call failed, and logged the previous chunk's text otherwise.
- **Validate:** code review against the Step 0 baseline; the live check is in the Deferred section.
- **Checkpoint:** commit and push.

### Step 4: Connect the new ratings to the analysis (configuration only) (done)
- **Action:** add `site_config_colors.py` entries for the new site keys (`vllm_0`, `vllm_10`, or whatever key the notebook derives).
  Add one clearly marked notebook variable for the path of a new ratings CSV, defaulting to the publication data. The analysis code itself is untouched.
- **Validate:** diff shows only config and path additions.
- **Checkpoint:** commit and push.

### Step 5: Documentation
- **Action:** rewrite the README setup for the new workflow: connect the SSH tunnel, then run the pipeline, then the notebook.
  Remove all API-key instructions. Record the vLLM version, model, and server flags of the remote host. Keep the Source credit and the publication-results links.
- **Checkpoint:** commit and push. The code refactor is complete.

---

## Deferred (needs the client environment, D3)

These run once you decide where the Python client runs. Because Step 0 is a verbatim upstream commit, the baseline can still be captured after the code changes.

- **D-1: Client environment.** A Python 3.12 venv with `requirements.txt`; the notebook runs top to bottom on the publication data.
- **D-2: Equivalence tests.** A pytest suite run against **both** the Step 0 commit and HEAD, with no LLM or network needed.
  It checks that `generate_instructions_prompts`, `get_schema_instruction` and `json_to_dataframe` → `merge_partial_rows` → `finalize_merged_row` give identical output on fixed inputs,
  and that the notebook still reproduces `outputs/tables/*` exactly.
- **D-3: Live smoke run** through the tunnel on `data/02_transcripts_example/` with `--number_of_definitions 10`, then `0`.
  Check the CSV columns, that `parsing_errors/` stays empty, Gemma's handling of the system prompt (problem 3), and time per chunk.
- **D-4: Timing script** (`time_one_call.py`) against the remote model.
- **D-5: Full run:** all transcripts, `--outer_runs 10`, with and without definitions.

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Remote desktop not reachable from the laptop's network (NAT, firewall) | Medium | Same LAN, or a VPN/Tailscale; Step 1 diagnoses which part fails |
| SSH tunnel drops mid-run | Medium | Existing retry covers connection errors (5 attempts, backoff); `ServerAliveInterval`; rerun the connect script |
| Remote vLLM exposed to the network | Medium until checked | Bind to `127.0.0.1` and/or `--api-key`; reach it only through SSH (Step 1) |
| Gemma template merges the system prompt into the user turn | Medium | Prompt text unchanged; verified in D-3; documented as a methodological note |
| A parse failure stops a model's run (existing behaviour) | Low | Structured outputs guarantee schema-valid JSON |
| Results differ from the paper | Certain | Expected: a different model, not a changed calculation. D-2 proves the maths is identical |
| No upstream license | — | Keep the repo private or get the authors' permission before publishing |

## Acceptance

- [ ] Pipeline contains only vLLM inference; no API keys or cloud SDKs anywhere
- [ ] SSH tunnel script connects to the remote vLLM server and fails with a clear message when it can't
- [ ] Algorithms and calculations byte-identical to upstream (D-2)
- [ ] README credits the source repo and documents the tunnel workflow
- [ ] Only program-essential files are tracked; Claude working files, envs, caches, outputs and weights are gitignored
