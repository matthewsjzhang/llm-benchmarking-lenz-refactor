# Refactor plan: local vLLM inference for LLMs_psychopathologies

Source: [`42elenz/LLMs_psychopathologies`](https://github.com/42elenz/LLMs_psychopathologies) @ `c89a3ea`
Target repo: `matthewsjzhang/llm-benchmarking-lenz-refactor`
Complexity: medium. The code changes are small; most of the risk is in connectivity and model behaviour.

## Goal

The rating pipeline should run against a self-hosted LLM served by vLLM instead of cloud APIs with API keys.
Every algorithm and calculation stays the same: prompt construction, definition chunking, JSON parsing,
chunk merging, CSV layout, and all analysis in the notebook, `helper.py` and R. The only code that changes
is the part that sends a prompt to a model and gets text back.

## Decisions (2026-10-22)

| # | Decision | Effect on the plan |
|---|---|---|
| D1 | Model: `gemma-4-31B-it`, already served by vLLM on the remote desktop as `/vllm/models/gemma-4-31B-it` (`max_model_len` 262144) | Context size is not a constraint. The served model id is a filesystem path, so the CSVs need a short display label (Step 3). |
| D2 | Delete all cloud-provider branches. Only self-hosted vLLM models can run. | The OpenAI cloud, Gemini, Anthropic, Mistral and Together code, imports, keys and SDK dependencies go (Step 2). |
| D3 | Client environment deferred. Nothing runs against the LLM yet. | Steps that need Python or a live model are listed under "Deferred" at the end. |
| D4 | Import the upstream code into this repo and modify it in place, with a README credit to the original repo. | Step 0 imports upstream verbatim as its own commit, which also serves as the unmodified baseline for equivalence tests. |
| D5 | vLLM runs only on the remote desktop, reached through an SSH tunnel. | No local vLLM, WSL or GPU setup is needed. Step 1 sets up the tunnel, and the pipeline talks to `http://localhost:<port>/v1`. |

## Current code

| Area | Location | Notes |
|---|---|---|
| Entry point | `src/00_call_models.py` | CLI args; already has `--vllm_url`, which it exports as `VLLM_BASE_URL` |
| Model selection | `src/utils/model_config.py` | list of `(api_family, model_name)` tuples |
| Client setup | `src/utils/helper_inference.py:109` `initialize_clients` | builds a vLLM `OpenAI(base_url=..., api_key="EMPTY")` client |
| LLM call | `src/utils/helper_inference.py:624` `call_api` (vllm branch) | `chat.completions.create`; system = `chunk_prompt + schema_instruction`; `max_tokens=8192`; `extra_body={"guided_json": schema}` |
| Retry | `helper_inference.py:653` | 5 attempts, 20s exponential backoff; "connection refused/error" and "timeout" count as retryable |
| Key loading | `helper_inference.py:22`, `appendix_code/time_one_call.py:14` | `load_dotenv("../api_keys/.env")` |
| Cloud SDK imports | `helper_inference.py:11-18` | google-genai, together, anthropic, mistralai, openai |
| Analysis keying | `helper.py:343` | each AI rating set is looked up by its `site_key` in `site_config_colors.py`; a key missing there isn't plotted |
| Paths | everywhere | relative (`../data`, `./raw_responses`), so scripts must run from `src/` |

### Problems found

1. vLLM removed `guided_json` in v0.12.0, so the existing vLLM branch fails on current vLLM.
   The replacement is `response_format={"type":"json_schema", ...}` or `extra_body={"structured_outputs":{"json": ...}}`.
2. The served model id is a path (`/vllm/models/gemma-4-31B-it`). The code writes `model_name.replace('/', '_')`
   into CSVs and file names, which would give `_vllm_models_gemma-4-31B-it`.
3. Gemma-family chat templates have historically folded the system message into the first user turn.
   The prompt text stays identical, but the role placement may differ from the paper's models. Check this on the first live call.
4. One upstream file is named `src/__init__.py ` (trailing space), which Windows can't create. It is recreated as `src/__init__.py`.
5. Upstream's `.gitignore` ignores `*.png`, `outputs/figs/` and `outputs/tables/` even though upstream tracks files there. Copying those rules would silently drop them.
6. Upstream has no LICENSE file, so redistribution rights are unclear. Ask the authors, or keep this repo private, before making it public.
7. German item 23 is truncated in the upstream data. Line 567 of `All_Items_german.txt` had `B#eispiele`; the parser splits on `#`, so item 23 (Vorbeireden) was sent as a 217-character stub and about 1,500 characters were dropped. It is fixed here, so the German item 23 prompt differs from the paper's run.
8. `model_parse_failed` is never reset. Once one chunk fails, the flag stays set and every later transcript is skipped.
9. The retry filter misses `Request timed out.` and server 5xx errors, so a transient failure aborts the chunk instead of being retried.
10. The seed is fixed across outer runs. With `--vllm_seed` set, every run repeats the same samples, so runs don't differ at temperature > 0.

## Workflow

Each step is one task. It ends with its Validate checks, a summary and a proposed commit message, and the next
step starts only after the owner reviews it. Commits are made under the owner's account, and the owner pushes
or asks for the push.

---

## Steps

### Step 0: Import upstream code as a verbatim baseline (done, `a52ef43`)
- Action: copy everything from the temporary clone `LLMs_psychopathologies/` (except `.git/`) into the repo root and recreate an empty `src/__init__.py`.
  Merge upstream's `.gitignore` into the root one without the `*.png`, `outputs/figs/` and `outputs/tables/` rules, so the publication figures and tables stay tracked.
  Add a source credit at the top of `README.md`, and delete the temporary clone once the import is verified.
- No code changes. This commit is the reference point for checking that calculations are unchanged.
- Validate: every imported file's git blob hash equals upstream's (`git ls-files -s` against upstream `git ls-tree -r HEAD`), apart from the README credit.
- Checkpoint: commit "Import upstream LLMs_psychopathologies @ c89a3ea" and push.

### Step 1: Connect to the remote vLLM server through an SSH tunnel
- Inputs: the remote desktop's address (LAN IP, VPN or Tailscale name), SSH username and port,
  the port vLLM listens on there (11436, per the provided SSH command), and whether SSH login already works with a password.
- Action:
  1. On the client machine (Windows has the OpenSSH client built in), generate a key with `ssh-keygen -t ed25519` and install it in the remote host's `~/.ssh/authorized_keys`.
     The owner runs the install step, since it needs the remote password once.
  2. Add a `Host vllm-remote` entry to `~/.ssh/config` (outside the repo) with `HostName`, `User`, `Port`, `IdentityFile`,
     `LocalForward 11436 localhost:11436`, `ServerAliveInterval 30` and `ExitOnForwardFailure yes`.
  3. Add `scripts/connect_remote.ps1` and a `.sh` version. The script runs `ssh -N vllm-remote`, waits until `http://localhost:11436/v1/models` answers,
     and prints the served model. If something fails, it says whether SSH auth, the port forward, or vLLM itself is the problem.
     It reads host settings only from `~/.ssh/config`, so no addresses or usernames are committed.
  4. Check how vLLM is bound on the remote host. If it listens on `0.0.0.0`, restart it on `127.0.0.1` (and/or with `--api-key`) so it's reachable only through SSH.
     Optionally restart it with `--served-model-name gemma-4-31B-it` for a short model id.
- Validate: with the tunnel up, `curl http://localhost:11436/v1/models` on the client returns the Gemma model, and one small chat request returns text.
  With the tunnel down, the script reports the failure.
- Checkpoint: commit `scripts/connect_remote.*` and setup notes in `docs/REMOTE_VLLM.md`. SSH keys and host details are never committed.

### Step 2: Remove cloud providers (D2) (done, `52f23fe`)
- Action:
  - `helper_inference.py`: delete the `openai`, `gemini`, `anthropic`, `mistral` and `open_source` branches from `initialize_clients`, `call_api` and `get_api_provider`.
    Delete `get_model_max_tokens` (used only by Together), the cloud SDK imports and `load_dotenv`. The `openai` package stays because it is the client library for vLLM's OpenAI-compatible API.
  - `model_config.py`: only `("vllm", "<served model id>")` entries.
  - `appendix_code/time_one_call.py`: replace the cloud `ALL_MODELS` list with `models_to_use` and drop `load_dotenv`. The timing and token maths stay the same.
  - Delete `api_keys/`. Remove `anthropic`, `google-genai`, `google-auth`, `mistralai` and `together` (plus dependencies only they need) from `requirements.txt`.
  - Left as is: `estimate_tokens.py` and `calc_costs.py`, which compute the paper's token and cost figures without calling an LLM, and the cloud entries in `site_config_colors.py`,
    which the notebook needs to reproduce the publication analysis.
- Validate: `grep` for `anthropic|genai|mistral|together|load_dotenv|api_keys` in `src/` finds no live code.
  A structural diff against the Step 0 commit shows changes only in the LLM-call layer, config, requirements and `api_keys/`.
- Checkpoint: commit and push.

### Step 3: Fix and harden the vLLM call (done, `9e9ec79`)
- Action, limited to the `vllm` branch of `call_api`, `initialize_clients` and the CLI:
  - Replace `extra_body={"guided_json": ...}` with `response_format={"type":"json_schema","json_schema":{"name":"psychopathologies","schema": json_schema}}`.
  - Keep the system message exactly `chunk_prompt + schema_instruction` and the user message exactly `transcript`.
  - Endpoint: `--vllm_url` / `VLLM_BASE_URL`, default `http://localhost:11436/v1` (the tunnel). An optional `VLLM_API_KEY` is read only from the environment and never stored in the repo.
  - At startup, call `/v1/models`. If the tunnel or server is down, or the configured model isn't served, stop with "is the SSH tunnel up? run scripts/connect_remote".
  - Give each model in `model_config.py` a `display_name` (e.g. `gemma-4-31B-it`) and use it for the CSV `model_name` and file names instead of the raw path.
    This is metadata and changes no values.
  - Add `--vllm_max_tokens` (default 8192, as before) and an optional `--seed`, and raise the client timeout for a 31B model over SSH.
  - Keep the markdown-fence stripping and the retry logic as they are.
  - Reset `rating_text` for each chunk. Upstream raised `UnboundLocalError` when the first chunk's call failed, and otherwise logged the previous chunk's text.
- Validate: code review against the Step 0 baseline. The live check is in the Deferred section.
- Checkpoint: commit and push.

### Step 4: Connect the new ratings to the analysis (configuration only) (done, `01905c3`)
- Action: add `site_config_colors.py` entries for the new site keys (`vllm_0`, `vllm_10`, or whatever key the notebook derives).
  Add one clearly marked notebook variable for the path of a new ratings CSV, defaulting to the publication data. The analysis code stays as it is.
- Validate: the diff shows only config and path additions.
- Checkpoint: commit and push.

### Step 5: Documentation (done, except the remote server version and flags, which need the tunnel)
- Action: rewrite the README setup for the new workflow: connect the SSH tunnel, run the pipeline, then the notebook.
  Remove the API-key instructions. Record the remote host's vLLM version, model and server flags. Keep the source credit and the links to the publication results.
- Checkpoint: commit and push. This completes the code refactor.

---

## Deferred (needs the client environment, D3)

These run once the location of the Python client is decided. Since Step 0 is a verbatim upstream commit, the baseline can still be captured after the code changes.

- D-1, client environment: a Python 3.12 venv with `requirements.txt`, and the notebook runs top to bottom on the publication data.
- D-2, equivalence tests: a pytest suite run against both the Step 0 commit and HEAD, with no LLM or network.
  It checks that `generate_instructions_prompts`, `get_schema_instruction`, and the chain `json_to_dataframe`, `merge_partial_rows`, `finalize_merged_row` give identical output on fixed inputs,
  and that the notebook still reproduces `outputs/tables/*` exactly.
- D-3, live smoke run through the tunnel on `data/02_transcripts_example/` with `--number_of_definitions 10`, then `0`.
  Check the CSV columns, that `parsing_errors/` stays empty, how Gemma handles the system prompt (problem 3), and the time per chunk.
- D-4: run the timing script (`time_one_call.py`) against the remote model.
- D-5, full run: all transcripts, `--outer_runs 10`, with and without definitions.

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Remote desktop unreachable from the client's network (NAT, firewall) | Medium | Same LAN, or a VPN/Tailscale; Step 1 shows which part fails |
| SSH tunnel drops mid-run | Medium | Existing retry covers connection errors (5 attempts, backoff); `ServerAliveInterval`; rerun the connect script |
| Remote vLLM exposed to the network | Medium until checked | Bind to `127.0.0.1` and/or use `--api-key`; reach it only through SSH (Step 1) |
| Gemma template merges the system prompt into the user turn | Medium | Prompt text unchanged; checked in D-3; documented as a methodological note |
| A parse failure stops a model's run (existing behaviour) | Low | Structured outputs guarantee schema-valid JSON |
| Results differ from the paper | Certain | Expected, since the model is different and the calculations are not; D-2 confirms the maths is identical |
| No upstream license | Not rated | Keep the repo private or get the authors' permission before publishing |

## Acceptance

- [ ] The pipeline contains only vLLM inference, with no API keys or cloud SDKs anywhere
- [ ] The SSH tunnel script connects to the remote vLLM server and reports clearly when it can't
- [ ] Algorithms and calculations are byte-identical to upstream (D-2)
- [ ] The README credits the source repo and documents the tunnel workflow
- [ ] Only files the program needs are tracked; Claude working files, envs, caches, outputs and weights are gitignored
