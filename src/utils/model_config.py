models_to_use = [
    # ── Local vLLM models (served on the remote desktop, reached via SSH tunnel) ──
    # model_name must match the id returned by the server's /v1/models endpoint.
    ("vllm", "/vllm/models/gemma-4-31B-it"),

    # ── Local test server (scripts/start_vllm.sh, run with --vllm_url http://localhost:8000/v1) ──
    #("vllm", "Qwen/Qwen2.5-7B-Instruct-AWQ"),
]
