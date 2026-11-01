"""Offline tests for src/utils/helper_inference.py (no vLLM server, no network)."""
import json
import os
from types import SimpleNamespace

import openai
import pandas as pd
import pytest

try:  # openai >= 3 builds its exceptions on httpx2
    import httpx2 as httpx
except ImportError:  # older openai versions use httpx
    import httpx

from utils import helper_inference as hi

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_DIR = os.path.join(REPO_ROOT, "data", "01_prompts")
MODEL = "test-model"

# ── Fakes ──

REQUEST = httpx.Request("POST", "http://localhost:11436/v1/chat/completions")


def timeout_error():
    return openai.APITimeoutError(request=REQUEST)


def bad_request_error(message="response_format json_schema not supported"):
    return openai.BadRequestError(message, response=httpx.Response(400, request=REQUEST), body=None)


def make_response(content, finish_reason="stop", reasoning_content=None):
    message = SimpleNamespace(content=content, reasoning_content=reasoning_content)
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish_reason, message=message)])


def valid_response(kwargs):
    """Answer with one entry per number the request's json_schema allows (prefixItems consts)."""
    schema = kwargs["response_format"]["json_schema"]["schema"]
    numbers = [item["properties"]["Number"]["const"]
               for item in schema["properties"]["psychopathologies"]["prefixItems"]]
    data = {"psychopathologies": [
        {"Number": n, "Trait": f"trait {n}", "Severness": n % 4, "Reasoning": f"reason {n}"}
        for n in numbers
    ]}
    return make_response(json.dumps(data))


class FakeCompletions:
    """chat.completions stand-in: each call takes the next scripted item.
    An item is an exception (raised), a callable (called with the request kwargs) or a response."""

    def __init__(self, script=None, default=None):
        self.script = list(script or [])
        self.default = default
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.script:
            item = self.script.pop(0)
        elif self.default is not None:
            item = self.default
        else:
            raise AssertionError("FakeCompletions called more often than scripted")
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            return item(kwargs)
        return item


class FakeClient:
    def __init__(self, completions, served=(MODEL,)):
        self.chat = SimpleNamespace(completions=completions)
        self.models = SimpleNamespace(list=lambda: [SimpleNamespace(id=m) for m in served])


def make_clients(completions, seed=None, max_tokens=123):
    return {"vllm": FakeClient(completions), "vllm_options": {"max_tokens": max_tokens, "seed": seed}}


# ── Fixtures ──

@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(hi.time, "sleep", lambda s: sleeps.append(s))
    return sleeps


@pytest.fixture
def schema():
    schema_instruction, custom_schema, Psychopathologies = hi.get_schema_instruction("english")
    return schema_instruction, custom_schema, Psychopathologies


@pytest.fixture
def reference_df():
    return pd.DataFrame({"ID_Video": [1], "p1_a_final": [0], "p2_b_final": [1],
                         "p3_c_final": [2], "p4_d_final": [3]})


CHUNKS = [
    {"prompt": "PROMPT ONE", "numbers": [1, 2], "names": ["a", "b"]},
    {"prompt": "PROMPT TWO", "numbers": [3, 4], "names": ["c", "d"]},
]


@pytest.fixture
def transcripts(tmp_path):
    out = {}
    for vid, vtype in [(1, "First"), (2, "Second")]:
        path = tmp_path / f"{vid:02d}_{vtype}_english.txt"
        path.write_text(f"Transcript of video {vid}", encoding="utf-8")
        out[f"{vid:02d}_{vtype}"] = {"path": str(path), "video_id": vid, "video_type": vtype}
    return out


def call_chunk(clients, schema, run=1):
    schema_instruction, custom_schema, Psychopathologies = schema
    return hi.process_chunk_with_retry("vllm", MODEL, "PROMPT", "TRANSCRIPT", 0.0, run, "01_First", 1,
                                       clients, schema_instruction, Psychopathologies, custom_schema,
                                       allowed_numbers=[1, 2])


def run_model(clients, schema, transcripts, reference_df, tmp_path, outer_runs=1, **kwargs):
    schema_instruction, custom_schema, Psychopathologies = schema
    return hi.process_single_model("vllm", MODEL, transcripts, CHUNKS, outer_runs, 0.0, 2, clients,
                                   schema_instruction, Psychopathologies, custom_schema, reference_df,
                                   raw_folder=str(tmp_path / "raw"), error_folder=str(tmp_path / "errors"),
                                   **kwargs)


def run_job(clients, schema, transcripts, reference_df, rating_folder, outer_runs=1, resume_from=None):
    schema_instruction, custom_schema, Psychopathologies = schema
    os.makedirs(rating_folder, exist_ok=True)
    return hi.run_psychopathology_rating(
        models=[("vllm", MODEL)], transkripts=transcripts, instruction_prompts_list=CHUNKS,
        outer_runs=outer_runs, temperature=0.0, number_of_definitions_in_one_run=2, clients=clients,
        schema_instruction=schema_instruction, Psychopathologies=Psychopathologies,
        psychopathologies_schema=custom_schema, reference_df=reference_df,
        rating_folder=str(rating_folder), resume_from=resume_from)


# ── Retry classification ──

def test_retry_on_timeout_then_success(schema, no_sleep):
    completions = FakeCompletions([timeout_error(), valid_response])
    text = call_chunk(make_clients(completions), schema)
    assert json.loads(text)["psychopathologies"][0]["Number"] == 1
    assert len(completions.calls) == 2
    assert no_sleep == [20]


def test_no_retry_on_bad_request(schema, no_sleep):
    completions = FakeCompletions([bad_request_error(), valid_response])
    with pytest.raises(openai.BadRequestError, match="not supported"):
        call_chunk(make_clients(completions), schema)
    assert len(completions.calls) == 1
    assert no_sleep == []


def test_retryable_classification():
    server_error = openai.InternalServerError("boom", response=httpx.Response(500, request=REQUEST), body=None)
    rate_limit = openai.RateLimitError("slow down", response=httpx.Response(429, request=REQUEST), body=None)
    assert hi.is_retryable_error(timeout_error())
    assert hi.is_retryable_error(openai.APIConnectionError(request=REQUEST))
    assert hi.is_retryable_error(server_error)
    assert hi.is_retryable_error(rate_limit)
    assert hi.is_retryable_error(hi.TruncatedResponseError("x"))
    assert hi.is_retryable_error(hi.EmptyResponseError("x"))
    assert not hi.is_retryable_error(bad_request_error())
    assert hi.is_retryable_error(RuntimeError("503 Service Unavailable"))  # substring fallback
    assert not hi.is_retryable_error(ValueError("Expecting value: line 1 column 1 (char 0)"))


# ── Truncation and empty content ──

def test_truncated_response_raises(schema, no_sleep):
    completions = FakeCompletions(default=make_response('{"psychopathologies": [', finish_reason="length"))
    with pytest.raises(hi.TruncatedResponseError, match=r"truncated.*max_tokens=123"):
        call_chunk(make_clients(completions, max_tokens=123), schema)
    assert len(completions.calls) == 5  # retryable, gives up after max_retries


def test_reasoning_content_without_content_raises(schema):
    completions = FakeCompletions(default=make_response(None, reasoning_content='{"psychopathologies": []}'))
    with pytest.raises(hi.EmptyResponseError, match="reasoning_content but no content"):
        call_chunk(make_clients(completions), schema)


def test_messages_unchanged(schema):
    """System message is exactly chunk_prompt + schema_instruction, user message exactly the transcript."""
    schema_instruction = schema[0]
    completions = FakeCompletions([valid_response])
    call_chunk(make_clients(completions), schema)
    assert completions.calls[0]["messages"] == [
        {"role": "system", "content": "PROMPT" + schema_instruction},
        {"role": "user", "content": "TRANSCRIPT"},
    ]


# ── Failure isolation ──

def test_failed_chunk_does_not_stop_next_transcript(schema, transcripts, reference_df, tmp_path):
    # transcript 1, run 1, chunk 1 returns invalid JSON; everything after succeeds
    completions = FakeCompletions([make_response("not json")], default=valid_response)
    results, failed = run_model(make_clients(completions), schema, transcripts, reference_df, tmp_path)

    assert len(results) == 1
    assert list(results[0]["txt_name"]) == ["02_Second"]
    assert results[0].loc[0, "p3_c_final"] == 3
    assert [(t, r, c) for t, r, c, _ in failed] == [("01_First", 1, 1)]
    # transcript 1 stops after its failed chunk; transcript 2 makes both chunk calls
    assert len(completions.calls) == 3
    err_file = tmp_path / "errors" / f"error_answer_vllm_{MODEL}_run1_01_First_chunk1.txt"
    assert err_file.read_text(encoding="utf-8") == "not json"


def test_failed_run_does_not_stop_next_run(schema, transcripts, reference_df, tmp_path):
    completions = FakeCompletions([bad_request_error()], default=valid_response)
    results, failed = run_model(make_clients(completions), schema, transcripts, reference_df, tmp_path,
                                outer_runs=2)
    assert [list(df["run"]) for df in results] == [[2], [1, 2]]
    assert len(failed) == 1


def test_no_results_writes_nothing(schema, transcripts, reference_df, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    completions = FakeCompletions(default=make_response("not json"))
    out_dir = tmp_path / "ratings"
    df = run_job(make_clients(completions), schema, transcripts, reference_df, out_dir)
    assert df.empty
    assert os.listdir(out_dir) == []
    assert len(os.listdir(tmp_path / "parsing_errors")) == 1  # one job folder


# ── Seed ──

def test_seed_offset_per_run(schema, transcripts, reference_df, tmp_path):
    completions = FakeCompletions(default=valid_response)
    run_model(make_clients(completions, seed=42), schema, {"01_First": transcripts["01_First"]},
              reference_df, tmp_path, outer_runs=2)
    assert [c["seed"] for c in completions.calls] == [43, 43, 44, 44]


def test_no_seed_sent_when_unset(schema):
    completions = FakeCompletions([valid_response])
    call_chunk(make_clients(completions, seed=None), schema)
    assert "seed" not in completions.calls[0]


# ── Job folders, incremental save, resume ──

def test_job_scoped_folders_and_resume(schema, transcripts, reference_df, tmp_path, monkeypatch):
    first_dir = tmp_path / "first"
    first_dir.mkdir()
    monkeypatch.chdir(first_dir)
    completions = FakeCompletions(default=valid_response)
    df_first = run_job(make_clients(completions), schema, transcripts, reference_df,
                       first_dir / "ratings", outer_runs=2)
    assert len(completions.calls) == 8

    job_folders = os.listdir(first_dir / "raw_responses")
    assert len(job_folders) == 1 and job_folders[0].endswith("_defs2_temp0.0")
    resume_folder = first_dir / "raw_responses" / job_folders[0]
    assert len(os.listdir(resume_folder)) == 8
    (first_csv,) = os.listdir(first_dir / "ratings")

    # Second job: every chunk comes from the resume folder, the client is never called
    second_dir = tmp_path / "second"
    second_dir.mkdir()
    monkeypatch.chdir(second_dir)
    never_called = FakeCompletions()
    df_second = run_job(make_clients(never_called), schema, transcripts, reference_df,
                        second_dir / "ratings", outer_runs=2, resume_from=str(resume_folder))
    assert never_called.calls == []
    pd.testing.assert_frame_equal(df_first, df_second)
    (second_csv,) = os.listdir(second_dir / "ratings")
    assert ((first_dir / "ratings" / first_csv).read_bytes()
            == (second_dir / "ratings" / second_csv).read_bytes())


def test_partial_resume_calls_only_missing_chunks(schema, transcripts, reference_df, tmp_path):
    resume = tmp_path / "old_job"
    resume.mkdir()
    saved = valid_response({"response_format": {"json_schema": {"schema": hi.build_chunk_response_schema(
        schema[2], [1, 2])}}}).choices[0].message.content
    hi.save_raw_response(saved, str(resume), "vllm", MODEL, 1, "01_First", 1)
    completions = FakeCompletions(default=valid_response)
    results, failed = run_model(make_clients(completions), schema, {"01_First": transcripts["01_First"]},
                                reference_df, tmp_path, resume_folder=str(resume))
    assert failed == []
    assert len(completions.calls) == 1  # only chunk 2 was requested
    assert completions.calls[0]["messages"][0]["content"].startswith("PROMPT TWO")


def test_incremental_save_survives_crash(schema, transcripts, reference_df, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # transcript 1 completes (2 chunks), then the process is interrupted during transcript 2
    completions = FakeCompletions([valid_response, valid_response, KeyboardInterrupt()])
    out_dir = tmp_path / "ratings"
    with pytest.raises(KeyboardInterrupt):
        run_job(make_clients(completions), schema, transcripts, reference_df, out_dir)
    (csv_name,) = os.listdir(out_dir)
    saved = pd.read_csv(out_dir / csv_name)
    assert list(saved["txt_name"]) == ["01_First"]


# ── Preflight probe ──

def test_probe_rejected_schema_fails_fast(monkeypatch):
    monkeypatch.delenv("VLLM_SKIP_PROBE", raising=False)
    completions = FakeCompletions([bad_request_error("prefixItems is not supported")])
    with pytest.raises(RuntimeError, match="does not support the chunk schema"):
        hi.check_vllm_server(FakeClient(completions), "http://x/v1", [MODEL])
    schema = completions.calls[0]["response_format"]["json_schema"]["schema"]
    assert "prefixItems" in schema["properties"]["psychopathologies"]


def test_probe_skippable(monkeypatch):
    monkeypatch.setenv("VLLM_SKIP_PROBE", "1")
    completions = FakeCompletions()
    hi.check_vllm_server(FakeClient(completions), "http://x/v1", [MODEL])
    assert completions.calls == []


# ── Definitions parsing ──

@pytest.mark.parametrize("language", ["english", "german"])
def test_parse_definitions_yields_100_items(language):
    with open(os.path.join(PROMPT_DIR, f"All_Items_{language}.txt"), encoding="utf-8") as f:
        items = hi.parse_definitions(f.read().strip())
    assert [it["number"] for it in items] == list(range(1, 101))


def test_parse_definitions_ignores_inline_hash():
    text = "1. Foo\nB#eispiele: x\n####\n2. Bar\ntext\n#### \n3. Baz"
    items = hi.parse_definitions(text)
    assert [it["number"] for it in items] == [1, 2, 3]
    assert items[0]["full_text"] == "1. Foo\nB#eispiele: x"
