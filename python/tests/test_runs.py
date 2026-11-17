import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from askkit.cli import main
from askkit.prompts import render
from askkit.runs import (
    Results,
    add_run,
    diff_lines,
    load_results,
    run_scores,
    verdict,
)

PYTHON = Path(__file__).parent.parent
FAULT = ["wish: 'refund', not 'replacement'"]


def results(model: str, version: int, **cases: list[list[str]]) -> Results:
    return Results(model=model, prompt_version=version, cases=cases)


def test_a_first_run_starts_the_file_and_a_second_adds_to_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "v2.json"
    add_run(path, "claude-haiku-5-5", 2, {"maria": [], "pans": FAULT})
    add_run(path, "claude-haiku-5-5", 2, {"maria": [], "pans": []})
    assert load_results(path) == results(
        "claude-haiku-5-5", 2, maria=[[], []], pans=[FAULT, []]
    )


def test_a_run_of_other_settings_is_refused_and_the_file_kept(
    tmp_path: Path,
) -> None:
    path = tmp_path / "v2.json"
    add_run(path, "claude-haiku-5-5", 2, {"maria": []})
    before = path.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="holds runs of prompt v2 on"):
        add_run(path, "claude-haiku-5-5", 3, {"maria": []})
    with pytest.raises(ValueError, match="claude-haiku-5-5, not v2 on"):
        add_run(path, "claude-sonnet-5-5", 2, {"maria": []})
    assert path.read_text(encoding="utf-8") == before


def test_a_run_of_other_cases_is_refused(tmp_path: Path) -> None:
    # A run with -k clock, or of the held-out part, scores other cases: in
    # the same file its score would not be on the same set.
    path = tmp_path / "v2.json"
    add_run(path, "claude-haiku-5-5", 2, {"maria": [], "clock": []})
    with pytest.raises(ValueError, match="holds runs of other cases"):
        add_run(path, "claude-haiku-5-5", 2, {"clock": []})


def test_a_case_is_fixed_or_broken_only_when_every_run_agrees() -> None:
    assert verdict([FAULT, FAULT], [[], []]) == "fixed"
    assert verdict([[], []], [FAULT, FAULT]) == "broke"
    assert verdict([[], FAULT], [[], []]) == "unsure"
    assert verdict([FAULT, FAULT], [[], FAULT]) == "unsure"
    assert verdict([FAULT], [FAULT, FAULT]) == "fails in both"
    assert verdict([[]], [[], []]) == "passes in both"


def test_each_run_gets_its_own_score() -> None:
    scored = results("m", 2, maria=[[], []], pans=[FAULT, []], clock=[[], []])
    assert run_scores(scored) == [2, 3]


def test_the_diff_lists_every_case_that_failed_a_run() -> None:
    old = results(
        "claude-haiku-5-5",
        2,
        maria=[[], []],
        clock=[FAULT, FAULT],
        pans=[FAULT, []],
    )
    new = results(
        "claude-haiku-5-5",
        3,
        maria=[[], []],
        clock=[[], []],
        pans=[FAULT, FAULT],
    )
    assert diff_lines("v2.json", old, "v3.json", new) == [
        "              v2.json            v3.json",
        "Model         claude-haiku-5-5   claude-haiku-5-5",
        "Prompt        v2                 v3",
        "Changed       prompt version",
        "",
        "clock         0 of 2             2 of 2             fixed",
        "pans          1 of 2             0 of 2             unsure",
        "1 more cases pass every run of both.",
        "Scores        1, 2               2, 2               of 3",
    ]


def test_two_files_of_one_version_say_nothing_changed() -> None:
    # Two files of the same settings show how much the runs alone vary.
    old = results("claude-haiku-5-5", 2, maria=[[]])
    new = results("claude-haiku-5-5", 2, maria=[FAULT])
    lines = diff_lines("a.json", old, "b.json", new)
    assert "Changed       nothing: two files of one version" in lines


def test_two_files_of_different_cases_are_not_compared() -> None:
    old = results("claude-haiku-5-5", 2, maria=[[]], clock=[[]])
    new = results("claude-haiku-5-5", 3, maria=[[]], lampe=[[]])
    with pytest.raises(ValueError, match="different cases.*clock, lampe"):
        diff_lines("v2.json", old, "v3.json", new)


def test_the_diff_command_prints_the_diff_and_refuses_a_bad_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    old = tmp_path / "v2.json"
    new = tmp_path / "v3.json"
    add_run(old, "claude-haiku-5-5", 2, {"maria": [], "clock": FAULT})
    add_run(new, "claude-haiku-5-5", 3, {"maria": [], "clock": []})
    main(["diff", str(old), str(new)])
    output = capsys.readouterr().out
    assert (
        "clock         0 of 1             1 of 1             fixed" in output
    )
    new.write_text('{"model": "claude-haiku-5-5"}', encoding="utf-8")
    with pytest.raises(SystemExit) as stopped:
        main(["diff", str(old), str(new)])
    assert str(stopped.value).startswith("askkit: ")


# A pytest plugin for the run below: it answers every request at the
# network hop with a right ticket and notes the system prompt it got, so
# the eval runs end to end with a made-up key and never reaches the API.
PLUGIN = """
import json
from pathlib import Path

import httpx2

from askkit.golden import load_cases

CASES = load_cases(Path(r"{main}"))


def handle(self, request):
    body = json.loads(request.content)
    with open(r"{seen}", "a", encoding="utf-8") as seen:
        seen.write(json.dumps([body["model"], body["system"]]) + "\\n")
    for case in CASES:
        if case.mentions[0] in body["messages"][0]["content"]:
            break
    ticket = {{
        "order": case.order, "problem": case.mentions[0], "wish": case.wish
    }}
    return httpx2.Response(200, json={{
        "id": "msg_01AskKitRuns", "type": "message", "role": "assistant",
        "model": body["model"],
        "content": [{{"type": "text", "text": json.dumps(ticket)}}],
        "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {{"input_tokens": 236, "output_tokens": 31}},
    }})


httpx2.HTTPTransport.handle_request = handle
"""


def run_eval(
    tmp_path: Path, *arguments: str
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["ANTHROPIC_API_KEY"] = "sk-ant-api03-a-test-key-Ab12"
    environment["PYTHONPATH"] = str(tmp_path)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(PYTHON / "pyproject.toml"),
            "-p",
            "no:cacheprovider",
            "-p",
            "answer_at_the_network",
            str(PYTHON / "evals"),
            *arguments,
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_eval_scores_the_chosen_version_and_model_into_a_file(
    tmp_path: Path,
) -> None:
    seen = tmp_path / "seen.jsonl"
    plugin = PLUGIN.format(
        main=PYTHON / "evals" / "extract" / "main", seen=seen
    )
    (tmp_path / "answer_at_the_network.py").write_text(
        plugin, encoding="utf-8"
    )
    path = tmp_path / "v3.json"
    arguments = ["--prompt-version", "3", "--model", "claude-sonnet-5-5"]
    for _ in range(2):
        result = run_eval(tmp_path, *arguments, "--results", str(path))
        assert result.returncode in (0, 1), result.stdout + result.stderr
    sent = [json.loads(line) for line in seen.read_text().splitlines()]
    assert len(sent) == 32
    for model, system in sent:
        assert model == "claude-sonnet-5-5"
        assert system == render("extract_ticket", {}, 3)
    saved = load_results(path)
    assert (saved.model, saved.prompt_version) == ("claude-sonnet-5-5", 3)
    assert len(saved.cases) == 16
    for case_runs in saved.cases.values():
        assert len(case_runs) == 2


def test_a_replay_cannot_be_counted_as_a_run(tmp_path: Path) -> None:
    (tmp_path / "answer_at_the_network.py").write_text("", encoding="utf-8")
    result = run_eval(
        tmp_path, "--replay", "--results", str(tmp_path / "v2.json")
    )
    assert result.returncode != 0
    assert "--results counts runs of the model" in result.stderr
    assert not (tmp_path / "v2.json").exists()
