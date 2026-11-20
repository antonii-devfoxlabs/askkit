import hashlib
import json
from pathlib import Path
from typing import Any

import httpx2
import pytest
from anthropic import Anthropic

from askkit.ask import ask
from askkit.cli import main
from askkit.rubric import (
    Answer,
    Grade,
    answer_faults,
    answers_passed,
    criterion_scores,
    grade_answer,
    load_answers,
    load_questions,
    load_rubric,
)

ASK_SET = Path(__file__).parent.parent / "evals" / "ask"
MAIN = ASK_SET / "main"
HELD_OUT = ASK_SET / "held_out"
RUBRIC = ASK_SET / "rubric.json"

# The held-out part of the ask set is kept apart as the extract set's
# is (tests/test_golden.py): a change to it is made on purpose.
HELD_OUT_FINGERPRINT = (
    "22c8586ce9f99bc9c6caf0e698aaddc56120579206c26bfa2a9ab41d2425a1b7"
)

POLICY = (ASK_SET / "documents" / "returns-policy.txt").read_text(
    encoding="utf-8"
)

# Answers written for these tests, in the shape the quote request asks
# for. The first is right; the second quotes two real sentences and
# still promises a refund the policy does not give for a wrong colour.
RIGHT_POSTAGE = (
    "<quote>If we sent the wrong item or colour, we pay the return"
    " postage.</quote>\n"
    "<answer>We pay the return postage, because we sent the wrong colour."
    "</answer>"
)
REFUND_POSTAGE = (
    "<quote>If we sent the wrong item or colour, we pay the return"
    " postage.</quote>\n"
    "<quote>We refund the price of the item within 5 working days of"
    " receiving it.</quote>\n"
    "<answer>We pay the return postage, and we refund the price within 5"
    " working days of receiving the shade.</answer>"
)


def fingerprint(folder: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(folder.iterdir()):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def response_body(text: str, stop_reason: str = "end_turn") -> dict[str, Any]:
    return {
        "id": "msg_01AskKitRubric",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 410, "output_tokens": 52},
    }


def fake_client(body: dict[str, Any]) -> Anthropic:
    def reply(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=body)

    return Anthropic(
        api_key="sk-ant-api03-a-test-key-Ab12",
        http_client=httpx2.Client(transport=httpx2.MockTransport(reply)),
    )


def faults_of(text: str, stop_reason: str = "end_turn") -> list[str]:
    client = fake_client(response_body(text, stop_reason))
    response = ask(client, "Who pays?", "claude-haiku-5-5", 1000, "Be brief.")
    return answer_faults(response, {"returns-policy.txt": POLICY})


def replies(*typed: str) -> Any:
    # Stands in for input(): returns what a grader types, in order, and
    # keeps every prompt it was shown.
    queue = list(typed)
    shown: list[str] = []

    def read(prompt: str) -> str:
        shown.append(prompt)
        return queue.pop(0)

    read.shown = shown  # type: ignore[attr-defined]
    return read


def test_both_parts_of_the_ask_set_load() -> None:
    assert len(load_questions(MAIN)) == 6
    assert len(load_questions(HELD_OUT)) == 2


def test_no_question_is_in_both_parts() -> None:
    main_questions = [item.question for item in load_questions(MAIN)]
    for item in load_questions(HELD_OUT):
        assert item.question not in main_questions


def test_the_held_out_part_of_the_ask_set_is_unchanged() -> None:
    assert fingerprint(HELD_OUT) == HELD_OUT_FINGERPRINT


def test_the_rubric_loads_with_an_example_each_way() -> None:
    rubric = load_rubric(RUBRIC)
    assert [criterion.name for criterion in rubric] == [
        "must",
        "grounded",
        "direct",
    ]
    for criterion in rubric:
        assert criterion.asks.endswith("?")
        assert criterion.passes and criterion.fails


def test_a_question_without_its_document_is_refused(tmp_path: Path) -> None:
    part = tmp_path / "main"
    part.mkdir()
    (tmp_path / "documents").mkdir()
    case = {
        "name": "gift",
        "question": "Can a customer return a gift card?",
        "documents": ["returns-policy.txt"],
        "must": ["No"],
        "why": "One sentence answers it",
    }
    (part / "cases.json").write_text(json.dumps([case]), encoding="utf-8")
    with pytest.raises(ValueError, match="no document 'returns-policy.txt'"):
        load_questions(part)


def test_a_criterion_named_twice_is_refused(tmp_path: Path) -> None:
    line = {"name": "must", "asks": "?", "passes": "a", "fails": "b"}
    path = tmp_path / "rubric.json"
    path.write_text(json.dumps([line, line]), encoding="utf-8")
    with pytest.raises(ValueError, match="two criteria 'must'"):
        load_rubric(path)


def test_an_empty_rubric_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "rubric.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="no criteria"):
        load_rubric(path)


def test_a_fail_is_not_taken_without_a_note() -> None:
    rubric = load_rubric(RUBRIC)
    answer = Answer(case="postage", text=REFUND_POSTAGE, faults=[])
    read = replies("y", "n", "", "refund is section 1's", "maybe", "y")
    grades = grade_answer(answer, rubric, read)
    assert [grade.passed for grade in grades] == [True, False, True]
    assert grades[1].note == "refund is section 1's"
    assert read.shown[1].startswith("grounded: Is the rest of it")
    assert read.shown[2:4] == ["Why not? ", "Why not? "]
    assert read.shown[5].startswith("direct: ")


def test_an_answer_code_failed_is_not_shown_to_the_grader() -> None:
    rubric = load_rubric(RUBRIC)
    faults = ["the answer quotes nothing from the documents"]
    answer = Answer(case="postage", text="We pay.", faults=faults)
    read = replies()
    grades = grade_answer(answer, rubric, read)
    assert read.shown == []
    assert [grade.passed for grade in grades] == [False, False, False]
    assert grades[0].note == "the answer quotes nothing from the documents"


def test_scores_count_each_criterion_and_whole_answers() -> None:
    rubric = load_rubric(RUBRIC)
    grades = [
        Grade(case="postage", criterion="must", passed=True, note=""),
        Grade(
            case="postage",
            criterion="grounded",
            passed=False,
            note="refund",
        ),
        Grade(case="postage", criterion="direct", passed=True, note=""),
        Grade(case="gift", criterion="must", passed=True, note=""),
        Grade(case="gift", criterion="grounded", passed=True, note=""),
        Grade(case="gift", criterion="direct", passed=True, note=""),
    ]
    assert criterion_scores(grades, rubric) == {
        "must": (2, 2),
        "grounded": (1, 2),
        "direct": (2, 2),
    }
    assert answers_passed(grades) == (1, 2)


def test_a_right_answer_has_no_code_faults() -> None:
    assert faults_of(RIGHT_POSTAGE) == []


def test_a_wrong_answer_with_real_quotes_passes_the_code_checks() -> None:
    # The reason this lesson grades by hand: code finds nothing wrong.
    assert faults_of(REFUND_POSTAGE) == []


def test_code_faults_name_a_quote_the_documents_lack() -> None:
    text = "<quote>We pay for everything.</quote><answer>We pay.</answer>"
    assert faults_of(text) == [
        'not in the documents: "We pay for everything."'
    ]


def test_code_faults_name_an_answer_with_no_quote() -> None:
    assert faults_of("<answer>We pay.</answer>") == [
        "the answer quotes nothing from the documents"
    ]


def test_the_way_out_needs_no_quote() -> None:
    assert faults_of("<answer>I don't know.</answer>") == []


def test_a_cut_off_answer_is_one_fault() -> None:
    assert faults_of("<quote>If we", "max_tokens") == [
        "the answer was cut off at 52 tokens; give it a larger --max-tokens"
    ]


@pytest.fixture
def ask_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # A one-part copy of the set with two of its questions.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    (tmp_path / "documents").mkdir()
    for name in ("returns-policy.txt", "order-6083.txt"):
        source = ASK_SET / "documents" / name
        (tmp_path / "documents" / name).write_bytes(source.read_bytes())
    part = tmp_path / "main"
    part.mkdir()
    cases = json.loads((MAIN / "cases.json").read_text(encoding="utf-8"))
    chosen = [cases[0], cases[4]]
    (part / "cases.json").write_text(json.dumps(chosen), encoding="utf-8")
    return part


def test_answer_saves_every_answer_with_its_code_faults(
    ask_set: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bodies = [
        response_body(REFUND_POSTAGE),
        response_body("<answer>No.</answer>"),
    ]
    sent: list[dict[str, Any]] = []

    def reply(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return httpx2.Response(200, json=bodies[len(sent) - 1])

    def make(api_key: str, **options: Any) -> Anthropic:
        transport = httpx2.MockTransport(reply)
        return Anthropic(
            api_key=api_key, http_client=httpx2.Client(transport=transport)
        )

    monkeypatch.setattr("askkit.cli.Anthropic", make)
    main(["answer", str(ask_set), "--out", "answers.json"])
    assert capsys.readouterr().out.splitlines() == [
        "postage       to grade",
        "gift          the answer quotes nothing from the documents",
        "",
        "Model:         claude-haiku-5-5",
        "Tokens:        820 in, 104 out",
        "Cost:          $0.000134",
        "Saved:         answers.json",
    ]
    question = sent[0]["messages"][0]["content"]
    assert "<source>order-6083.txt</source>" in question
    assert question.endswith(
        "<question>\nWho pays the return postage for order 6083?\n</question>"
    )
    answers = load_answers(Path("answers.json"))
    assert answers[0] == Answer(
        case="postage", text=REFUND_POSTAGE, faults=[]
    )
    assert answers[1].faults == [
        "the answer quotes nothing from the documents"
    ]


def test_a_declined_answer_is_saved_without_its_text(
    ask_set: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = response_body("<quote>If we", "refusal")

    def make(api_key: str, **options: Any) -> Anthropic:
        return fake_client(body)

    monkeypatch.setattr("askkit.cli.Anthropic", make)
    main(["answer", str(ask_set), "--out", "answers.json"])
    answers = load_answers(Path("answers.json"))
    assert answers[0].text == ""
    assert answers[0].faults == ["the model declined to answer this question"]


def test_grade_asks_each_line_and_saves_the_grades(
    ask_set: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    answers = [
        Answer(case="postage", text=REFUND_POSTAGE, faults=[]),
        Answer(
            case="gift",
            text="<answer>No.</answer>",
            faults=["the answer quotes nothing from the documents"],
        ),
    ]
    saved = [answer.model_dump() for answer in answers]
    Path("answers.json").write_text(json.dumps(saved), encoding="utf-8")
    typed = iter(["y", "n", "promises section 1's refund", "y"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(typed))
    main(
        [
            "grade",
            "answers.json",
            "--cases",
            str(ask_set),
            "--rubric",
            str(RUBRIC),
            "--out",
            "grades.json",
        ]
    )
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "Rubric"
    assert (
        out[1]
        == "  must: Does it state every fact on the must list, in any words?"
    )
    assert out[11:16] == [
        "Answer 1 of 2: postage",
        "Question: Who pays the return postage for order 6083?",
        "Must:     The shop pays the return postage",
        "Must:     Because it sent the wrong colour",
        "",
    ]
    assert (
        "Failed by code: the answer quotes nothing from the documents" in out
    )
    assert out[-5:] == [
        "must          1 of 2",
        "grounded      0 of 2",
        "direct        1 of 2",
        "All criteria  0 of 2 answers",
        "Saved:         grades.json",
    ]
    grades = json.loads(Path("grades.json").read_text(encoding="utf-8"))
    assert grades[1] == {
        "case": "postage",
        "criterion": "grounded",
        "passed": False,
        "note": "promises section 1's refund",
    }


def test_grade_refuses_an_answer_to_a_question_not_in_the_set(
    ask_set: Path,
) -> None:
    answer = Answer(case="canada", text="I don't know.", faults=[])
    Path("answers.json").write_text(
        json.dumps([answer.model_dump()]), encoding="utf-8"
    )
    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                "grade",
                "answers.json",
                "--cases",
                str(ask_set),
                "--rubric",
                str(RUBRIC),
                "--out",
                "grades.json",
            ]
        )
    assert exit_info.value.code == f"askkit: no case 'canada' in {ask_set}"


def test_answer_refuses_a_max_tokens_too_large_to_wait_for(
    ask_set: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # askkit answer waits for each whole answer, at most --timeout
    # seconds; an answer that may take longer is refused before any call,
    # instead of timing out after the tokens were paid for.
    def make(api_key: str, **options: Any) -> Anthropic:
        return fake_client(response_body(REFUND_POSTAGE))

    monkeypatch.setattr("askkit.cli.Anthropic", make)
    with pytest.raises(SystemExit) as stopped:
        main(
            [
                "answer",
                str(ask_set),
                "--out",
                "answers.json",
                "--max-tokens",
                "64000",
            ]
        )
    message = str(stopped.value.code)
    assert message.startswith("askkit: --max-tokens 64,000 may take up to")
    assert not Path("answers.json").exists()
