import json
from collections.abc import Callable
from pathlib import Path

from anthropic.types import Message
from pydantic import BaseModel, ConfigDict

from askkit.ask import answer_text, stop_warning
from askkit.documents import missing_quotes, unchecked


# One question of the ask set, with the documents it is asked about. An
# answer to it has no field to compare, so the case gives the facts a
# right answer must state, in words for the person who grades it.
class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    question: str
    documents: list[str]
    must: list[str]
    why: str


# One criterion of the rubric: a question the grader answers yes or no
# about every answer, with an answer that passes and one that fails, so
# two graders read it the same way.
class Criterion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    asks: str
    passes: str
    fails: str


# What the model wrote for one case's question, which model wrote it, and
# what code already found wrong with it, so the grader's time goes only to
# what code cannot see. The model matters once a question can fall back to
# another one: a grade belongs to the model that earned it.
class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case: str
    model: str
    text: str
    faults: list[str]


# One grader's yes or no on one criterion for one case's answer. A fail has a
# note saying why: it is what tells you what to change.
class Grade(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case: str
    criterion: str
    passed: bool
    note: str


def load_questions(folder: Path) -> list[Question]:
    # The documents live beside the parts, in documents/, shared by both.
    data = json.loads((folder / "cases.json").read_text(encoding="utf-8"))
    questions: list[Question] = []
    names: list[str] = []
    for item in data:
        question = Question.model_validate(item)
        if question.name in names:
            raise ValueError(
                f"{folder.name}: two cases for {question.name!r}"
            )
        for document in question.documents:
            if not (folder.parent / "documents" / document).is_file():
                raise ValueError(
                    f"{folder.name}: no document {document!r}"
                    f" for {question.name!r}"
                )
        questions.append(question)
        names.append(question.name)
    return questions


def load_rubric(path: Path) -> list[Criterion]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rubric: list[Criterion] = []
    names: list[str] = []
    for item in data:
        criterion = Criterion.model_validate(item)
        if criterion.name in names:
            raise ValueError(f"{path.name}: two criteria {criterion.name!r}")
        rubric.append(criterion)
        names.append(criterion.name)
    if not rubric:
        raise ValueError(f"{path.name}: no criteria")
    return rubric


def yes_or_no(prompt: str, read: Callable[[str], str]) -> bool:
    while True:
        reply = read(prompt).strip().lower()
        if reply in ("y", "n"):
            return reply == "y"


def grade_answer(
    answer: Answer,
    rubric: list[Criterion],
    read: Callable[[str], str],
) -> list[Grade]:
    # An answer code has already failed is failed on every criterion with
    # code's faults as the note: no person reads it. Otherwise the grader
    # answers each criterion in turn, and a fail is not taken without a
    # note.
    grades: list[Grade] = []
    for criterion in rubric:
        if answer.faults:
            note = "; ".join(answer.faults)
            passed = False
        else:
            passed = yes_or_no(f"{criterion.name}: {criterion.asks} ", read)
            note = ""
            while not passed and not note:
                note = read("Why not? ").strip()
        grades.append(
            Grade(
                case=answer.case,
                criterion=criterion.name,
                passed=passed,
                note=note,
            )
        )
    return grades


def criterion_scores(
    grades: list[Grade], rubric: list[Criterion]
) -> dict[str, tuple[int, int]]:
    # Passed and graded, per criterion, in the rubric's order.
    scores: dict[str, tuple[int, int]] = {}
    for criterion in rubric:
        passed = 0
        graded = 0
        for grade in grades:
            if grade.criterion == criterion.name:
                graded += 1
                if grade.passed:
                    passed += 1
        scores[criterion.name] = (passed, graded)
    return scores


def answers_passed(grades: list[Grade]) -> tuple[int, int]:
    # An answer passes only when it passes every criterion.
    failed: list[str] = []
    cases: list[str] = []
    for grade in grades:
        if grade.case not in cases:
            cases.append(grade.case)
        if not grade.passed and grade.case not in failed:
            failed.append(grade.case)
    return len(cases) - len(failed), len(cases)


def load_answers(path: Path) -> list[Answer]:
    data = json.loads(path.read_text(encoding="utf-8"))
    answers: list[Answer] = []
    for item in data:
        answers.append(Answer.model_validate(item))
    return answers


def answer_faults(response: Message, documents: dict[str, str]) -> list[str]:
    # The checks askkit ask runs, in code: a whole answer, quotes that are
    # in the documents, and at least one quote unless it is the way out.
    warning = stop_warning(response)
    if warning is not None:
        return [warning]
    text = answer_text(response)
    faults: list[str] = []
    for quote in missing_quotes(text, documents):
        faults.append(f'not in the documents: "{quote}"')
    if unchecked(text):
        faults.append("the answer quotes nothing from the documents")
    return faults
