from pydantic import BaseModel

from askkit.golden import Case
from askkit.runs import Results, run_scores


# One failed answer of an eval run: the results file and the run it
# came from, its case, and every fault the checks found in it.
class Failure(BaseModel):
    source: str
    case: str
    faults: list[str]


def failures(file_name: str, results: Results) -> list[Failure]:
    # Every failed answer in one results file, run by run, the cases of
    # each run in alphabetical order. A pass has no faults: left out.
    found: list[Failure] = []
    for run in range(len(run_scores(results))):
        for name in sorted(results.cases):
            faults = results.cases[name][run]
            if faults != []:
                source = f"{file_name} run {run + 1}"
                found.append(Failure(source=source, case=name, faults=faults))
    return found


def fault_field(fault: str) -> str:
    # The words before the first colon: the field a check found wrong,
    # such as "problem" or "wish", or "stop" for an answer cut off or
    # declined. A schema fault starts with its field the same way.
    return fault.split(":")[0]


def by_field(found: list[Failure]) -> dict[str, list[str]]:
    # Each field with the case of every failed answer that names it. An
    # answer with two faults in one field counts once for that field.
    groups: dict[str, list[str]] = {}
    for failure in found:
        fields: list[str] = []
        for fault in failure.faults:
            if fault_field(fault) not in fields:
                fields.append(fault_field(fault))
        for field in fields:
            if field not in groups:
                groups[field] = []
            groups[field].append(failure.case)
    return groups


def by_kind(
    found: list[Failure], cases: dict[str, Case]
) -> dict[str, list[str]]:
    # Each kind of email (lesson 5.2) with the case of every failed
    # answer whose email is of that kind.
    groups: dict[str, list[str]] = {}
    for failure in found:
        kind = cases[failure.case].kind
        if kind not in groups:
            groups[kind] = []
        groups[kind].append(failure.case)
    return groups


def group_lines(title: str, groups: dict[str, list[str]]) -> list[str]:
    # The largest group first, the order to read them in; a tie goes in
    # alphabetical order. A case that failed two runs counts twice in
    # Answers, and is named once.
    def size(item: tuple[str, list[str]]) -> tuple[int, str]:
        return (-len(item[1]), item[0])

    lines = [f"{title:<14}{'Answers':<9}Cases"]
    for name, group in sorted(groups.items(), key=size):
        names = ", ".join(sorted(set(group)))
        lines.append(f"{name:<14}{len(group):<9}{names}")
    return lines


def fault_lines(
    files: dict[str, Results], cases: dict[str, Case]
) -> list[str]:
    # Every failed answer of every file, grouped by field and by kind,
    # then listed one by one, to be read with its email. Only cases of
    # the given part are read: a held-out result is refused, so its
    # failures stay unread.
    found: list[Failure] = []
    runs = 0
    for file_name, results in files.items():
        unknown = sorted(set(results.cases) - set(cases))
        if unknown:
            raise ValueError(
                f"{file_name} holds cases that are not in this part of the"
                " golden set: " + ", ".join(unknown) + "; read the failures"
                " of the main part only"
            )
        found.extend(failures(file_name, results))
        runs = runs + len(run_scores(results))
    runs_text = f"{runs} runs"
    if runs == 1:
        runs_text = "1 run"
    lines = [
        f"{len(found)} failed answers in {runs_text} of {len(cases)} cases",
        "",
    ]
    if not found:
        return lines[:1]
    lines.extend(group_lines("By field", by_field(found)))
    lines.append("")
    lines.extend(group_lines("By kind", by_kind(found, cases)))
    lines.append("")
    lines.append("Every failed answer, to read beside its email")
    for failure in found:
        faults = "; ".join(failure.faults)
        lines.append(f"{failure.source:<16}{failure.case:<10}{faults}")
    return lines
