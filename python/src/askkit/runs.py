from pathlib import Path

from pydantic import BaseModel, ConfigDict


# The results of one version of the eval: the model and the prompt
# version it ran with, and for each case the faults of every run, [] for
# a pass. Runs of the same settings add up in one file. The model
# samples, so one run is one draw: a case that passes one run and fails
# the next is the draw, not the version, and only a second run shows it.
class Results(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    prompt_version: int
    cases: dict[str, list[list[str]]]


def load_results(path: Path) -> Results:
    return Results.model_validate_json(path.read_text(encoding="utf-8"))


def add_run(
    path: Path, model: str, version: int, faults: dict[str, list[str]]
) -> Results:
    # Adds one run to the file, or starts the file with it. A run of other
    # settings or of other cases would mix two things in one file, so it
    # is refused and the file stays as it was.
    if path.is_file():
        results = load_results(path)
        if (results.model, results.prompt_version) != (model, version):
            raise ValueError(
                f"{path.name} holds runs of prompt v{results.prompt_version}"
                f" on {results.model}, not v{version} on {model}; give"
                " this run a file of its own"
            )
        if sorted(results.cases) != sorted(faults):
            raise ValueError(
                f"{path.name} holds runs of other cases; run the same"
                " cases again, or give this run a file of its own"
            )
    else:
        results = Results(model=model, prompt_version=version, cases={})
        for name in faults:
            results.cases[name] = []
    for name, run_faults in faults.items():
        results.cases[name].append(run_faults)
    text = results.model_dump_json(indent=2)
    path.write_text(text + "\n", encoding="utf-8")
    return results


def passes(runs: list[list[str]]) -> int:
    count = 0
    for faults in runs:
        if faults == []:
            count = count + 1
    return count


def verdict(old: list[list[str]], new: list[list[str]]) -> str:
    # What one case says about the change. A version's runs that disagree
    # on a case make it "unsure": that case cannot tell the versions
    # apart, whatever the other version did. Only a case that holds in
    # every run of both is fixed or broken.
    old_passes = passes(old)
    new_passes = passes(new)
    if old_passes not in (0, len(old)) or new_passes not in (0, len(new)):
        return "unsure"
    if old_passes == 0 and new_passes == len(new):
        return "fixed"
    if old_passes == len(old) and new_passes == 0:
        return "broken"
    if new_passes == 0:
        return "fails in both"
    return "passes in both"


def run_scores(results: Results) -> list[int]:
    # How many cases passed in each run, in the order of the runs. Every
    # case has one entry per run, so the first case gives the count.
    runs = 0
    for case_runs in results.cases.values():
        runs = len(case_runs)
        break
    scores: list[int] = []
    for run in range(runs):
        score = 0
        for case_runs in results.cases.values():
            if case_runs[run] == []:
                score = score + 1
        scores.append(score)
    return scores


def scores_text(results: Results) -> str:
    return ", ".join([str(score) for score in run_scores(results)])


def changed(old: Results, new: Results) -> str:
    differences: list[str] = []
    if old.model != new.model:
        differences.append("model")
    if old.prompt_version != new.prompt_version:
        differences.append("prompt version")
    if not differences:
        return "nothing: two files of one version"
    return ", ".join(differences)


def diff_lines(
    old_name: str, old: Results, new_name: str, new: Results
) -> list[str]:
    # The two files side by side, case by case: every case that failed a
    # run of either, with how many of its runs passed and its verdict.
    # Cases that passed every run of both are counted, not listed.
    if sorted(old.cases) != sorted(new.cases):
        only = sorted(set(old.cases) ^ set(new.cases))
        raise ValueError(
            "the two files hold different cases, so their scores are not"
            " on the same set: " + ", ".join(only)
        )

    def row(label: str, old_text: str, new_text: str, note: str = "") -> str:
        return f"{label:<14}{old_text:<19}{new_text:<19}{note}".rstrip()

    lines = [
        row("", old_name, new_name),
        row("Model", old.model, new.model),
        row("Prompt", f"v{old.prompt_version}", f"v{new.prompt_version}"),
        row("Changed", changed(old, new), ""),
        "",
    ]
    both_pass = 0
    for name in sorted(old.cases):
        found = verdict(old.cases[name], new.cases[name])
        if found == "passes in both":
            both_pass = both_pass + 1
            continue
        old_text = f"{passes(old.cases[name])} of {len(old.cases[name])}"
        new_text = f"{passes(new.cases[name])} of {len(new.cases[name])}"
        lines.append(row(name, old_text, new_text, found))
    lines.append(f"{both_pass} more cases pass every run of both.")
    lines.append(
        row(
            "Scores",
            scores_text(old),
            scores_text(new),
            f"of {len(old.cases)}",
        )
    )
    return lines
