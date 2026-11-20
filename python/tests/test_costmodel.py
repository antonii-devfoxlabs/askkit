import json
from pathlib import Path

import pytest

from askkit.cli import main
from askkit.costmodel import (
    CostModel,
    Feature,
    load_cost_model,
    month_costs,
    report_lines,
    use_cost,
)

SAMPLE = Path(__file__).parent.parent / "samples" / "month.json"


def feature(**changes: object) -> Feature:
    fields: dict[str, object] = {
        "name": "classify",
        "model": "claude-haiku-5-5",
        "uses": 12000,
        "calls": 1,
        "input_tokens": 214,
        "output_tokens": 12,
        "fees": 0.0,
    }
    fields.update(changes)
    return Feature.model_validate(fields)


def write_model(tmp_path: Path, features: list[dict[str, object]]) -> Path:
    path = tmp_path / "month.json"
    path.write_text(json.dumps({"features": features}), encoding="utf-8")
    return path


def test_one_use_costs_its_tokens_at_the_models_prices() -> None:
    # 214 x $0.10 and 12 x $0.50 a million tokens.
    assert use_cost(feature()) == pytest.approx(0.0000274)


def test_a_fee_is_added_to_every_use_not_once_a_month() -> None:
    search = feature(input_tokens=9240, output_tokens=120, fees=0.01)
    assert use_cost(search) == pytest.approx(0.010984)
    [row] = month_costs(CostModel(features=[search.model_copy()]))
    assert row.month == pytest.approx(0.010984 * 12000)


def test_the_price_step_goes_by_each_call_not_by_the_whole_use() -> None:
    # Eight calls of 60,000 tokens: 480,000 in a use, each call under the
    # 100,000 of Haiku 5.5's step, so every token is at the usual price.
    chat = feature(calls=8, input_tokens=480_000, output_tokens=0)
    assert use_cost(chat) == pytest.approx(0.048)
    # One call of 120,000 tokens is over the step: the whole call at $0.50.
    long_call = feature(calls=1, input_tokens=120_000, output_tokens=0)
    assert use_cost(long_call) == pytest.approx(0.06)


def test_features_come_dearest_first_with_their_share_of_the_month() -> None:
    rows = month_costs(
        CostModel(
            features=[
                feature(name="classify"),
                feature(name="search", uses=600, fees=0.01),
            ]
        )
    )
    assert [row.name for row in rows] == ["search", "classify"]
    assert sum([row.share for row in rows]) == pytest.approx(1.0)


def test_a_month_with_no_uses_has_no_shares_and_no_error() -> None:
    [row] = month_costs(CostModel(features=[feature(uses=0)]))
    assert (row.month, row.share) == (0.0, 0.0)


def test_the_sample_month_comes_mostly_from_the_searches() -> None:
    rows = month_costs(load_cost_model(SAMPLE))
    assert rows[0].name == "ask, a search"
    assert rows[0].share == pytest.approx(0.708, abs=0.001)
    assert sum([row.month for row in rows]) == pytest.approx(9.30628)
    assert sum([row.uses for row in rows]) == 18_700
    classify = [row for row in rows if row.name == "classify"][0]
    assert classify.uses / 18_700 > 0.6
    assert classify.share < 0.04


def test_the_report_lines_up_its_columns() -> None:
    lines = report_lines(month_costs(load_cost_model(SAMPLE)))
    assert lines[0] == (
        "Feature                Uses  Calls     One use   A month   Share"
    )
    assert lines[1] == (
        "ask, a search           600      1   $0.010984     $6.59   70.8%"
    )
    assert lines[-1] == (
        "All features         18,700                        $9.31"
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"model": "claude-haiku-9"},
            "AskKit has no prices for claude-haiku-9",
        ),
        ({"calls": 0}, "a use makes at least one call"),
        ({"uses": -1}, "uses, tokens and fees cannot be below zero"),
        ({"fees": -0.01}, "uses, tokens and fees cannot be below zero"),
    ],
)
def test_a_feature_that_cannot_be_priced_is_refused(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    fields = feature().model_dump()
    fields.update(changes)
    with pytest.raises(ValueError, match=f"month.json: classify: {message}"):
        load_cost_model(write_model(tmp_path, [fields]))


def test_two_features_of_one_name_are_refused(tmp_path: Path) -> None:
    fields = feature().model_dump()
    with pytest.raises(ValueError, match="two features named classify"):
        load_cost_model(write_model(tmp_path, [fields, fields]))


def test_a_field_the_model_does_not_know_is_refused(tmp_path: Path) -> None:
    fields = feature().model_dump()
    fields["cache_hits"] = 0.5
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        load_cost_model(write_model(tmp_path, [fields]))


def test_askkit_costs_prints_the_report(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["costs", str(SAMPLE)])
    printed = capsys.readouterr().out.splitlines()
    assert printed == report_lines(month_costs(load_cost_model(SAMPLE)))


def test_askkit_costs_stops_with_a_message_on_a_bad_file(
    tmp_path: Path,
) -> None:
    path = write_model(tmp_path, [feature(calls=0).model_dump()])
    with pytest.raises(SystemExit) as stop:
        main(["costs", str(path)])
    assert stop.value.code == (
        "askkit: month.json: classify: a use makes at least one call"
    )


def test_askkit_costs_names_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as stop:
        main(["costs", str(tmp_path / "nothing.json")])
    assert "askkit: [Errno 2] No such file or directory" in str(
        stop.value.code
    )
