# AskKit

The project of AI Engineering, Level 1 (LLM Application Foundations), on
devfoxlabs: a command-line app built on a large language model. Each
module of the course adds one piece; the tag `m01`, `m02`, … holds the
code as it stands at the end of that module.

## What it does so far

`askkit estimate` reads a text file, estimates how many tokens it is, and
prints what sending it to a model would cost:

```bash
cd python
uv run askkit estimate email.txt -m claude-haiku-5-5 --output 40 --calls 36000
```

It also checks that the text and the answer fit in the model's context
window, and that the answer is no longer than the model can write in one
call; if not, it says so and exits with status 1.

The token count is an estimate worked out from the number of characters.
Prices live in `python/src/askkit/prices.py` and the limits in
`python/src/askkit/limits.py`, each with the date it was checked.

## Layout

- `python/`: the Python app, a uv project (Python 3.14).
  - `src/askkit/`: the package.
  - `tests/`: pytest tests. They never call a model's API.

## Checks

```bash
cd python
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest
```

CI runs the same four on every push.
