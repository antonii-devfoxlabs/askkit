# AskKit

The project of AI Engineering, Level 1 (LLM Application Foundations), on
devfoxlabs: a command-line app built on a large language model. Each
module of the course adds one piece; the tag `m01`, `m02`, … holds the
code as it stands at the end of that module.

## What it does so far

`askkit estimate` reads a text file, works out how many tokens it is for
one model, and prints what sending it to that model would cost:

```bash
cd python
uv run askkit estimate email.txt -m claude-haiku-5-5 --output 40 --calls 36000
```

It also checks that the text and the answer fit in the model's context
window, and that the answer is no longer than the model can write in one
call; if not, it says so and exits with status 1.

A wrong answer has a price too. Give `estimate` the share of calls you
expect to come back wrong and what one costs you, and for more than one
call it adds the errors to the bill:

```bash
cd python
uv run askkit estimate email.txt --output 5 --calls 36000 --error-rate 0.03 --cost-per-error 0.50
```

`askkit compare` prices the same file on every model AskKit knows, so
the trade-off between the providers is one command:

```bash
cd python
uv run askkit compare email.txt --output 40 --calls 36000
```

AskKit knows one comparable model from each of four providers
(Anthropic, Google, OpenAI and Qwen, an open-weight family), the
high-volume tier of each; add a model with one line in each data file. Every model counts tokens its own way: a
model whose tokenizer is published (Qwen) is counted exactly, on this
machine, with Hugging Face's `tokenizers`; the first count downloads the
tokenizer into `~/.cache/askkit/`. Every other model gets an estimate
from the number of characters, at the rate its provider's docs give.

Prices live in `python/src/askkit/prices.py` and the limits in
`python/src/askkit/limits.py`, each with its sources and the date they
were checked.

## Tags

- `m01`, end of Module 1 (How LLMs Work, for Engineers): `estimate` and
  `compare`, with no call to a model yet.

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
