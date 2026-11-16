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
high-volume tier of each; add a model with one line in each data file.
Every model counts tokens its own way: a model whose tokenizer is
published (Qwen) is counted exactly, on this machine, with Hugging
Face's `tokenizers`; the first count downloads the tokenizer into
`~/.cache/askkit/`. Every other model gets an estimate from the number
of characters, at the rate its provider's docs give.

Prices live in `python/src/askkit/prices.py` and the limits in
`python/src/askkit/limits.py`, each with its sources and the date they
were checked.

`askkit key` checks that AskKit finds your API key in `python/.env`
(copy `python/.env.example` and put the key after the `=`), and shows it
masked. `askkit ask` sends a question to a model with Anthropic's SDK and
prints the answer, the tokens the call used (and how many of the output
tokens were thinking, when the API says) and what it cost:

```bash
cd python
uv run askkit ask "In one sentence, what is a context window?"
```

`ask` calls Anthropic's models only, so `-m` takes the Claude models in
the price table (Claude Haiku 5.5 by default); `--max-tokens` is the
room for the answer (1000 by default), the model's thinking included;
`--effort` sets how much work the model puts into the answer (`low`,
`medium`, `high`, `xhigh` or `max`), and without it the model's own
default holds, `medium` on Claude Haiku 5.5. Every question goes with
AskKit's system prompt, `ASK_SYSTEM` in `python/src/askkit/prompts.py`;
`--system-file ticket.txt` sends the prompt in that file instead.
Under the cost, `ask` prints the stop reason, the API's word for why
the answer ended. When the answer is not whole (cut off at
`--max-tokens` or at the end of the context window, or declined),
`ask` says so and exits with exit code 1. Of a declined answer it prints
no part, as Anthropic advises, and names the category when the API
gives one.
When the call itself fails (a rejected key, a rate limit, Anthropic's
servers overloaded, no connection), `ask` prints one short message
from `python/src/askkit/errors.py` saying what happened and what to do,
instead of a traceback, and exits with exit code 1. The SDK has already
sent a busy or failed request twice more by then.

`--document FILE` sends a long text with the question: a policy, a
contract, a manual. Give it once per file, as many times as you need.
Each file goes into the user message in `<documents>` tags, numbered and
named after its path, then a request to quote first
(`QUOTE_REQUEST` in `python/src/askkit/prompts.py`), then the question,
last:

```bash
cd python
uv run askkit ask "Who pays the postage for a wrong colour?" --document returns-policy.txt
```

The model copies the sentences its answer rests on into `<quote>` tags
and checks that it quoted every part that applies, an exception
included, before it answers. Under the stop reason, `ask` prints how many of those
quotes it found in the documents (line breaks and runs of spaces do not
count), and when one is not there it names it and exits with exit code
1, so the answer goes to a person instead of on.

## Exploring in a notebook

`python/notebooks/explore.ipynb` is a Jupyter notebook for trying a call
and reading its response object piece by piece. It imports AskKit's own
functions, so what you find there is one step from the package. Start
JupyterLab inside the project, without adding it to AskKit's
dependencies:

```bash
cd python
uv run --with jupyter jupyter lab
```

Every run of the cell that calls `ask` is a real, billed call. A
notebook saves its outputs in the file, a key or a paid answer
included, so clear them before you commit (**Edit**, then **Clear
Outputs of All Cells**, then save); a test fails on any notebook saved
with outputs. What a notebook settles moves into `src/askkit/` with a
test, as the thinking count of `ask` did.

## Tags

- `m01`, end of Module 1 (How LLMs Work, for Engineers): `estimate` and
  `compare`, with no call to a model yet.
- `m02`, end of Module 2 (Your First API Calls): `key` and `ask`, which
  sends a question with AskKit's system prompt and prints the answer,
  its tokens, its cost and its stop reason, or one line saying what
  failed; and the exploring notebook.

## Layout

- `python/`: the Python app, a uv project (Python 3.14).
  - `src/askkit/`: the package.
  - `notebooks/`: notebooks for exploring, saved without outputs.
  - `tests/`: pytest tests. They never call a model's API: the calls go
    to a fake transport that answers with a recorded response.

## Checks

```bash
cd python
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest
```

CI runs the same four on every push.
