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
AskKit's system prompt, the template `ask_system` (see "Prompts" below);
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
named after its path, then a request to quote first, then the
question, last; the template `quote_request` holds that layout:

```bash
cd python
uv run askkit ask "Who pays the postage for a wrong colour?" --document returns-policy.txt
```

The model copies the sentences its answer rests on into `<quote>` tags
and checks that it quoted every part that applies, an exception
included, before it answers. Under the stop reason, `ask` prints how many of those
quotes it found in the documents (line breaks and runs of spaces do not
count), and when one is not there it names it and exits with exit code
1, so the answer goes to a person instead of on. An answer that quotes
nothing, such as one whose quotes are in a tag the request did not ask
for, exits the same way, unless it is "I don't know": there would be
nothing to check it against.

`askkit extract` turns a customer's email into a support ticket, as
JSON. The ticket is one class, `Ticket` in `python/src/askkit/extract.py`,
a Pydantic model with three fields: `order` (a whole number, or `null`
when the email gives none), `problem` and `wish` (`replacement`,
`refund` or `not stated`), and nothing else. `extract` sends the email
with the template `extract_ticket` as the system prompt and the JSON
Schema made from `Ticket` (`TICKET_SCHEMA`), which the API holds the
answer to while the model writes it:

```bash
cd python
uv run askkit extract email.txt
```

It prints the ticket, then the same lines as `ask`. It reads the stop
reason first: an answer cut off at `--max-tokens` (1000 by default) or
declined may not match the schema, so it is not read, and `extract`
says why and exits with exit code 1; neither is asked for again, since
the same room or the same email stops the same way. A finished answer
is checked against `Ticket` again in AskKit's own code, because the API
holds the shape but not, for example, the capitals of an allowed value
or the length of `problem` (at most 40 characters). What fails takes
one of three ways:

- **Repaired**: a wish that differs from an allowed one only in its
  capitals (`not Stated`) is repaired in code and checked again in full,
  and a `Repaired:` line says what changed. Nothing else is repaired: any
  other fault is a question about the email.
- **Asked again**: the model gets the email, its answer and what was
  wrong with it (the template `retry_ticket`), once by default
  (`--retries 1`; `--retries 0` turns it off). An `Asked again:` line
  says why, and the tokens and cost add up every call.
- **Rejected**: an answer that still fails names each field that broke
  the contract and exits with exit code 1, so whatever runs AskKit
  hands the email to a person.

A ticket that passes `Ticket` can still be wrong, so two business rules
in `python/src/askkit/rules.py` check it next:

- **The email**: the order number must be one of the numbers in the
  email. One that is not is the model's misreading (`4471` for `4417`),
  so it is asked for again, as a broken field is, within `--retries`.
- **The shop's orders**: with `--orders orders.json` (a JSON object
  whose keys are the shop's order numbers), the order must be one of
  them. A number the email itself gives but the shop does not have is
  not asked for again, since the same email gives the same number:
  `extract` says so and exits with exit code 1, for a person to check.

```bash
cd python
uv run askkit extract email.txt --orders orders.json
```

A ticket with no order number (`null`) has nothing for either rule to
check.

Before any of this, `extract` cleans the email
(`python/src/askkit/clean.py`): lines quoted from an earlier email, which
start with `>`, and the "On …, … wrote:" line right above them are left
out, and a `Left out:` line says how many. The model reads the cleaned
email, and the email rule checks against it, so a number that appears
only in a quoted earlier email, such as another order's, is asked for
again. Nothing else is cut: a signature or a P.S. may hold what the
ticket needs, and only the model can tell.

## Sorting an email into a queue

`askkit classify` puts a customer's email in one of the shop's four
queues: `damaged`, `refund`, `delivery`, or `other` for an email that
meets none of the first three conditions. The queues are `Queue` in
`python/src/askkit/classify.py`; the template `classify_email` defines
each one by what happened to the order, and the JSON schema made from
`Classification` holds the answer to one of the four values:

```bash
cd python
uv run askkit classify gift.txt
```

It prints the classification as JSON, then the same lines as `ask`. The
email is cleaned first, as for `extract`. A queue that differs from an
allowed one only in its capitals is repaired, and a `Repaired:` line says
so; any other answer is not asked for again, since the schema leaves
nothing else to go wrong, and `classify` exits with exit code 1 for a
person to sort the email.

## Prompts

Every prompt AskKit sends is a template file in
`python/src/askkit/templates/`, named `<name>.v<version>.txt`: the
words, with a `$name` placeholder where the code fills in a value on
each call (`$documents` and `$question` in `quote_request`). `IN_USE` in
`python/src/askkit/prompts.py` says which version of each is sent, and
`render(name, values)` fills one in, stopping with a `ValueError` when
the values and the placeholders do not match.

A version that has been sent is never edited. To change a prompt, copy
its file to the next version number, change the copy, write in
`templates/CHANGES.md` what changed and why, add its fingerprint to the
test, run the labelled examples on both versions, and then move `IN_USE`
to it. The old file stays, so an answer can be traced to the words that
made it, and the two versions can be compared. A test fails when a sent
version changes by one byte. A `$` meant as a dollar sign is written
`$$`.

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
- `m03`, end of Module 3 (Prompt Engineering): `ask --document` with
  quotes checked against the documents, `ask --effort`, and every prompt
  in a versioned template file.

## Layout

- `python/`: the Python app, a uv project (Python 3.14).
  - `src/askkit/`: the package.
  - `src/askkit/templates/`: the prompts, one file per version.
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
