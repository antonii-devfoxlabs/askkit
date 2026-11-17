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
prints the answer as the model writes it, streamed piece by piece, then
the tokens the call used (and how many of the output tokens were
thinking, when the API says) and what it cost:

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
`ask` says so and exits with exit code 1. A declined answer may have
shown some text before the model stopped: `ask` then says not to use it,
as Anthropic advises, and names the category when the API
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

`--history FILE` keeps a conversation going from one run to the next.
The API keeps nothing between two calls, so AskKit keeps the
conversation itself: it reads the messages saved in the file (none
the first time), sends them before the new question, and after a whole
answer saves the question and the answer's text back to the file, as
JSON a person can read. A cut-off or declined answer is not saved, and
a file whose messages are out of turn is refused before any call:

```bash
cd python
uv run askkit ask "Order 4417 came with a broken mug. Can I get a new one?" --history chat.json
uv run askkit ask "And who pays the postage?" --history chat.json
```

Every run sends the whole conversation again, so each one's input
tokens, and its cost, grow with it. `History:` under the stop reason
says how many messages the file now holds.

`askkit chat` asks question after question in one run, until you type
`/exit` or end the input (Ctrl-D on macOS and Linux, Ctrl-Z then Enter
on Windows); Ctrl-C stops it too. It keeps the conversation in
`chat.json` by default (`--history FILE` for another), reads it when it
starts, and sends the whole conversation with every question. It takes
the same `-m`, `--max-tokens`, `--effort` and `--system-file` as `ask`:

```bash
cd python
uv run askkit chat
```

Each answer is streamed, as in `ask`. A turn joins the history only once
its answer is whole (an answer whose stream breaks off midway is not),
and the file is written after every such turn, so stopping at any moment
loses at most the answer still on its way. An API error, a cut-off or a declined
answer prints `askkit:` and why, keeps nothing, and the chat goes on, so
the same question can be asked again. Each whole turn prints its tokens
and cost on a `Turn N:` line; they grow with the conversation.

Every turn of a chat is held to a budget: the most input tokens one call
may send, 90,000 unless you give `--budget N`. Before each call AskKit
counts what it would send with the API's free token count; a turn over
the budget is not sent, and AskKit says so. The default stays under the
100,000-token prompt above which Claude Haiku 5.5 bills the whole call
at five times the price. `--budget` and `--max-tokens` together must fit
the model's context window, or the chat does not start:

```bash
uv run askkit chat --budget 20000
```

When a turn would go over the budget, AskKit first makes room. By
default (`--old-turns summarise`) it folds every turn but the last two
into a short summary, with one call to the same model at low effort, and
sends the summary in the system prompt in their place; a later fold
takes the summary so far in with the next turns. `--old-turns drop`
leaves the oldest turns out instead, one at a time, until the turn
fits: no call, but what they said is gone. Either way the last two
turns go word for word, `chat.json` keeps every turn, and a turn that
still does not fit is not sent. The summary lives only while the chat
runs; a chat started again on the same file summarises anew when it
needs to:

```bash
uv run askkit chat --budget 20000 --old-turns drop
```

What should outlast a conversation goes in memory: short notes you ask
AskKit to keep, in `memory.json` by default (`--memory FILE` for
another), shared by every chat file. Every turn sends them in the
system prompt, after AskKit's rules, with the line that your latest
message wins over a note. Three commands in a chat look after them, and
none of them calls the model:

- `/remember TEXT` adds a note and saves it at once. A note holds up to
  200 characters, memory up to 20 notes, and a note with an API key in
  it is refused: notes go with every call.
- `/memory` lists the notes by number.
- `/forget N` takes note N out.

Only you write memory: the model never adds a note by itself. The
notes are plain JSON you can open, and `memory.json` is in `.gitignore`.

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

A field that holds its way out, an order of `null` or a wish of
`not stated`, is a true answer: the email does not give it. The ticket
is printed, and a `Not stated:` line names those fields, so whoever acts
on the ticket knows what to ask the customer.

A declined answer (`stop_reason: refusal`) is never read, not even the
part of a ticket written before the decline. `extract` and `classify`
say what was declined ("the model declined to turn this email into a
ticket"), with the refusal's category and the API's explanation when it
gives them, and exit with exit code 1. The same email is not sent to the
same model again: it would most likely be declined again. A person reads
it, or runs it again with another model (`-m`).

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

## The golden set

`python/evals/extract/` holds the golden set for `extract`: emails, each
with what a right ticket for it must hold, written by a person from the
email, never copied from a model's answer. It has two parts, each a
folder with a `cases.json` and one `.txt` file per email:

- `main/`: 16 cases, the ones you read and change the prompt for.
- `held_out/`: 8 cases, drawn before any prompt was tuned and never
  read while one is. They are run once a version is ready, to see
  whether it works on emails it was not written for.

A case names its email by the file without `.txt` and gives `order`
(`null` when the email has none), `wish`, `mentions` (the patterns
its ticket's problem must match), its `kind` and `why` it is in the
set. The kinds: `plain` (one order, one problem, one wish), `no
order`, `no wish` (no word on a replacement or a refund), `two wishes`
(the first one counts), `quoted` (an earlier email quoted with it),
`look-alike` (a phone number, a date or a quantity beside the order
number), `problems` (several things went wrong) and `language` (not
written in English). Every kind has cases in both parts. `load_cases(folder)` in
`python/src/askkit/golden.py` reads one part and stops with a
`ValueError` on a case without its email, an email without a case or a
name used twice; an expected value `Ticket` would refuse stops it too.

`askkit cases evals/extract` (run in `python/`) prints how many cases
of each kind are in each part, and exits with exit code 1 when a kind
has no case in one of them. It reads only the two `cases.json` files:
checking the set's shape does not open the held-out emails.

A held-out case whose result you read and tune for is no longer held
out: move it to `main/` and write a new one in its place. A test holds
a fingerprint of `held_out/`, so every change to it is made on purpose.

`case_faults(answer, case)` in `python/src/askkit/checks.py` checks one
answer against its case and returns every fault, an empty list for a
pass. The schema comes first (the answer must be a `Ticket`, repair
included, as `extract` checks it); then the order and the wish must
match exactly; then each of the case's `mentions` must be found in the
problem by `re.search`, in any capitals (`|` between words means any of
them); and the problem must not hold an order number, four digits in a
row. A test holds a right ticket, written by a person, for every main
case: a check that fails one of them is stricter than the job.

## Running the eval

`python/evals/test_extract_cases.py` is the eval: one pytest test per
case of the golden set, each sending its email to the model as
`extract` does on its first call (the cleaned email, the prompt in use,
the stop reason read first, no retry) and asserting that
`case_faults` finds nothing. `run_case` in `checks.py` is that one case.
In `python/`:

```bash
uv run pytest evals
```

runs the 16 main cases, one billed call each to Claude Haiku 5.5, and
ends with the score as pytest counts it, such as `3 failed, 13 passed,
8 deselected`; each failure names its case and its faults.

- `uv run pytest evals -k clock` runs one case, `-k "clock or pans"`
  two.
- `uv run pytest evals -m held_out` runs the held-out part. Its tests
  carry the `held_out` marker, which `pyproject.toml` deselects on every
  other run, so it runs only when a version is ready.
- Without `ANTHROPIC_API_KEY` every case is skipped, never failed.

### Recorded answers

`uv run pytest evals --record` runs the eval as usual and saves every
answer the API gives in `python/evals/extract/recording.json`: the body
of each request beside the body of its answer, never the headers, so
the key stays out of the file. `uv run pytest evals --replay` then runs
the eval from that file, with no key, no call and no cost; a request
the file has no answer to (a changed prompt, email or model) fails
with `recording.json has no answer to this request; record it again
with --record`. Replay a recording to try a change to a check on real
answers for free; record again after a change to a prompt.
`src/askkit/recordings.py` holds both transports.

Run it after every change to a prompt template. `uv run pytest` and CI
run `tests/` only, so they never call the model; the eval is apart
because it costs money, needs a key and scores a model that can answer
differently from one run to the next.

### Comparing two versions

`--prompt-version 3` scores the file `extract_ticket.v3.txt` instead of
the version in `IN_USE`, and `--model claude-sonnet-5-5` another model
(Claude Haiku 5.5 by default). `--results v3.json` adds the run's faults
to a results file; run the same command again and the file holds two
runs, so a case that passes one and fails the other shows as the model's
chance, not the version. A file takes runs of one model and one prompt
version over the same cases only, and never from `--replay`, which
repeats one run. `askkit diff v2.json v3.json` then reads two files side
by side, with no call to the model: every case that failed a run of
either, how many of its runs passed, and `fixed` or `broken` only when
every run of both agrees (`unsure` when one version's runs disagree),
then each run's score. Change one thing between the two files, and put
a version in `IN_USE` only when the diff shows what it fixed and that
it broke nothing. Then, once, check the choice on the held-out part:
two runs of each version with `-m held_out`, each version into a results
file of its own. `src/askkit/runs.py` holds the results file and the
diff.

### Reading the failures

A diff names the cases that changed; it does not say why they fail.
`askkit faults v2.json v3.json` reads every failed answer in one or more
results files of the main part, with no call to the model, and sorts
them by what code can tell: the fields each fault names (the words before
the colon of each part, `stop` for an answer cut off or declined) and the kind
of the case's email, the largest group first. Then it lists every failed
answer with its faults, to be read beside its email. Write a note on
each one, saying what went wrong in your own words; group the notes that
share a cause, count each group, and write the next prompt version
against the largest. A results file of the held-out part is refused, so
its failures stay unread. `src/askkit/analysis.py` holds the sorting.

## Grading answers by hand

An answer of `askkit ask` has no field to compare: it can be right in
many wordings, and wrong in a way no word gives away. `python/evals/ask/`
holds a golden set of questions for it, graded by a person against a
rubric:

- `documents/`: the shop's returns policy and three orders, the files
  the questions are asked about.
- `main/` and `held_out/`: a `cases.json` each (6 and 2 questions). A
  case names the question, the documents it is asked with, the facts a
  right answer `must` state, in words for the grader, and `why` it is
  in the set. The held-out part is kept apart as the extract set's is,
  with a fingerprint in `tests/test_rubric.py`.
- `rubric.json`: the criteria every answer is graded on. Each is a
  question the grader answers yes or no, with an answer that `passes`
  and one that `fails`, so two graders read it the same way.

Grading takes two commands, run in `python/`:

```bash
uv run askkit answer evals/ask/main --out answers.json
uv run askkit grade answers.json --cases evals/ask/main --rubric evals/ask/rubric.json --out grades.json
```

`answer` sends every question of one part as `ask --document` would,
and saves the answers. It runs `ask`'s own checks first (a whole
answer, quotes found in the documents, a quote unless the answer is
"I don't know") and saves what fails as the answer's faults. These are
real, billed calls, one per question. `grade` shows the rubric, then
each answer with its question and its must-list, and asks each
criterion in turn: `y` or `n`, and for every `n` a note saying why. An
answer with faults from `answer` is failed on every criterion without
being shown: a person's time goes to what code cannot check. The grades
are saved, and `grade` prints how many answers passed each criterion and
how many passed all of them. The answers file names no prompt version,
so whoever grades it does not know which version wrote it.

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
- `m04`, end of Module 4 (Structured Output): `extract` and `classify`,
  which return objects checked against a Pydantic model and the shop's
  rules, repair or ask again where one right reading or a reread can
  fix them, and say what a person must do with a declined, cut-off or
  partly stated answer.
- `m05`, end of Module 5 (First Evals): a golden set for `extract` with a
  held-out part, code-based checks, a pytest eval run after every prompt
  change, answers recorded and replayed, `diff` to compare two versions
  or models case by case, and `faults` to read the failures; `answer`
  and `grade` for grading `ask` answers by hand against a rubric.
- `m06`, end of Module 6 (Conversations and Context): `ask --history`
  and `chat`, which keep a conversation in a file, hold every turn to a
  budget of input tokens, summarise or drop the oldest turns to stay
  within it, and send the notes you asked AskKit to remember with every
  chat.

## Layout

- `python/`: the Python app, a uv project (Python 3.14).
  - `src/askkit/`: the package.
  - `src/askkit/templates/`: the prompts, one file per version.
  - `src/askkit/history.py`: a conversation saved in a file of its own.
  - `src/askkit/chat.py`: the chat loop of `askkit chat`.
  - `src/askkit/budget.py`: the chat's budget of input tokens a turn.
  - `src/askkit/trim.py`: dropping or summarising a chat's oldest turns.
  - `src/askkit/memory.py`: the notes every chat sends, kept between
    chats.
  - `notebooks/`: notebooks for exploring, saved without outputs.
  - `evals/`: the golden sets, a main and a held-out part each, the
    ask set's documents and rubric, and the eval that runs the extract
    set through the model (`uv run pytest evals`), with the version and
    the model to score chosen on the command line.
  - `tests/`: pytest tests. They never call a model's API: the real SDK
    sends each call to a fake transport (`httpx2.MockTransport`) that
    answers with a response written in the shape of the API's.

## Checks

```bash
cd python
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests evals
uv run pytest
```

CI runs the same four on every push.
