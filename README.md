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
thinking, when the API says), what it cost, and how long it took: the
time to the first piece of text and the total time, both counted from
the moment the request is sent:

```bash
cd python
uv run askkit ask "In one sentence, what is a context window?"
```

Press Ctrl-C while an answer is still streaming to cancel it: AskKit
closes the connection, so no more of the answer comes. It prints how
long the answer ran (`cancelled at …`), says the text above is not a
whole answer and exits with code 1. A cancelled answer has no stop reason and no
usage, so no tokens or cost are printed, and it is never saved.

`ask` offers the model up to three tools, defined in
`python/src/askkit/tools.py`: a clock (`current_time`), because a model
has no clock of its own, and a calculator (`calculate`: one of `add`,
`subtract`, `multiply` or `divide` on two numbers `a` and `b`), because
a model writing a number token by token can get it wrong, and a
web fetch (`fetch_page`, below). The model
never runs anything: when an answer depends on today's date or needs a
sum, it stops with the stop reason `tool_use` and asks for a tool
instead of answering. Each definition's description says when to call
the tool and when not, and both are strict (`"strict": true`), so the
API holds the model's input to the tool's schema. AskKit still checks
every request (a tool it offered, with an input it can use: no input
for the clock; for the calculator, a known operation, two numbers that
are not `true` or `false`, and no division by zero), runs it, prints a
`Tool:` line with the result, and sends the result back with everything
before it, so the model can answer. A request that fails the check runs
nothing: the model gets an error result instead.

`ask` runs the tool loop (`python/src/askkit/loop.py`): call, run every
request, send the results back, and call again, until the model answers
instead of asking. A question whose second sum needs the first one's
result takes two rounds and three calls. Every request of every round
passes the checks and your permission below. After 5 rounds the next
call is sent with `tool_choice` `none`: the tools are still described,
but the model may not ask for one, so it answers with what it has
(`Tool limit:` says so). If it asks anyway, AskKit runs nothing and `ask`
stops with exit code 1. The calls' tokens and cost are added up, with
one `Time:` line per call.

By default the model decides for itself whether to ask for a tool.
`--use-tool TOOL` makes the first call ask for that tool
(`tool_choice` `{"type": "tool", "name": TOOL}`); every later call is
free again, so the model can answer once it has the result. A forced
call writes no text and does no thinking first, and the tool must be
offered: a denied tool, or `fetch_page` with no `--allow-site`, is
refused before anything is sent. The request still passes your
permission, so `--confirm` still asks.

When requests do not depend on each other, such as today's date and a
sum, the model can ask for several in one turn. AskKit runs them one
after another, in the order asked, and sends all their results back in
one message, each matched to its request by `tool_use_id`. A
`Tool requests:` line says how many requests ran in how many rounds:
`2 in 1 round` took two calls, `2 in 2 rounds` three.

Which tools AskKit may run for you is your decision, not the model's.
Each tool has a permission in `python/src/askkit/permissions.py`:
`allow` (run it), `ask` (ask you first) or `deny` (never offer it). The
clock and the calculator are allowed by default, since they change
nothing and send nothing anywhere. `--confirm TOOL` makes `ask` show you
the request exactly as it would run, such as
`Run calculate {"operation": "multiply", "a": 1284, "b": 0.000137}? [y/N]`
on stderr, and run it only on `y` or `yes`; anything else, or no answer
at all (input from a pipe, Ctrl-C), is a no. `--deny TOOL` leaves a tool
out of the call. Both may be given more than once; a tool both confirmed
and denied is denied. A request that is declined or not allowed runs
nothing: the model gets an error result that says so.

An error result is all the model learns about a failure, so every one
AskKit sends says what went wrong and what to do next: call again with
a fixed input (`current_time takes no input; call it again with an
empty input, {}`), or tell the user and not call again (`cannot divide
by zero; …`, a result too large for a number, a tool you declined or
did not allow). Nothing a tool raises stops AskKit: a fault in AskKit's
own code goes back as an error result that names only its kind, such
as `(OSError)`, never its text, which may hold your paths. Every error
result prints as a `Tool error:` line instead of `Tool:`.

`fetch_page` (`python/src/askkit/fetch.py`) downloads one page and sends
its text to the model. AskKit makes that request from your computer, so
it could reach whatever your computer can: your router, a service on
your own machine, a server at work. So it fetches only from the sites
you name with `--allow-site HOST` (given more than once for several;
with none, the tool is not offered), only `https://` addresses that name
the site alone, with no user name, password or port. It looks the name
up and refuses it if any address it leads to is not on the public
internet (`ipaddress`'s `is_global`: loopback, private, link-local such
as `169.254.169.254`, shared and reserved ranges all fail), then
connects to the address it checked, never to a second look-up of the
name, with the site's name in the `Host` header and the TLS handshake,
so the certificate is checked for the site. It follows no redirect: a
moved page goes back to the model as an error result naming the new
address, and the model's next request passes every check again. It
reads only text, at most 100,000 bytes, counted as they arrive (the size
a site states is checked too, never trusted), within 10 seconds for the
whole download and 5 for any one wait. Every refusal is an error result
that says what to do next; a fetched page prints as its size, such as
`Tool:          fetch_page: 36,371 characters`. Tests never reach the
network: every look-up and every site is a stand-in.

`--search-site DOMAIN` (given more than once for several) also offers
Anthropic's web search (`python/src/askkit/server_tools.py`). This tool
runs on Anthropic's servers, not in AskKit: the model asks, the API
searches, and the request, the results and the answer come back in one
reply, so no permission check of AskKit's ever sees the request. Its
limits go into the definition instead: results only from the domains
you name (`allowed_domains`), and at most 3 searches a call
(`max_uses`). Each search prints a `Search:` line as it runs, a failed
one a `Search error:` line with its error code; every page the answer
cites prints as a `Source:` line under it; and the searches are billed
on top of the tokens, $10 per 1,000, shown on a `Searches:` line and
added to `Cost:`. When the API pauses its own loop (`pause_turn`),
AskKit sends the paused turn back as it came, at most 3 times.

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

`--image FILE` sends a picture with the question: a screenshot of an
error, a chart, a photo. Give it once per image. AskKit reads the
file's first bytes to tell its format (PNG, JPEG, GIF or WebP, the four
Anthropic's API reads; anything else is refused before a call), sends
it as base64 text in an `image` block, and puts the images before the
question, each labelled "Image 1:", "Image 2:" when there are several.
`python/samples/` holds two images made for the course: a terminal
with a traceback, and a chart of invented daily costs:

```bash
cd python
uv run askkit ask "On which day did AskKit cost the most, and about how much?" --image samples/chart.png
```

A saved history (`--history`) holds text only, so `--image` and
`--history` are not taken together.

`--document FILE` sends a long text with the question: a policy, a
contract, a manual. Give it once per file, as many times as you need.
Each file goes into the user message in `<documents>` tags, numbered and
named after its path, then a request to quote first, then the
question, last; the template `quote_request` holds that layout. A
`--document` is read as text, so a PDF is refused before a call (its
first bytes say what it is): an invoice goes to `askkit read`.

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
on Windows); Ctrl-C at the `You:` prompt stops it too. It keeps the conversation in
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
and cost on a `Turn N:` line; they grow with the conversation. A `Time:`
line under it gives the turn's time to first token and total time; a
turn that is not kept prints its `Time:` line too. Ctrl-C while an
answer streams cancels only that answer: it is not kept, its `Time:`
line says when it was cancelled, and the chat asks for the next
question.

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

The answer streams, as in `ask`: the SDK refuses an ordinary call whose
`--max-tokens` could take over ten minutes to write, and a stream also
shows the ticket filling in. Each field is printed on the error output
once it has **settled**, once a comma or the closing brace follows it,
as a `Settled:` line: a preview for a person, which nothing in AskKit
uses (`settled_fields` and `stream_json` in
`python/src/askkit/structured.py`). The ticket itself is read only from
the whole answer, once the stream has ended, and only after the checks
below. Ctrl-C while it streams cancels it, and no ticket is printed.

It prints the ticket, then the same lines as `ask`, with a `Time:` line
for each call. It reads the stop reason first: an answer cut off at `--max-tokens` (1000 by default) or
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

It streams as `extract` does, prints the classification as JSON, then
the same lines as `ask`, with a `Time:` line. The email is cleaned first, as for `extract`. A queue that differs from an
allowed one only in its capitals is repaired, and a `Repaired:` line says
so; any other answer is not asked for again, since the schema leaves
nothing else to go wrong, and `classify` exits with exit code 1 for a
person to sort the email.

## Reading an invoice

`askkit read FILE` takes the fields out of a supplier's invoice sent as a
PDF: the seller, the invoice number, the dates it was issued and is due,
the currency, the subtotal, the tax and the total. The whole PDF goes in
the call as base64 text in a `document` block, alone in the user
message; the API turns each page into an image and takes out its text,
and the model reads both. The fields are `Invoice` in
`python/src/askkit/invoices.py`, whose JSON schema is sent beside the
template `read_invoice`:

```bash
cd python
uv run askkit read samples/invoice.pdf
```

It streams as `extract` does, prints the invoice as JSON, then the same
lines as `ask`. A file whose first bytes are not a PDF's (`%PDF-`) is
refused before a call. After `Invoice` has passed, two rules check the
fields against each other: the subtotal and the tax add up to the total,
in whole cents, and the due date is not before the issue date. When one
fails, `read` prints no invoice and exits with exit code 1: a misread
number and an invoice that is wrong itself look the same to the check,
so a person reads the PDF. `python/samples/invoice.pdf` is a two-page
invoice from an invented company, made for the course.

## Transcribing a voice message

Anthropic's API takes text and images, not sound, so `askkit transcribe
FILE` turns a voice message into text with OpenAI's transcription model,
`gpt-transcribe`, and prints it with the call's tokens, the recording's
length and what it cost (estimated at $0.0045 a minute, OpenAI's pricing
page, checked on 20 November 2026). It needs an OpenAI key in
`python/.env` (`OPENAI_API_KEY`). `--out FILE` also saves the text, so a
voice message becomes a ticket the way an email does:

```bash
cd python
uv run askkit transcribe samples/voicemail.wav --out voicemail.txt
uv run askkit extract voicemail.txt
```

The file's format is told by its first bytes (WAV, MP3, M4A, Ogg, FLAC
or WebM), and a file that is none of them, or over the 25 MB a call
takes, stops before the call. Only a WAV gives its length in its header,
so only a WAV gets a cost. `python/samples/voicemail.wav` is an
11-second voice message made for the course with a synthetic voice
(macOS `say`): Maria, an invented customer, about order 4417.

## What an image, a PDF or a recording costs

`askkit estimate` takes an image, a PDF or a voice message as well as
text, tells which by the file's first bytes, and says what the model would
read from it and where it stops, without a call:

```bash
cd python
uv run askkit estimate samples/chart.png
uv run askkit estimate samples/invoice.pdf
uv run askkit estimate samples/voicemail.wav
```

- An image costs one token for each square of 28 x 28 pixels. Its width
  and height come from its header (with Pillow). Every Claude model
  AskKit knows is on the high-resolution tier of Anthropic's vision
  guide: an image with a side over 2,576 px, or one over 4,784 tokens,
  is shrunk by the API to the largest size that fits, and billed at that
  size. Over 8,000 px a side, or over 10 MB as base64, it is refused.
- A PDF costs its text, estimated from its characters (taken out with
  pypdf), plus at most 1,568 tokens for each page's image. Over 600
  pages (100 when the model's window is under a million tokens), or over
  128 MB as base64, it is refused.
- A recording is never sent to the model as sound: it costs its minutes
  at OpenAI (only a WAV gives its length), and the model named with `-m`
  reads the transcript.

Image and PDF tokens are counted for Claude only: every provider counts
them its own way. A HEIC photo, which iPhones often save, is named as
one (it starts the way an M4A recording does) and refused: Claude's API
does not read it. `ask --image` and `read` stop at the same limits
before any call. The limits and the image rules were checked on 20
November 2026.

## What each feature costs a month

The bill says what a month cost by model, workspace and API key, not by
what people did with the app. `askkit costs` works it out per feature
from a cost model: a JSON file that lists each feature with its uses a
month, the calls one use makes, the input and output tokens of those
calls added up (read from the usage of real runs), and any fees a use
pays on top of the tokens, such as a web search or a recording written
down at another provider. No call is made:

```bash
cd python
uv run askkit costs samples/month.json
```

It prints every feature, the most expensive first, with what one use and a
month of uses cost and its share of the month. Lesson 1.2's price step
is checked on an average call of a use, since it goes by each call's
prompt. `python/samples/month.json` is a month made up for the course:
the uses are invented, and the tokens are those of the runs the lessons
show.

## What the cache saves

Every call of `ask` sends the same tools and system prompt, and the
second call of a tool loop sends the whole first one again. AskKit marks
where that repeated opening ends with cache breakpoints: one on the
system prompt (the tools come before it, so it keeps both), one after
the documents of `--document`, and one on the question. Within 5
minutes, a call that starts with the same bytes reads that opening from
Anthropic's cache for a tenth of the input price; the first call writes
it for 1.25 times the price. `askkit chat` lets the API move a
breakpoint to the end of every call, so each turn reads the turns before
it. A prompt under 512 tokens on Claude Haiku 5.5 is not cached at all.

Every command that calls a model prints a `Cache:` line when the cache
read or wrote anything, and its `Tokens:` and `Cost:` count each kind of
input at its own price.

## Answers kept on your side

`askkit read` and `askkit classify` keep every answer that passed their
checks, and give it back when the same request comes again: nothing is
sent and nothing is paid. The key is a SHA-256 hash of the whole request
as it would be sent (`json_request` in
`python/src/askkit/structured.py`): the model ID, `max_tokens`, the
template's text, the schema and the email or the PDF. Change any of them
and the request gets a key of its own. The answers are files in
`.askkit-cache/` in the folder you run AskKit from, left out of Git,
since an answer can hold what a customer wrote; the request itself is not
stored, only its key. An answer is kept for 30 days (`MAX_AGE` in
`python/src/askkit/response_cache.py`), then asked for again, and
every answer kept removes the files past that age. A kept
answer goes through the same checks as a new one, and an answer that
stopped early or failed a check is never kept. `--fresh` asks the model
anyway and keeps the new answer in place of the old one:

```bash
cd python
uv run askkit read samples/invoice.pdf
uv run askkit read samples/invoice.pdf
```

The second run prints the same invoice, then
`Answer:        kept on <date> UTC; nothing was sent` and what it saved.
The eval never reads the cache: it calls the model through its own
functions, so two runs of it are two samples.

## Every call, in a log

Every call AskKit makes to a model is written down the moment its answer
arrives, before it is checked: one line of JSON a call in
`.askkit-log/calls-<day>.jsonl` in the folder you run AskKit from, with
the command, the model, the tokens (plain, read from the cache and
written to it, and out), the seconds, the cost at that day's prices, the
stop reason, the retries before it and the API's request ID. The line
holds no word of the question or the answer. `--log-content`, given
before the command, also keeps what was sent and what came back, in
`.askkit-log/content-<day>.jsonl`, for 7 days (`CONTENT_MAX_AGE` in
`python/src/askkit/calllog.py`); an image or a PDF is named, never
copied. A kept answer from `.askkit-cache/` makes no call and no line.
The folder is left out of Git. `askkit day` adds a day up, by command and
by model; a day is a UTC day:

```bash
cd python
uv run askkit --log-content ask "What is 18% of 4,417?"
uv run askkit day
uv run askkit day 2026-11-20
```

The eval calls the model through its own functions and logs nothing.

## Sending a ticket to a larger model

Claude Sonnet 5.5 costs twenty times Claude Haiku 5.5 a token, so
`extract` starts every email on Haiku 5.5 and can send only the ones it
got visibly wrong to Sonnet 5.5:

```bash
cd python
uv run askkit extract email.txt --escalate-to claude-sonnet-5-5
```

The first call goes to `--model`. When its ticket fails `extract`'s own
checks (the `Ticket` contract, or an order number the email does not
give), the one retry goes to the model `--escalate-to` names, with the
failed answer and its faults, and an `Escalated:` line says so. Each
call is priced at its own model's prices, and `Model:` names both. A
ticket that passes the checks but is wrong is not escalated: no code
sees it. `src/askkit/routing.py` holds the two models and the decision.

The eval scores the same route: `uv run pytest evals --escalate-to
claude-sonnet-5-5 --results routed.json` asks the larger model again
wherever `extract` would, scores the answer `extract` would print, and
ends with how many cases were escalated, the share the route's cost
depends on. A results file keeps the model escalated to, and `askkit
diff v2.json routed.json` names it as the change. `--escalate-to
claude-haiku-5-5` scores the plain retry on the same model, the
cheaper thing to beat.

`askkit compare` still prices the high-volume model of each provider;
Sonnet 5.5 is in `prices.py` and `limits.py` for routing.

## Retries and timeouts in a run of questions

`askkit answer` sends one call per question, and a question that fails
used to end the run with the SDK's three tries spent in about a second
and a half. It now retries each question's call itself:

```bash
cd python
uv run askkit answer evals/ask/main --out answers.json --tries 5 --timeout 60
```

A call that fails with a 408, 409, 429, a status of 500 or more, a
timeout or no connection is sent again, up to `--tries` requests in all
(5 by default), and no retry starts more than two minutes after the
first send. Before each retry AskKit waits a random time between nothing
and a limit that doubles from one second up to thirty ("full jitter"),
or exactly as long as the API's `retry-after` header asks. The SDK's own
retries are off for these calls, so one try is one request. Each request
may wait `--timeout` seconds (60 by default) for its answer; a
`--max-tokens` that could take longer at the SDK's slowest pace (128,000
tokens an hour) is refused before any call. A run that retried prints a
`Retries:` line with the reasons and the time spent waiting. Only the
failed question is sent again, never the questions before it.
`src/askkit/retry.py` holds the policy.

## Falling back to another model or provider

When a question's tries are used up, `--fallback` sends it to the next
model of a chain instead of ending the run. Give it once per model, in
order:

```bash
cd python
uv run askkit answer evals/ask/main --out answers.json --fallback gpt-6-luna
```

A model hands a question on only for what another model can fix: an
overloaded or failing provider, a rate limit, a timeout or no
connection. A rejected request (400) or key (401) ends the run as before,
since every model would reject it too. A declined answer goes on only to
another Claude model (`--fallback claude-sonnet-5-5`), never to another
provider, and not at all for the `reasoning_extraction` category. A model
that used up its tries is passed over for 60 seconds, so the questions
after it do not wait for it again. `gpt-6-luna` runs at OpenAI, through
its Responses API, with `OPENAI_API_KEY` from `.env` and `store=False`;
its answer is turned into the same shape as Claude's, so the same checks
read it and it is priced at its own prices. Each saved answer names the
model that wrote it, the question's line names a fallback model, and a
`Fallbacks:` line says which questions moved and why. Before you rely on
a fallback, answer the whole set with it first (`-m gpt-6-luna`) and
grade it. `src/askkit/fallback.py` holds the chain.

## Sorting a whole inbox

`askkit inbox FILE` sorts every email of a CSV file into the queues of
`askkit classify`, one email at a time. The file has two columns, `id`
and `email`, named on its first line; an email may hold commas and line
breaks inside double quotes. `python/samples/inbox.csv` holds 1,000
emails made up for this course (invented names and order numbers),
written by `python/samples/make_inbox.py`. `--first N` sorts only the
first N, to measure before paying for all:

```bash
cd python
uv run askkit inbox samples/inbox.csv --first 12
```

It prints each email's queue, then the run: the tokens and the cost, the
time in all and per email, the share of that time AskKit spent waiting
(the wall clock, `perf_counter`, against the CPU clock, `process_time`),
and how long the whole file would take at that pace. An answer that
stopped early or fails its check goes to a person, and the run goes on;
a rejected request stops it and names its email. Every call is a line
of the log, under the command `inbox`.

## The models AskKit calls, and when each retires

Every model ID AskKit calls is a pinned snapshot, and the one every
command starts with, `DEFAULT_MODEL`, is set in one place:
`src/askkit/models.py`. The same file keeps, for each model, the day its
provider retires it, read from each provider's deprecations page:

```bash
cd python
uv run askkit models
```

A date shown as `not before` is only the earliest day the provider could
retire the model. Once a provider announces the day, enter it in the
model's row with the replacement it recommends. From then on every
command that calls the model names the day on stderr, and from that day
stops before its first call, the last model of a `--fallback` chain
included. `askkit models --live` also asks each
provider's models endpoint (Anthropic's `lifecycle` and `retires_at`,
OpenAI's `shutdown_date`) and exits with 1 when a row no longer matches.
Before you move `DEFAULT_MODEL` to the replacement, run the eval on it.

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
- `m07`, end of Module 7 (Streaming and Latency): `ask`, `chat`,
  `extract` and `classify` stream their answers and print the time to
  the first token and the total time; Ctrl-C cancels only the answer
  that is streaming; `extract` and `classify` preview each settled
  field and use the answer only once it is whole and checked.
- `m08`, end of Module 8 (Tool Use): `ask` offers a clock, a calculator
  and a web fetch that reaches only the sites you allow, never a
  private address, and reads at most 100,000 bytes; it runs them in a
  loop of at most 5 rounds, each request checked against your
  permissions first, with error results the model can act on; it can
  force the first call's tool, and with `--search-site` it offers
  Anthropic's web search, run on Anthropic's servers.
- `m09`, end of Module 9 (Beyond Text): `ask --image` sends screenshots
  and charts, `read` takes the fields out of an invoice PDF and checks
  them, `transcribe` turns a voice message into text at OpenAI, and
  `estimate` prices an image, a PDF or a recording and stops at the
  API's limits before a call.
- `m10`, end of Module 10 (Cost, Speed and Reliability): `costs` prices a
  month by feature; prompt caching on the parts every call repeats;
  `extract --escalate-to` sends a retry to a larger model; `answer`
  retries with backoff, a deadline and a timeout, and falls back to
  another model or provider; every model has a retirement date that
  stops or warns before a call; `read` and `classify` keep the answers
  they paid for; every call is logged with its model, tokens, latency
  and cost, and `day` adds up what a day cost.

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
  - `src/askkit/structured.py`: a JSON answer streamed, with the fields
    that have settled so far.
  - `src/askkit/tools.py`: the tools `askkit ask` offers the model (a
    clock, a calculator and a web fetch), their definitions, the check each request
    passes before anything runs, and the error results the model gets
    when one fails.
  - `src/askkit/server_tools.py`: Anthropic's web search, which runs on
    Anthropic's servers: its definition, the lines it prints, its
    searches counted and the pages the answer cites.
  - `src/askkit/images.py`: an image file as an `image` block, its
    format told by its first bytes, placed before the question; its
    tokens, the size the API shrinks it to, and the API's limits.
  - `src/askkit/pdfs.py`: a PDF as a `document` block, told by its
    first bytes; its pages' text, and the API's limits.
  - `src/askkit/inputs.py`: what `estimate` prints for an image or a
    PDF: what the file is and the tokens Claude reads from it.
  - `src/askkit/invoices.py`: the fields of an invoice, and the rules
    that check them against each other.
  - `src/askkit/fetch.py`: the web fetch, which reaches only the sites
    you allow, never a private address, and reads at most 100,000 bytes
    of text within 10 seconds.
  - `src/askkit/permissions.py`: which tools AskKit may run for you
    (allow, ask first, or deny), and the check every request passes
    before it runs.
  - `src/askkit/loop.py`: the tool loop, which calls the model, runs
    what it asks for and sends the results back until it answers, for
    at most 5 rounds, sets each call's `tool_choice`, and counts the
    requests and rounds a question took.
  - `src/askkit/audio.py`: a voice message turned into text at OpenAI:
    its format told by its first bytes, its length read from a WAV's
    header, and what the call cost.
  - `src/askkit/caching.py`: the cache breakpoints AskKit puts on the
    system prompt and the question.
  - `src/askkit/routing.py`: the small and the large model, and when a
    ticket goes to the large one.
  - `src/askkit/fallback.py`: when a question moves on to the next
    model of a chain, and OpenAI's answer in Claude's shape.
  - `src/askkit/models.py`: the default model, and the day each model
    AskKit calls retires.
  - `src/askkit/response_cache.py`: the answers `read` and `classify`
    keep, by a hash of the whole request, for 30 days.
  - `src/askkit/retry.py`: which failed calls are sent again, how long
    AskKit waits before each retry, and the timeout of `askkit answer`.
  - `src/askkit/costmodel.py`: a cost model of the app's features, and
    what each one costs a month.
  - `src/askkit/calllog.py`: the log of every call, and a day added up.
  - `src/askkit/inbox.py`: an inbox CSV file read, and the two clocks of
    a run of many emails.
  - `samples/`: images, an invoice PDF, a voice message, a month of uses
    and an inbox of 1,000 emails (with the script that writes it), made
    for the course, with invented data.
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
