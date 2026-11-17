# Prompt versions

Every prompt AskKit sends is a template file here, named
`<name>.v<version>.txt`. `IN_USE` in `../prompts.py` says which version
is sent. A version that has been sent is never edited: a change is a new
file with the next number, and a line below says what changed and why.
`$name` is a placeholder the code fills; `$$` is a dollar sign.

## ask_system

The system prompt sent with every question: AskKit's job and its rules.

- v1: the job, plain text without Markdown (the terminal shows `**` and
  `#` as they are), under 120 words, and "I don't know" in exact words
  when the model is not sure.
- v2: sentences copied from documents do not count toward the 120 words.
  `quote_request` asks for every sentence that applies, word for word;
  counting them set the two rules against each other, and a quote
  shortened to fit failed the quote check.

## quote_request

The user message for a question about documents: `$documents` first, then
the request to quote, then `$question` last.

- v1: the documents first and the question last; the model copies the
  sentences its answer rests on into `<quote>` tags before the
  `<answer>`, so code can check them, with the same way out as
  `ask_system`.
- v2: a check before the answer that every part of the documents that
  applies was quoted, an exception to a rule included. With v1 the model
  quoted the general rule and missed the exception that overrode it, and
  every quote still passed the check.

## extract_ticket

The system prompt of `askkit extract`: what each field of the ticket
means and which value says the email does not give it. The shape itself
is the class `Ticket` in `../extract.py`, whose JSON schema is sent
beside the prompt.

- v1: the three fields, `order`, `problem` and `wish`, each with its
  meaning, and the way out of each: `null` for no order number, `not
  stated` for no wish.
- v2: for emails that hold more than the ticket needs, such as a reply
  with a phone number, a P.S. and an earlier order mentioned in passing.
  v1 did not say which number is the order, what to write when two
  things went wrong, or which wish counts when the email asks for two.
  v2 says all three: the order this email is about, every thing that
  went wrong with the most serious first, and the wish asked for first.
- v3: the problem under 40 characters, the length `Ticket` holds. v2's
  "name each" made one problem 43 characters long for an email where three
  things went wrong, and `Ticket` rejected the ticket. Not in use yet: it
  goes in when a comparison with v2 on the golden set shows what it fixes
  and that it breaks nothing (see "Comparing two versions" in the README).
- v4: which thing the problem starts with. Reading every failed answer
  of v2's and v3's runs with `askkit faults`, most had one cause: the
  problem kept a lesser detail and lost the customer's main complaint,
  "crumb tray missing" for a toaster that sparks and, under v3's limit,
  "blanket not clean" for one that smells of smoke. v2 and v3 asked for
  the most serious first without saying what counts as serious; v4 says
  it (a danger, otherwise what the email says most about) and adds the
  rest only if it fits. Not in use yet, as v3: it goes in when a
  comparison with v2 shows what it fixes and that it breaks nothing.

## retry_ticket

The user message `askkit extract` sends when a ticket broke its contract
and is asked for again: `$faults` is what `Ticket` found wrong, one part
per field. It follows the email and the answer that failed.

- v1: names what was wrong and asks for the ticket again, as the system
  prompt describes it.

## classify_email

The system prompt of `askkit classify`: the condition of each queue the
email can go to. The queues themselves are `Queue` in `../classify.py`,
whose values the JSON schema sent beside the prompt allows.

- v1: the four queues, `damaged`, `refund`, `delivery` and `other`, each
  defined by what happened to the order, `other` for an email that meets
  none of the three conditions, and the rule to decide by what happened,
  not by the words the email uses.

## summarise_turns

The system prompt of the call that folds a chat's oldest turns into a
summary, when a turn of `askkit chat` would not fit its budget.

- v1: keep every fact the user may come back to word for word (names,
  numbers, prices, order numbers, decisions), leave out greetings and
  repeats, plain text under 100 words.

## fold_turns

The user message of that call: the `$summary` so far and the `$turns` to
fold in, each in its own tags, then the request for one summary.

- v1: the summary so far first, the turns after it, oldest first.

## earlier_turns

The system prompt of a chat turn once old turns were summarised: the
chat's own `$system` prompt, then the `$summary` in tags.

- v1: the summary after the chat's rules, under a line that says what it
  is.

## remembered

The system prompt of every `askkit chat` turn once the user has notes in
memory: the chat's own `$system` prompt, then the `$notes`, one per line,
in tags.

- v1: the notes after the chat's rules, under a line that says the user
  asked for them and that the user's latest message wins over a note.
