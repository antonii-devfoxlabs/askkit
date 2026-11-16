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

## retry_ticket

The user message `askkit extract` sends when a ticket broke its contract
and is asked for again: `$faults` is what `Ticket` found wrong, one part
per field. It follows the email and the answer that failed.

- v1: names what was wrong and asks for the ticket again, as the system
  prompt describes it.
