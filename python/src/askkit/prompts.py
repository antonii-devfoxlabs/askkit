# AskKit's system prompt: the job and the rules for every question.
# QUOTE_REQUEST asks for every sentence that applies, word for word, so
# the word limit leaves those out: counting them would set the two rules
# against each other, and a quote shortened to fit fails the quote check.
ASK_SYSTEM = """You answer questions for AskKit, a command-line tool.
Write plain text without Markdown, because the terminal shows it as raw
symbols such as ** and #.
Keep the answer under 120 words unless the question asks for more.
Sentences you copy from documents do not count toward the 120 words.
If you are not sure of the answer, write exactly: I don't know."""

# Sent after the documents and before the question, so the model copies
# the sentences it rests on first, where code can check them. The check
# before the answer names what a whole answer covers: an exception missed
# still leaves every quote true.
QUOTE_REQUEST = """Answer the question below from the documents above.
First copy the sentences of the documents that your answer rests on, word
for word, each inside its own <quote></quote> tags. Before you write the
answer, check that you quoted every part of the documents that applies to
the question, such as a rule that makes an exception to another. Then
write the answer inside <answer></answer> tags. If the documents do not
answer the question, write exactly: I don't know."""
