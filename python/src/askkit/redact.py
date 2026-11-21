import re

# The personal data a pattern can find, each with the placeholder that
# takes its place. The placeholder names the kind, so a reader of the
# log still sees that a card number was there, and a model still reads
# a sentence that makes sense. The card comes first: its 16 digits would
# also match the phone pattern and be named a phone.
PATTERNS: tuple[tuple[str, str], ...] = (
    # Four groups of four digits, with a space, a dash or nothing between.
    ("[CARD]", r"\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}"),
    # A name, @, a domain with at least one dot, ending on a letter or a
    # digit, so the full stop after an address stays in the sentence.
    ("[EMAIL]", r"[\w.+-]+@[\w-]+\.[\w.-]*\w"),
    # An optional +, then at least nine digits and spaces, starting and
    # ending on a digit. An order number has four, so it stays.
    ("[PHONE]", r"\+?\d[\d ]{7,}\d"),
)


def redacted(text: str) -> str:
    # The text with every card number, email address and phone number
    # replaced by its placeholder. A pattern errs on the side of taking
    # too much: a long number that is not a phone costs a digit or two
    # of the log, a phone it missed is a person's number kept. Names and
    # street addresses follow no pattern, so they pass: this lowers what
    # a log or a prompt holds, it does not make it anonymous.
    for placeholder, pattern in PATTERNS:
        text = re.sub(pattern, placeholder, text)
    return text
