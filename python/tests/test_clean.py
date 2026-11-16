from askkit.clean import clean_email


def test_quoted_lines_and_their_wrote_line_are_left_out() -> None:
    text = (
        "Order 4417 came broken.\n"
        "\n"
        "On 3 Nov 2026, at 10:14, the shop wrote:\n"
        "> Your return of order 7302 is accepted.\n"
        ">\n"
        "> Please send the vase back within 14 days.\n"
    )
    assert clean_email(text) == ("Order 4417 came broken.", 4)


def test_a_quote_in_the_middle_is_left_out_and_the_rest_kept() -> None:
    text = (
        "> Your order 7302 has shipped.\n"
        "It did, but this is about order 4417.\n"
        "  > an indented quote is still a quote\n"
        "Thanks, Maria"
    )
    assert clean_email(text) == (
        "It did, but this is about order 4417.\nThanks, Maria",
        2,
    )


def test_a_signature_and_a_ps_are_kept() -> None:
    # Only the model can tell what in them matters: a P.S. may hold the
    # order number.
    text = "Thanks,\nMaria\nTel. 0161 496 0731\n\nP.S. The order is 4417."
    assert clean_email(text) == (text, 0)


def test_a_line_that_only_ends_in_wrote_is_kept() -> None:
    # "wrote:" is left out only right above a quoted line.
    text = "My son wrote:\nthe mug is broken."
    assert clean_email(text) == (text, 0)


def test_an_arrow_inside_a_line_is_not_a_quote() -> None:
    text = "Box torn -> mug broken. Order 4417."
    assert clean_email(text) == (text, 0)
