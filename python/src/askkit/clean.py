def clean_email(text: str) -> tuple[str, int]:
    # Leaves out only what is certainly not the customer's new message:
    # lines quoted from an earlier email, which start with ">", and the
    # "On 3 Nov 2026, ... wrote:" line right above them. A signature, a
    # P.S. and everything else stay: only the model can tell what in them
    # matters. Returns the cleaned email and how many lines were left out.
    kept: list[str] = []
    left_out = 0
    for line in text.splitlines():
        if line.lstrip().startswith(">"):
            if kept and kept[-1].rstrip().endswith("wrote:"):
                kept.pop()
                left_out += 1
            left_out += 1
            continue
        kept.append(line)
    return "\n".join(kept).strip(), left_out
