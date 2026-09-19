import re


def dynamic_chunk(
    text: str,
    min_size: int = 300,
    max_size: int = 1000
):

    paragraphs = re.split(
        r"\n\s*\n",
        text
    )

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        proposed = (
            current_chunk + "\n" + paragraph
        ).strip()

        if len(proposed) <= max_size:

            current_chunk = proposed

        else:

            if len(current_chunk) >= min_size:
                chunks.append(current_chunk)

            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks