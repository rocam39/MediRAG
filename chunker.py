import re


def clean_text(text):

    # Replace multiple spaces/newlines with a single space
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_into_sentences(text):

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def chunk_pages(
    pages,
    chunk_size=1000,
    overlap=100
):

    chunks = []

    for page in pages:

        text = clean_text(page["text"])

        sentences = split_into_sentences(text)

        current_chunk = ""

        for sentence in sentences:

            # If adding the sentence still fits
            if len(current_chunk) + len(sentence) <= chunk_size:

                if current_chunk:
                    current_chunk += " "

                current_chunk += sentence

            else:

                # Save current chunk
                if current_chunk.strip():

                    chunks.append({
                        "text": current_chunk.strip(),
                        "page": page["page"]
                    })

                # Create overlap from previous chunk
                overlap_text = current_chunk[-overlap:]

                current_chunk = (
                    overlap_text + " " + sentence
                ).strip()

        # Save final chunk
        if current_chunk.strip():

            chunks.append({
                "text": current_chunk.strip(),
                "page": page["page"]
            })

    return chunks