import pymupdf


def load_pdf(path):

    document = pymupdf.open(path)

    pages = []

    for page_number, page in enumerate(document):

        text = page.get_text()

        if text.strip():

            pages.append({
                "page": page_number + 1,
                "text": text.strip()
            })

    document.close()

    return pages