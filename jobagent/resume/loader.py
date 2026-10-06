"""Resume PDF to clean text."""

import re
from pathlib import Path


def clean_resume_text(text: str) -> str:
    text = text.replace("​", "").replace("\xa0", " ")

    lines = [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in text.splitlines()
    ]

    # PDF extraction leaves long runs of empty lines; keep one as a separator.
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def load_resume_text(path: Path) -> str:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Resume not found: {path}"
        )

    import pymupdf

    document = pymupdf.open(str(path))
    try:
        text = "".join(
            page.get_text()
            for page in document
        )
    finally:
        document.close()

    return clean_resume_text(text)
