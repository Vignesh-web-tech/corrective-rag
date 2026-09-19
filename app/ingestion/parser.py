from pathlib import Path
from pypdf import PdfReader
from docx import Document
import pandas as pd


def parse_document(file_path: str) -> str:

    extension = Path(file_path).suffix.lower()

    if extension == ".txt":
        return Path(file_path).read_text(
            encoding="utf-8"
        )

    elif extension == ".pdf":

        reader = PdfReader(file_path)

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return text

    elif extension == ".docx":

        document = Document(file_path)

        return "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
        )

    elif extension == ".csv":

        df = pd.read_csv(file_path)

        return df.to_string(index=False)

    else:

        raise ValueError(
            f"Unsupported document type: {extension}"
        )