import io
import unittest
from unittest import mock

from pypdf import PdfWriter

import app


class AppTests(unittest.TestCase):
    def test_extract_text_from_pdf_blank_page(self) -> None:
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        buffer = io.BytesIO()
        writer.write(buffer)

        text = app.extract_text_from_pdf(buffer.getvalue())

        self.assertEqual(text, "")

    def test_chunk_text_respects_chunk_size(self) -> None:
        text = ("word " * 500).strip()
        chunks = app.chunk_text(text, chunk_size=100, chunk_overlap=10)

        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk) <= 100 for chunk in chunks))

    def test_build_docs_from_pdfs_uses_chunk_metadata(self) -> None:
        with mock.patch("app.extract_text_from_pdf", return_value="ignored"):
            with mock.patch("app.chunk_text", return_value=["a", "b"]):
                docs = app.build_docs_from_pdfs([("f1.pdf", b"bytes")])

        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].page_content, "a")
        self.assertEqual(docs[0].metadata["source"], "f1.pdf")
        self.assertEqual(docs[0].metadata["chunk"], 0)
        self.assertEqual(docs[1].page_content, "b")
        self.assertEqual(docs[1].metadata["chunk"], 1)


if __name__ == "__main__":
    unittest.main()
