"""Tests for local file extraction before document retrieval."""

import unittest
import io
from unittest.mock import patch

from uploaded_documents import (
    MAX_UPLOAD_BYTES,
    DocumentUploadError,
    extract_uploaded_content,
)


class UploadedDocumentTests(unittest.TestCase):
    def test_extracts_utf8_text_and_removes_a_bom(self) -> None:
        category, text = extract_uploaded_content(
            "notes.txt",
            b"\xef\xbb\xbfUseful document text.",
        )

        self.assertEqual(category, "Text document")
        self.assertEqual(text, "Useful document text.")

    def test_video_upload_only_exposes_file_metadata(self) -> None:
        category, text = extract_uploaded_content("lecture.mp4", b"video-bytes")

        self.assertEqual(category, "Video metadata")
        self.assertIn("lecture.mp4", text)
        self.assertIn("audio and frames were not transcribed", text)

    def test_pdf_and_docx_use_their_extractors(self) -> None:
        with (
            patch("uploaded_documents._extract_pdf", return_value="PDF text"),
            patch("uploaded_documents._extract_docx", return_value="Word text"),
        ):
            pdf_category, pdf_text = extract_uploaded_content("report.pdf", b"pdf")
            word_category, word_text = extract_uploaded_content("report.docx", b"docx")

        self.assertEqual((pdf_category, pdf_text), ("PDF", "PDF text"))
        self.assertEqual((word_category, word_text), ("Word document", "Word text"))

    def test_scanned_or_blank_pdf_reports_missing_selectable_text(self) -> None:
        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        content = io.BytesIO()
        writer.write(content)

        with self.assertRaisesRegex(DocumentUploadError, "Scanned PDFs need OCR"):
            extract_uploaded_content("blank.pdf", content.getvalue())

    def test_extracts_word_paragraphs_and_table_cells(self) -> None:
        from docx import Document

        document = Document()
        document.add_paragraph("Word paragraph")
        table = document.add_table(rows=1, cols=2)
        table.cell(0, 0).text = "First cell"
        table.cell(0, 1).text = "Second cell"
        content = io.BytesIO()
        document.save(content)

        category, text = extract_uploaded_content("notes.docx", content.getvalue())

        self.assertEqual(category, "Word document")
        self.assertIn("Word paragraph", text)
        self.assertIn("First cell | Second cell", text)

    def test_extracts_image_text_with_local_ocr(self) -> None:
        from PIL import Image

        image_bytes = io.BytesIO()
        Image.new("RGB", (20, 20), color="white").save(image_bytes, format="PNG")
        with patch("pytesseract.image_to_string", return_value="OCR result"):
            category, text = extract_uploaded_content("note.png", image_bytes.getvalue())

        self.assertEqual(category, "Image")
        self.assertEqual(text, "OCR result")

    def test_missing_tesseract_reports_ocr_unavailable(self) -> None:
        from pytesseract.pytesseract import TesseractNotFoundError
        from PIL import Image

        image_bytes = io.BytesIO()
        Image.new("RGB", (20, 20), color="white").save(image_bytes, format="PNG")
        with patch(
            "pytesseract.image_to_string",
            side_effect=TesseractNotFoundError(),
        ):
            with self.assertRaisesRegex(DocumentUploadError, "Tesseract OCR program"):
                extract_uploaded_content("note.png", image_bytes.getvalue())

    def test_rejects_unsupported_and_oversized_files(self) -> None:
        with self.assertRaisesRegex(DocumentUploadError, "Unsupported file type"):
            extract_uploaded_content("archive.zip", b"data")
        with self.assertRaisesRegex(DocumentUploadError, "20 MB or smaller"):
            extract_uploaded_content("notes.txt", b"x" * (MAX_UPLOAD_BYTES + 1))

    def test_rejects_invalid_utf8_text(self) -> None:
        with self.assertRaisesRegex(DocumentUploadError, "not valid UTF-8"):
            extract_uploaded_content("notes.txt", b"\xff")


if __name__ == "__main__":
    unittest.main()
