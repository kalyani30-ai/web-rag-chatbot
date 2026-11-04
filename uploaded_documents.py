"""Extract searchable text and metadata from user-uploaded local files."""

import io
import mimetypes
from pathlib import Path
from zipfile import BadZipFile


MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".log"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
SUPPORTED_EXTENSIONS = (
    {".pdf", ".docx"}
    | TEXT_EXTENSIONS
    | IMAGE_EXTENSIONS
    | VIDEO_EXTENSIONS
)


class DocumentUploadError(ValueError):
    """A user upload could not be processed into searchable content."""


def _extract_pdf(content: bytes) -> str:
    try:
        from pypdf import PdfReader
        from pypdf.errors import FileNotDecryptedError, PdfReadError
    except ImportError as error:
        raise DocumentUploadError(
            "PDF support is unavailable. Install the project requirements."
        ) from error

    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            raise DocumentUploadError(
                "This PDF is password-protected. Upload an unlocked copy."
            )
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except (PdfReadError, FileNotDecryptedError) as error:
        raise DocumentUploadError("This PDF could not be read.") from error


def _extract_docx(content: bytes) -> str:
    try:
        from docx import Document
        from docx.opc.exceptions import PackageNotFoundError
    except ImportError as error:
        raise DocumentUploadError(
            "Word support is unavailable. Install the project requirements."
        ) from error

    try:
        document = Document(io.BytesIO(content))
    except (BadZipFile, PackageNotFoundError) as error:
        raise DocumentUploadError("This Word document could not be read.") from error

    parts = [paragraph.text for paragraph in document.paragraphs if paragraph.text]
    for table in document.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(parts).strip()


def _extract_image(content: bytes) -> str:
    try:
        from PIL import Image, UnidentifiedImageError
        import pytesseract
        from pytesseract.pytesseract import TesseractNotFoundError
    except ImportError as error:
        raise DocumentUploadError(
            "Image OCR support is unavailable. Install the project requirements."
        ) from error

    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise DocumentUploadError(
                    "This image is too large to OCR (maximum 20 megapixels)."
                )
            return pytesseract.image_to_string(image).strip()
    except TesseractNotFoundError as error:
        raise DocumentUploadError(
            "The Tesseract OCR program is not installed. Install Tesseract OCR "
            "and restart the app to read images."
        ) from error
    except UnidentifiedImageError as error:
        raise DocumentUploadError("This image file could not be read.") from error
    except OSError as error:
        raise DocumentUploadError("This image could not be processed.") from error


def extract_uploaded_content(
    file_name: str,
    content: bytes,
) -> tuple[str, str]:
    """Return a file category and extracted text for supported uploads."""
    if not isinstance(content, bytes):
        raise TypeError("Uploaded file content must be bytes.")
    if not content:
        raise DocumentUploadError("The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise DocumentUploadError("Each uploaded file must be 20 MB or smaller.")

    extension = Path(file_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentUploadError(
            f"Unsupported file type {extension or '(no extension)'}. "
            "Supported types: PDF, DOCX, TXT, MD, CSV, JSON, LOG, common images, "
            "and common videos."
        )

    if extension == ".pdf":
        category, text = "PDF", _extract_pdf(content)
    elif extension == ".docx":
        category, text = "Word document", _extract_docx(content)
    elif extension in TEXT_EXTENSIONS:
        category = "Text document"
        try:
            text = content.decode("utf-8-sig").strip()
        except UnicodeDecodeError as error:
            raise DocumentUploadError(
                "This text file is not valid UTF-8. Save it as UTF-8 and retry."
            ) from error
    elif extension in IMAGE_EXTENSIONS:
        category, text = "Image", _extract_image(content)
    else:
        category = "Video metadata"
        mime_type = mimetypes.guess_type(file_name)[0] or "unknown video format"
        text = (
            f"Video file named {Path(file_name).name}. "
            f"Format: {mime_type}. File size: {len(content)} bytes. "
            "Video audio and frames were not transcribed or analyzed."
        )

    if not text:
        if category == "Image":
            raise DocumentUploadError(
                "No readable text was found in this image. OCR works best on "
                "clear images containing printed text."
            )
        if category == "PDF":
            raise DocumentUploadError(
                "No selectable text was found in this PDF. Scanned PDFs need OCR "
                "before their content can be searched."
            )
        raise DocumentUploadError("No readable text was found in this file.")
    return category, text
