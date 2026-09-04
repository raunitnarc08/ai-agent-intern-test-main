from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import re


@dataclass
class DocumentMetadata:
    document_id: str
    title: str
    status: str
    effective_date: str
    last_reviewed: Optional[str] = None
    audience: str = "customer"
    policy_authority: str = "official"
    supersedes: Optional[str] = None
    superseded_by: Optional[str] = None
    customer_answering: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentSection:
    filename: str
    document_id: str
    doc_title: str
    heading: str
    content: str
    metadata: DocumentMetadata

    @property
    def citation(self) -> str:
        if self.heading and self.heading != self.doc_title:
            return f"{self.filename} ({self.heading})"
        return self.filename


def parse_frontmatter(text: str) -> tuple[Dict[str, Any], str]:
    """Parse YAML-like frontmatter between --- markers."""
    meta: Dict[str, Any] = {}
    body = text

    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            fm_text = parts[1]
            body = parts[2].strip()
            for line in fm_text.splitlines():
                line = line.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                # Parse boolean / strings
                if val.lower() == "true":
                    val_parsed: Any = True
                elif val.lower() == "false":
                    val_parsed = False
                elif val.lower() == "null" or val == "":
                    val_parsed = None
                else:
                    val_parsed = val
                meta[key] = val_parsed
    return meta, body


def parse_markdown_file(file_path: Path) -> tuple[DocumentMetadata, List[DocumentSection]]:
    """Parse a markdown file into metadata and sections."""
    text = file_path.read_text(encoding="utf-8")
    meta_dict, body = parse_frontmatter(text)

    metadata = DocumentMetadata(
        document_id=meta_dict.get("document_id", file_path.stem),
        title=meta_dict.get("title", file_path.stem),
        status=meta_dict.get("status", "unknown").lower(),
        effective_date=str(meta_dict.get("effective_date", "")),
        last_reviewed=str(meta_dict.get("last_reviewed", "")) if meta_dict.get("last_reviewed") else None,
        audience=meta_dict.get("audience", "customer").lower(),
        policy_authority=meta_dict.get("policy_authority", "official").lower(),
        supersedes=meta_dict.get("supersedes"),
        superseded_by=meta_dict.get("superseded_by"),
        customer_answering=meta_dict.get("customer_answering", True),
        extra=meta_dict
    )

    # Split body into sections by ## headings
    lines = body.splitlines()
    sections: List[DocumentSection] = []

    current_heading = metadata.title
    current_lines: List[str] = []

    for line in lines:
        if line.startswith("## "):
            if current_lines:
                content = "\n".join(current_lines).strip()
                if content:
                    sections.append(DocumentSection(
                        filename=file_path.name,
                        document_id=metadata.document_id,
                        doc_title=metadata.title,
                        heading=current_heading,
                        content=content,
                        metadata=metadata
                    ))
                current_lines = []
            current_heading = line.replace("## ", "").strip()
            # NOTE: intentionally NOT appending the raw '## Heading' line to
            # current_lines. The heading is already carried structurally via
            # `heading` and `citation` — duplicating it into `content` used
            # to glue a markdown header token onto the front of the first
            # real sentence in every section (e.g. "## Breeze Tumbler The
            # stainless-steel body..."), which corrupted the fallback's
            # sentence-splitting and the evaluator's embedding comparisons
            # system-wide.
        else:
            current_lines.append(line)

    if current_lines:
        content = "\n".join(current_lines).strip()
        if content:
            sections.append(DocumentSection(
                filename=file_path.name,
                document_id=metadata.document_id,
                doc_title=metadata.title,
                heading=current_heading,
                content=content,
                metadata=metadata
            ))

    return metadata, sections