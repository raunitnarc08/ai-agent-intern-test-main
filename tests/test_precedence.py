from pathlib import Path
import pytest
from retrieval.parser import parse_markdown_file
from retrieval.precedence import load_and_filter_corpus, is_authoritative_customer_doc
from config import KNOWLEDGE_BASE_DIR


def test_precedence_filtering():
    all_meta, sections, excluded = load_and_filter_corpus(KNOWLEDGE_BASE_DIR)
    
    assert len(all_meta) == 14
    assert "02-returns-policy-legacy.md" in excluded
    assert "13-support-escalation.md" in excluded
    assert "14-internal-content-migration-notes.md" in excluded
    
    # Assert authoritative files are in sections
    authoritative_files = {sec.filename for sec in sections}
    assert "01-returns-policy-current.md" in authoritative_files
    assert "03-final-sale-and-promotions.md" in authoritative_files
    assert "06-international-shipping.md" in authoritative_files
    assert "07-warranty.md" in authoritative_files
    assert "09-trailplus-membership.md" in authoritative_files
    assert "11-product-care.md" in authoritative_files
    assert "12-breeze-tumbler-product-card.md" in authoritative_files


def test_section_parsing():
    doc_path = KNOWLEDGE_BASE_DIR / "01-returns-policy-current.md"
    meta, sections = parse_markdown_file(doc_path)
    
    assert meta.status == "active"
    assert meta.policy_authority == "official"
    assert meta.supersedes == "RET-2024-01"
    assert len(sections) >= 4
