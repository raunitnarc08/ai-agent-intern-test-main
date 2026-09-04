from pathlib import Path
from typing import List, Tuple
from retrieval.parser import DocumentMetadata, DocumentSection, parse_markdown_file


def is_authoritative_customer_doc(metadata: DocumentMetadata) -> bool:
    """
    Check if a document is an active, official customer-facing policy/product document.
    
    Exclusion rules:
    - Must have status == 'active' (excludes superseded/draft docs like 02-returns-policy-legacy.md)
    - Must have policy_authority == 'official' (excludes draft/unapproved like 14-internal-content-migration-notes.md)
    - Must have customer_answering is True (excludes internal scratchpads)
    - Must have audience == 'customer' (excludes internal operating guides like 13-support-escalation.md)
    """
    if metadata.status != "active":
        return False
    if metadata.policy_authority != "official":
        return False
    if metadata.customer_answering is False:
        return False
    if metadata.audience != "customer":
        return False
    return True


def load_and_filter_corpus(kb_dir: Path) -> Tuple[List[DocumentMetadata], List[DocumentSection], List[str]]:
    """
    Load all markdown documents from knowledge-base, apply precedence filter,
    and return filtered metadata, indexed sections, and a list of excluded document filenames.
    """
    all_metadata: List[DocumentMetadata] = []
    authoritative_sections: List[DocumentSection] = []
    excluded_files: List[str] = []

    for md_file in sorted(kb_dir.glob("*.md")):
        meta, sections = parse_markdown_file(md_file)
        all_metadata.append(meta)
        if is_authoritative_customer_doc(meta):
            authoritative_sections.extend(sections)
        else:
            excluded_files.append(md_file.name)

    return all_metadata, authoritative_sections, excluded_files
