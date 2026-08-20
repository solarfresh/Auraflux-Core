from typing import List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class BaseSectionMetadata(BaseModel):
    """
    Base metadata model containing fields common to all document types.
    """
    source_file: str = Field(..., description="Name or path of the original file")
    file_type: str = Field(..., description="File extension/type (e.g., 'pdf', 'docx', 'txt')")
    char_count: Optional[int] = Field(None, description="Total character count of the raw section text")

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)


class PDFSectionMetadata(BaseSectionMetadata):
    """
    Format-specific metadata schema for PDF documents.
    """
    file_type: Literal["pdf"] = "pdf"
    page_num: int = Field(..., description="Page number where the section originates")
    bbox: Optional[List[float]] = Field(None, description="Bounding box coordinates [x0, y0, x1, y1]")


class DOCXSectionMetadata(BaseSectionMetadata):
    """
    Format-specific metadata schema for Word documents.
    """
    file_type: Literal["docx"] = "docx"
    heading_level: Optional[int] = Field(None, description="Word paragraph heading level (1-6)")


class TXTSectionMetadata(BaseSectionMetadata):
    """
    Format-specific metadata schema for plain text files.
    """
    file_type: Literal["txt"] = "txt"
    encoding: str = Field(default="utf-8", description="Detected or used file encoding")


# Discriminator Union for flexible validation across formats
SectionMetadata = Union[PDFSectionMetadata, DOCXSectionMetadata, TXTSectionMetadata, BaseSectionMetadata]


class StandardSection(BaseModel):
    """
    Standardized intermediate section schema output by Stage 1 Parsers.
    Strictly validates both content and metadata structure.
    """
    section_id: str = Field(..., description="Unique section identifier, e.g., 'sec_001'")
    breadcrumb: List[str] = Field(
        default_factory=list,
        description="Hierarchical section path, e.g., ['Employee_Handbook.pdf', 'Chapter 1']"
    )
    title: str = Field(default="", description="Current subsection title, e.g., '1.1 Scope'")
    text: str = Field(..., description="Full raw text content of the section prior to chunking")
    metadata: SectionMetadata = Field(..., description="Strictly-typed format-specific metadata")

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)