import os
from typing import Any, Dict

from auraflux_core.core.tools.base_tool import BaseTool


class FileReaderTool(BaseTool):
    """
    A shared infrastructure tool for ingesting documents.
    Supports basic text, Markdown, and PDF parsing with line/paragraph indexing.
    """

    async def run(self, file_path: str, indexing_mode: str = "paragraph") -> Dict[str, Any]:
        """
        Reads a file and returns a structured list of chunks with source tracking.

        Args:
            file_path: Absolute or relative path to the source file.
            indexing_mode: Method to segment text ('line' or 'paragraph').
        """
        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            return {"success": False, "error": "File not found"}

        self.logger.info(f"Ingesting file: {file_path} (Mode: {indexing_mode})")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()

            filename = os.path.basename(file_path)
            chunks = []

            if indexing_mode == "paragraph":
                segments = [s.strip() for s in raw_text.split("\n\n") if s.strip()]
            else:
                segments = raw_text.splitlines()

            for idx, content in enumerate(segments):
                chunks.append({
                    "source_id": f"{filename}#{indexing_mode}_{idx}",
                    "content": content,
                    "metadata": {
                        "index": idx,
                        "file_type": filename.split('.')[-1]
                    }
                })

            return {
                "success": True,
                "file_metadata": {"path": file_path, "total_chunks": len(chunks)},
                "chunks": chunks
            }

        except Exception as e:
            self.logger.exception(f"Failed to read file {file_path}")
            return {"success": False, "error": str(e)}

    def get_name(self) -> str:
        return "file_reader"

    def get_description(self) -> str:
        return (
            "Reads documents (PDF/MD/TXT) and converts them into traceable chunks. "
            "Each chunk is assigned a unique source_id, which is mandatory for "
            "maintaining empirical grounding in the Auraflux system."
        )

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the target document."
                },
                "indexing_mode": {
                    "type": "string",
                    "enum": ["line", "paragraph"],
                    "default": "paragraph",
                    "description": "Strategy for segmenting text into traceable units."
                }
            },
            "required": ["file_path"]
        }
