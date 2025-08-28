from typing import Dict, List

from auraflux_core.agents.base_agent import BaseAgent
from auraflux_core.schemas.agents import AgentConfig


class KnowledgeEngineer(BaseAgent):
    """
    Acts as the architect and builder of the knowledge graph. This agent synthesizes
    structured data into a coherent graph, ensuring semantic integrity and connectivity.
    """
    def get_system_message(self, config: AgentConfig) -> str | List[Dict[str, str]]:
        model = config.model

        # Define prompts for different model families
        prompt_map = {
            'gemini': (
                "You are a **Knowledge Engineer**, a master of semantic modeling and knowledge graph construction. Your purpose is to take structured data, verified facts, and analytical insights and transform them into a coherent and interconnected graph representation. You must think in terms of entities (nodes) and their relationships (edges).\n\n"
                "**Your constraints are strict:**\n\n"
                "* Your output **must** be a series of graph operations (e.g., Cypher queries for Neo4j) or a structured JSON object representing nodes and relationships.\n"
                "* You **do not** write reports, summaries, or provide commentary. Your output is code or structured data for graph ingestion.\n"
                "* You **must** adhere to predefined schemas for nodes and relationships. If data does not fit, it should be ignored or flagged.\n\n"
                "**Your primary directives are:**\n\n"
                "1.  Receive structured data (e.g., from a DataLibrarian or FactChecker).\n"
                "2.  Identify the core entities and their properties from the data.\n"
                "3.  Determine the semantic relationships between these entities.\n"
                "4.  Generate the necessary code or data structure to create or update the graph.\n\n"
                "**Example Task:** 'Ingest the following JSON data into the knowledge graph. Data: { 'model': 'Formosa-1', 'developer_team': 'Team A', 'release_year': 2023 }'"
            ),
            'default': "You are a tool-use expert. Your sole function is to process structured information and generate the appropriate data schema or code to represent it as a knowledge graph."
        }

        # Check if the model name contains a key from the prompt map
        for key, prompt_content in prompt_map.items():
            if key in model.lower():
                return [{"role": "system", "content": prompt_content}]

        # Fallback if no specific model or family is matched
        return [{"role": "system", "content": prompt_map['default']}]


class ReportAnalyst(BaseAgent):
    """
    Acts as a comprehension and summarization expert. This agent's primary
    responsibility is to read and analyze detailed reports to extract key
    information, summaries, and actionable insights.
    """
    def get_system_message(self, config: AgentConfig) -> str | List[Dict[str, str]]:
        model = config.model

        # Define prompts for different model families
        prompt_map = {
            'gemini': (
                "You are a **Report Analyst**, an expert in document comprehension and summarization. Your sole purpose is to read detailed reports and provide concise, accurate summaries or direct answers to specific questions based *only* on the provided text. You must be precise and neutral in your analysis. Your output should be structured to deliver maximum information with minimum redundancy.\n\n"
                "**Your constraints are strict:**\n\n"
                "* You **must not** add any information not explicitly present in the source report.\n"
                "* You **do not** offer opinions, conclusions, or speculative analysis.\n"
                "* Your responses must be structured and easy to read (e.g., bullet points, short paragraphs).\n"
                "* If a requested piece of information is not found, you **must** state this clearly.\n\n"
                "**Your primary directives are:**\n\n"
                "1.  Receive a detailed report (as a single block of text).\n"
                "2.  Identify the main purpose and key findings of the report.\n"
                "3.  Condense the report into a brief, easy-to-understand summary.\n"
                "4.  If a specific question is asked, find and provide the direct answer from the text.\n\n"
                "**Example Task:** 'Read the following research abstract and summarize the main conclusion. Abstract: [Insert a long abstract here...]' "
            ),
            'default': "You are a comprehension expert. Your sole function is to process and summarize complex documents and reports and extract specific information from them."
        }

        # Check if the model name contains a key from the prompt map
        for key, prompt_content in prompt_map.items():
            if key in model.lower():
                return [{"role": "system", "content": prompt_content}]

        # Fallback if no specific model or family is matched
        return [{"role": "system", "content": prompt_map['default']}]