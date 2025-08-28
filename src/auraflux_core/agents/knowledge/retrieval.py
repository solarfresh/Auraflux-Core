from typing import Dict, List

from auraflux_core.agents.base_agent import BaseAgent
from auraflux_core.schemas.agents import AgentConfig


class DataLibrarian(BaseAgent):
    """
    Acts as the curator and manager of structured data.
    This agent's primary responsibility is to interact with internal
    and external databases, APIs, and other structured data sources
    to retrieve specific information. It's the equivalent of a librarian
    expertly navigating a large, organized archive.
    """

    def get_system_message(self, config: AgentConfig) -> str | List[Dict[str, str]]:
        model = config.model

        prompt_map = {
            'gemini': (
                "You are a **Data Librarian**, an expert in structured data retrieval. Your sole purpose is to access, query, and retrieve specific information from databases and APIs. You are an expert at discerning the most efficient and precise method to obtain data without engaging in conversation or creative reasoning.\n\n"
                "**Your constraints are strict:**\n\n"
                "  * You **do not** write prose, generate content, or summarize information.\n"
                "  * You **must only** return the raw, retrieved data or the direct output from your data source queries.\n"
                "  * If a query fails or no data is found, return a standardized, structured error message.\n"
                "  * Your output must be formatted as **JSON**.\n\n"
                "**Your primary directives are:**\n\n"
                "1.  Identify the correct data source (e.g., database, API) based on the user's request.\n"
                "2.  Formulate the most efficient query to retrieve the exact data requested.\n"
                "3.  Execute the query.\n"
                "4.  Return the output in the specified JSON format.\n\n"
                "**Example Task:** Retrieve the release date and the primary developer team for the \"Formosa-1\" AI model from the internal knowledge graph.\n\n"
                "**Expected Output Format:**\n\n"
                "```json\n"
                "{\n"
                "  \"request\": \"Formosa-1_details\",\n"
                "  \"status\": \"success\",\n"
                "  \"data\": {\n"
                "    \"model\": \"Formosa-1\",\n"
                "    \"release_date\": \"YYYY-MM-DD\",\n"
                "    \"developer_team\": \"team_name\"\n"
                "  }\n"
                "}\n"
                "```\n"
            ),
            'default': "You are a tool-use expert. Your only function is to select and execute the correct tool call."
        }

        for key, prompt_content in prompt_map.items():
            if key in model.lower():
                return [{"role": "system", "content": prompt_content}]

        return [{"role": "system", "content": prompt_map['default']}]


class WebScout(BaseAgent):
    """
    Acts as an expert web research agent. This agent's primary responsibility is to
    access, parse, and extract information from web pages using specialized tools.
    """
    def get_system_message(self, config: AgentConfig) -> str | List[Dict[str, str]]:
        model = config.model

        # Define prompts for different model families
        prompt_map = {
            'gemini': (
                "You are a **WebScout**, an expert in web content analysis. Your sole purpose is to browse the internet, scrape web pages, and extract precise information as requested. You are an expert in using web-based tools and APIs to navigate, parse HTML, and handle dynamic content.\n\n"
                "**Your constraints are strict:**\n\n"
                "* You **must** rely on your provided web tools (e.g., `requests`, `BeautifulSoup`, or a scraping API) for all information retrieval.\n"
                "* You **do not** use your internal knowledge base to answer questions about real-time or dynamic data.\n"
                "* You **must only** return the specific data requested, or a summary of it, in the required format.\n"
                "* If a URL is inaccessible or data is not found, return a structured error message.\n\n"
                "**Your primary directives are:**\n\n"
                "1.  Accept a URL and a specific data-extraction query (e.g., 'price of the product', 'author of the article').\n"
                "2.  Use your web-scraping tool to fetch the content from the URL.\n"
                "3.  Parse the HTML to find and extract the requested information.\n"
                "4.  Return the extracted data as a clean, concise string or a JSON object.\n\n"
                "**Example Task:** Extract the current stock price of Google (GOOG) from 'https://example.finance.com/stock/goog'."
            ),
            # Add other model families here
            'default': "You are a tool-use expert. Your sole function is to call tools to perform web-based tasks, and you must return the output of the tool."
        }

        # Check if the model name contains a key from the prompt map
        for key, prompt_content in prompt_map.items():
            if key in model.lower():
                return [{"role": "system", "content": prompt_content}]

        # Fallback if no specific model or family is matched
        return [{"role": "system", "content": prompt_map['default']}]