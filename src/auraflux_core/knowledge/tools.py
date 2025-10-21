import asyncio
import json
from typing import Any, Dict, List

import httpx
from playwright.async_api import async_playwright
from pydantic import ValidationError

from auraflux_core.core.tools.base_tool import BaseTool, ToolConfig
from auraflux_core.knowledge.graphs import MindMap
from auraflux_core.knowledge.schemas import GoogleSearchToolConfig


class GoogleSearchTool(BaseTool):
    """
    A tool that performs a Google search using the official Custom Search JSON API.

    API Documentation:
        https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list
    """
    def __init__(self, config: GoogleSearchToolConfig):
        super().__init__(config=config)
        # These should be loaded from a secure configuration or environment variables
        self.api_key = config.google_search_engine_api_key
        self.cx = config.google_search_engine_id
        self.base_url = config.google_search_engine_base_url
        self.client = httpx.AsyncClient()

    async def run(self, **kwargs) -> Any:
        """
        Executes a Google search for the given query using the Custom Search API.

        Args:
            query: The search query string.

        Returns:
            A JSON string containing the search results.
        """
        if not self.api_key or not self.cx:
            return json.dumps({"error": "API key or Search Engine ID (cx) is not configured."})

        query = kwargs.get('query', '')
        self.logger.debug(f"Running GoogleSearchTool with query: {query}")
        params = {
            "key": self.api_key,
            "cx": self.cx,
            "q": query.strip()
        }

        try:
            if not params.get('q'):
                raise ValueError("Query parameter cannot be empty or missing.")

            response = await self.client.get(self.base_url, params=params, timeout=10.0)
            response.raise_for_status()  # Raise an exception for bad status codes

            search_data = response.json()
            self.logger.debug(f"Google Search API response: {search_data}")
            items = search_data.get("items", [])

            processed_results: List[Dict[str, str | None]] = []
            for result in items:
                processed_results.append({
                    "title": result.get("title"),
                    "url": result.get("link"),
                    "snippet": result.get("snippet"),
                    "source": result.get("displayLink")
                })

            return json.dumps(processed_results, ensure_ascii=False)

        except httpx.HTTPStatusError as e:
            error_message = f"HTTP error during Google search: {e.response.text}"
            return json.dumps({"error": error_message})
        except Exception as e:
            error_message = f"An unexpected error occurred: {str(e)}"
            return json.dumps({"error": error_message})

    def get_name(self) -> str:
        return "google_search"

    def get_description(self) -> str:
        return "Performs a Google search and returns a list of web page links and snippets."

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string to search on Google."
                }
            },
            "required": ["query"]
        }


class MindMapGraphBuilder(BaseTool):
    """
    A tool that constructs and updates a MindMap graph from structured LLM output.
    It uses the schema's internal methods to handle conceptual uniqueness.
    """

    def __init__(self, config: ToolConfig):
        super().__init__(config=config)
        # The main graph object to be built
        self.graph: MindMap = MindMap()

    async def run(self, **kwargs) -> Any:
        """
        Parses LLM-generated JSON and updates the internal graph.
        """
        json_output = kwargs.get('json_output', '{}')
        if not json_output:
            return json.dumps({"status": "error", "message": "Input JSON is empty."})

        try:
            llm_data = json.loads(json_output)

            self.update_graph(llm_data)

            # Return a confirmation and a snapshot of the current graph state
            return json.dumps({"status": "success", "graph_snapshot": self.graph.model_dump_json()}, ensure_ascii=False)

        except json.JSONDecodeError as e:
            return json.dumps({"status": "error", "message": f"Failed to parse JSON: {e}"})
        except ValidationError as e:
            return json.dumps({"status": "error", "message": f"Validation error: {e}"})

    def update_graph(self, data: Dict[str, Any]) -> None:
        """
        Parses LLM output and updates the internal graph incrementally.
        """
        # Step 1: Handle the central idea
        if not data:
            return None

        self.graph.add_central_idea_data(data)

    def get_name(self) -> str:
        return "mind_map_graph_builder"

    def get_description(self) -> str:
        return "Builds a mind map graph from structured JSON output."

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "json_output": {
                    "type": "string",
                    "description": "A JSON string containing the mind map structure."
                }
            },
            "required": ["json_output"]
        }


class PlaywrightWrapper:
    """
    A wrapper for Playwright to perform web scraping tasks.
    """
    def __init__(self):
        self._browser = None
        self._context = None
        self._page = None
        self._playwright = None
        self.timeout = 300000

    async def __aenter__(self):
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch()
        self._context = await self._browser.new_context()
        self._page = await self._context.new_page()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def scrape_all_text_from_url(self, url: str) -> str:
        if not self._page:
            raise RuntimeError("Playwright page is not initialized.")

        try:
            await self._page.goto(url, timeout=self.timeout)
            element = self._page.locator('body')
            data = await element.all_inner_texts()
            return json.dumps({"url": url, "query": 'body', "data": ''.join(data)})
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def scrape_from_html(self, html_content: str, query: str) -> str:
        """
        Scrapes data from raw HTML content.

        Args:
            html_content (str): The raw HTML content to load.
            query (str): The CSS selector to identify the data to be scraped.
        """
        if not self._page:
            raise RuntimeError("Playwright page is not initialized.")

        try:
            await self._page.set_content(html_content)
            data = await self._page.locator(query).inner_text()
            return json.dumps({"query": query, "data": data})
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def scrape_from_url(self, url: str, query: str) -> str:
        """
        Navigates to a URL and scrapes data based on a query.

        Args:
            url (str): The URL of the webpage to scrape.
            query (str): The CSS selector to identify the data to be scraped.
        """
        if not self._page:
            raise RuntimeError("Playwright page is not initialized.")

        try:
            await self._page.goto(url, timeout=self.timeout)
            element = self._page.locator(query)
            data = await element.inner_text()
            return json.dumps({"url": url, "query": query, "data": data})
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def get_raw_html(self, url: str) -> str:
        """
        Navigates to a URL and returns the raw HTML content.
        """
        if not self._page:
            raise RuntimeError("Playwright page is not initialized.")

        try:
            await self._page.goto(url, timeout=self.timeout)
            html_content = await self._page.content()
            return json.dumps({"url": url, "html_content": html_content})
        except Exception as e:
            return json.dumps({"error": str(e)})


class PageLoaderTool(BaseTool):
    """
    A tool to load the raw HTML content of a webpage.
    """

    async def run(self, **kwargs) -> Any:
        """
        Executes the page loading task and returns the raw HTML.
        """
        url = kwargs.get('url', '')
        if not url:
            return json.dumps({"error": "Missing 'url' parameter."})

        # Call the asynchronous method to get the HTML content
        async with PlaywrightWrapper() as playwright:
            return await playwright.get_raw_html(url)

    def get_name(self) -> str:
        return "load_page_html"

    def get_description(self) -> str:
        return "Navigates to a given URL and returns the entire raw HTML content of the page as a JSON string."

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the webpage to load."
                }
            },
            "required": ["url"]
        }


class WebScrapingTool(BaseTool):
    """
    A concrete tool implementation for the WebScout agent.
    """

    async def run(self, **kwargs) -> str:
        """
        Executes the web scraping task and returns a JSON string
        containing a list of parsed outputs.
        """
        url_contents = kwargs.get('url_contents', '')
        self.logger.debug(f"Running WebScoutTool with url_contents: {url_contents}")
        if not url_contents:
            return json.dumps({"error": "Missing 'url_contents' parameter."})

        try:
            urls_to_scrape = json.loads(url_contents)
            if not isinstance(urls_to_scrape, list):
                return json.dumps({"error": "Input must be a list of URLs."})
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid JSON format for 'url_contents'."})

        tasks = []
        for url_data in urls_to_scrape:
            if isinstance(url_data, dict) and 'url' in url_data:
                tasks.append(self.scrape_single_url(url=url_data['url']))
            else:
                self.logger.warning(f"Skipping invalid URL entry: {url_data}")

        # Run all scraping tasks concurrently
        scraped_results: List[str | BaseException] = await asyncio.gather(*tasks, return_exceptions=True)

        # Parse each individual JSON string result
        final_results = self.postprocess_scraped_results(scraped_results)

        # Return the final list of parsed results as a single JSON string
        return json.dumps(final_results, ensure_ascii=False)

    async def scrape_single_url(self, url: str):
        """Scrapes a single URL using a dedicated Playwright instance."""
        try:
            async with PlaywrightWrapper() as playwright:
                return await playwright.scrape_all_text_from_url(url)
        except Exception as e:
            self.logger.error(f"Error scraping {url}: {e}")
            return {"error": str(e)}

    def postprocess_scraped_results(self, scraped_results: List[str | BaseException]) -> List[Dict[str, Any]]:
        """
        Parse each individual JSON string result
        """

        final_results: List[Dict[str, Any]] = []
        for result in scraped_results:
            if isinstance(result, BaseException):
                self.logger.error(f"An error occurred during scraping: {result}")
                final_results.append({"error": str(result)})
            else:
                try:
                    # Parse the JSON string from the scraping function
                    parsed_result = json.loads(result)
                    if parsed_result.get('error', ''):
                        continue

                    final_results.append(json.loads(result))
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse JSON from scraped content: {result}. Error: {e}")
                    final_results.append({"error": "Failed to parse scraped content as JSON."})
                except Exception as e:
                    self.logger.error(f"An unexpected error occurred: {e}")
                    final_results.append({"error": f"An unexpected error occurred: {str(e)}"})

        return final_results

    def get_name(self) -> str:
        return "scrape_web_data"

    def get_description(self) -> str:
        return "Scrapes data from a specified HTML content using a CSS selector query and returns it as a JSON string."

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url_contents": {
                    "type": "string",
                    "description": "A JSON string containing a list of URLs to scrape."
                }
            },
            "required": ["url_contents"]
        }