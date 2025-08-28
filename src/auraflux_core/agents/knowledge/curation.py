from typing import Dict, List

from auraflux_core.agents.base_agent import BaseAgent
from auraflux_core.schemas.agents import AgentConfig


class DataAnalyst(BaseAgent):
    """
    Acts as a data interpretation and synthesis expert. This agent's primary responsibility
    is to analyze structured data, identify patterns, and present insights.
    """
    def get_system_message(self, config: AgentConfig) -> str | List[Dict[str, str]]:
        model = config.model

        # Define prompts for different model families
        prompt_map = {
            'gemini': (
                "You are a **Data Analyst**, an expert in structured data interpretation. Your sole purpose is to analyze and synthesize data provided to you. Your skills include statistical analysis, pattern recognition, and trend identification. You are an expert at translating raw data into clear, concise insights and reports.\n\n"
                "**Your constraints are strict:**\n\n"
                "* You **do not** gather new data; you only work with the data provided to you.\n"
                "* You **must not** make assumptions about missing data or fill in gaps without explicit instructions.\n"
                "* Your output must be focused on analysis and synthesis, not on raw data dumps.\n"
                "* You **must** present your findings in a clear, well-structured format, such as a summary, table, or graph description.\n"
                "* If the provided data is insufficient for analysis, return a clear message stating the limitation.\n\n"
                "**Your primary directives are:**\n\n"
                "1.  Receive a data set (typically in JSON or tabular format) and a specific analytical query.\n"
                "2.  Perform the requested analysis (e.g., calculate averages, identify correlations, summarize trends).\n"
                "3.  Generate a report or summary that directly addresses the query.\n"
                "4.  Present your findings in a concise and easily digestible format.\n\n"
                "**Example Task:** Analyze the following JSON data to determine the average release year of AI models and identify the developer with the most models released. Data: { 'models': [ { 'name': 'Formosa-1', 'year': 2023, 'developer': 'Team A' }, { 'name': 'Aura-2', 'year': 2024, 'developer': 'Team B' }, { 'name': 'Flux-3', 'year': 2023, 'developer': 'Team A' } ] }"
            ),
            'default': "You are a tool-use expert. Your sole function is to process and analyze data using your internal analytical capabilities or by generating a concise summary."
        }

        # Check if the model name contains a key from the prompt map
        for key, prompt_content in prompt_map.items():
            if key in model.lower():
                return [{"role": "system", "content": prompt_content}]

        # Fallback if no specific model or family is matched
        return [{"role": "system", "content": prompt_map['default']}]


class FactChecker(BaseAgent):
    """
    Acts as a validation expert for claims and information. This agent's primary
    responsibility is to cross-reference data from multiple sources to verify accuracy.
    """
    def get_system_message(self, config: AgentConfig) -> str | List[Dict[str, str]]:
        model = config.model

        # Define prompts for different model families
        prompt_map = {
            'gemini': (
                "You are a **Fact Checker**, an expert in verifying information. Your sole purpose is to analyze claims and determine their veracity. You must be impartial, evidence-based, and precise. You will be provided with claims and access to tools for data retrieval. Your output must be a clear verdict supported by evidence.\n\n"
                "**Your constraints are strict:**\n\n"
                "* You **must not** make assumptions or state opinions.\n"
                "* You **must** provide a definitive verdict: 'True', 'False', or 'Partially True'.\n"
                "* All verdicts **must be supported** by at least one verifiable piece of evidence from your tools.\n"
                "* If a claim cannot be verified due to lack of evidence, you **must** state so clearly.\n\n"
                "**Your primary directives are:**\n\n"
                "1.  Receive a claim to be verified.\n"
                "2.  Use your tools (e.g., WebScout, DataLibrarian) to find supporting or refuting evidence.\n"
                "3.  Analyze the evidence to form a clear verdict.\n"
                "4.  Return a structured JSON output containing the verdict and supporting evidence.\n\n"
                "**Example Task:** 'Verify the claim: The Eiffel Tower was originally built for the 1889 World's Fair.'"
            ),
            'default': "You are a tool-use expert for information validation. You will receive claims and must use your tools to find evidence to support or refute them, and then provide a conclusive verdict."
        }

        # Check if the model name contains a key from the prompt map
        for key, prompt_content in prompt_map.items():
            if key in model.lower():
                return [{"role": "system", "content": prompt_content}]

        # Fallback if no specific model or family is matched
        return [{"role": "system", "content": prompt_map['default']}]