import json
import re
from copy import deepcopy
from typing import Any, Dict, List

from auraflux_core.core.agents.base_agent import BaseAgent
from auraflux_core.core.clients.client_manager import ClientManager
from auraflux_core.core.schemas.agents import AgentConfig
from auraflux_core.core.schemas.messages import Message
from auraflux_core.knowledge import tools
from auraflux_core.knowledge.schemas import (GoogleSearchAgentConfig,
                                             GoogleSearchToolConfig,
                                             ToolConfig)
from auraflux_core.knowledge.tools import BaseTool


class GraphBuilderAgent(BaseAgent):

    def __init__(self, config: AgentConfig, client_manager: ClientManager):
        super().__init__(config, client_manager)
        self.config.tool_use = 'TOOL_USE_DIRECT'
        self.max_prompt_length = 30000
        self.tool = tools.MindMapGraphBuilder(config=ToolConfig())

    async def generate(self, messages: List[Message]) -> Message:
        last_message = deepcopy(messages[-1])
        message_content = json.loads(last_message.content)
        data_payloads = []
        for content in message_content:
            data_payload = content.get('data', '')
            if not data_payload:
                continue

            data_payloads += self._split_data_payload(data_payload)

        for data_payload in data_payloads:
            mind_map_message = await self.generate_llm_message([Message(role='user', content=data_payload, name=self.name)])
            mind_map_json_output = self.postprocess_llm_output(mind_map_message.content)
            await self.tool.run(json_output=mind_map_json_output)

        node_labels = ', '.join([label for label, label_type in self.tool.graph.all_nodes if label_type in ['CentralIdea', 'MainTopic']])
        return Message(role='assistant', content=node_labels, name=self.name)

    def _split_data_payload(self, data_payload: str) -> List[str]:
        """
        Splits a data payload string into a list of smaller strings,
        each not exceeding self.max_prompt_length, while preserving line breaks.
        """
        lines  = data_payload.split('\n')
        chunks: List[str] = []
        current_chunk  = ''
        current_length  = 0

        for line in lines:
            line_with_newline = line + '\n'
            line_length = len(line_with_newline)

            if current_length + line_length > self.max_prompt_length and current_chunk:
                chunks.append(current_chunk)
                current_chunk = line_with_newline
                current_length = line_length
            else:
                current_chunk += line_with_newline
                current_length += line_length

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def postprocess_llm_output(self, message_content: str) -> str:
        self.logger.debug(f'message_content: {message_content}')
        return message_content.replace('```json\n', '').replace('\n```', '')

    def get_system_message_map(self) -> Dict[str, str]:
        return {
            'en': "You are a master at analyzing documents and extracting a complete mind map structure. Your task is to process the following text and create a JSON object that strictly represents a mind map based on the provided schema. Do not include any extra text or conversation. Only output the JSON object.\n\nThe output must contain:\n1. A single \"central_idea\" with a \"label\" and \"type\".\n2. An array of \"main_topics\", each representing a \"MainTopic\" with its own \"label\", \"type\", and \"sub_topics\" array.\n3. Within each \"sub_branch\", an array of \"keywords\", each with its own \"label\" and \"type\".\n4. An optional \"relationships\" array at the end for non-hierarchical connections.\n\nThe available entity types are: \"CentralIdea\", \"MainTopic\", \"SubTopic\", and \"Keyword\".\nThe available relationship types are: \"SUPPORTS\", \"LEADS_TO\", \"RELATED_TO\".\n\nJSON Schema:\n{\n  \"central_idea\": {\n    \"label\": \"string\",\n    \"type\": \"string\"\n  },\n  \"branches\": [\n    {\n      \"label\": \"string\",\n      \"type\": \"string\",\n      \"sub_branches\": [\n        {\n          \"label\": \"string\",\n          \"type\": \"string\",\n          \"keywords\": [\n            {\n              \"label\": \"string\",\n              \"type\": \"string\"\n            }\n          ]\n        }\n      ]\n    }\n  ],\n  \"relationships\": [\n    {\n      \"source\": \"string\",\n      \"target\": \"string\",\n      \"type\": \"string\"\n    }\n  ]\n}",
            'default': "You are a tool-use expert for information validation. You will receive claims and must use your tools to find evidence to support or refute them, and then provide a conclusive verdict."
        }

    def get_tool_call(self, messages: List[Message]) -> Dict[str, Any]:
        return {
            "tool": self.tool.get_name(),
            "args": {
                "query": messages[-1].content
            }
        }

    def get_tool_map(self) -> Dict[str, BaseTool | None]:
        return {
            self.tool.get_name(): self.tool
        }


class SearchAgent(BaseAgent):
    """
    Acts as an expert search agent responsible for finding relevant web pages
    based on a natural language query.
    """
    def __init__(self, config: GoogleSearchAgentConfig, client_manager: ClientManager):
        super().__init__(config, client_manager)
        self.config.tool_use = 'TOOL_USE_DIRECT'
        self.tool = tools.GoogleSearchTool(config=GoogleSearchToolConfig(
            google_search_engine_id=config.google_search_engine_id,
            google_search_engine_api_key=config.google_search_engine_api_key,
            google_search_engine_base_url=config.google_search_engine_base_url,
        ))

    def get_system_message_map(self) -> Dict[str, str]:
        return {'default': "This function is not adopted."}

    def get_tool_call(self, messages: List[Message]) -> Dict[str, Any]:
        return {
            "tool": self.tool.get_name(),
            "args": {
                "query": messages[-1].content
            }
        }

    def get_tool_map(self) -> Dict[str, BaseTool | None]:
        return {
            self.tool.get_name(): self.tool
        }
