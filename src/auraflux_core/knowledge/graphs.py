from typing import Any, Dict, List, Literal, Set, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field


class Node(BaseModel):
    label: str
    node_id: str = Field(default_factory=lambda: str(uuid4()))
    type: Literal['CentralIdea', 'MainTopic', 'SubTopic', 'Keyword']

    class ConfigDict:
        frozen = True # Makes the model immutable and hashable


class Edge(BaseModel):
    edge_id: str = Field(default_factory=lambda: str(uuid4()))
    source: str
    target: str
    type: Literal['SUPPORTS', 'LEADS_TO', 'RELATED_TO']

    class ConfigDict:
        frozen = True


class KeywordNode(Node):
    type: Literal['Keyword'] = 'Keyword'


class SubTopicNode(Node):
    type: Literal['SubTopic'] = 'SubTopic'
    keywords: List[KeywordNode] = Field(default_factory=list)
    unique_keywords: Set[Tuple[str, str]] = Field(default_factory=set, exclude=True)

    def add_keyword(self, keyword: KeywordNode):
        if (keyword.label, keyword.type) not in self.unique_keywords:
            self.keywords.append(keyword)
            self.unique_keywords.add((keyword.label, keyword.type))

class MainTopicNode(Node):
    type: Literal['MainTopic'] = 'MainTopic'
    sub_topics: List[SubTopicNode] = Field(default_factory=list)
    unique_sub_topics: Set[Tuple[str, str]] = Field(default_factory=set, exclude=True)

    def add_sub_topic(self, sub_topic: SubTopicNode):
        if (sub_topic.label, sub_topic.type) not in self.unique_sub_topics:
            self.sub_topics.append(sub_topic)
            self.unique_sub_topics.add((sub_topic.label, sub_topic.type))


class CentralIdeaNode(Node):
    type: Literal['CentralIdea'] = 'CentralIdea'
    main_topics: List[MainTopicNode] = Field(default_factory=list)
    unique_main_topics: Set[Tuple[str, str]] = Field(default_factory=set, exclude=True)

    def add_main_topic(self, main_topic: MainTopicNode):
        if (main_topic.label, main_topic.type) not in self.unique_main_topics:
            self.main_topics.append(main_topic)
            self.unique_main_topics.add((main_topic.label, main_topic.type))


class MindMap(BaseModel):
    central_ideas: Dict[str, CentralIdeaNode] = Field(default_factory=dict)
    all_nodes: Set[Tuple[str, str]] = Field(default_factory=set)
    all_relationships: Set[Tuple[str, str, str]] = Field(default_factory=set)
    unique_central_ideas: Set[Tuple[str, str]] = Field(default_factory=set, exclude=True)

    def add_node(self, node: Node):
        self.all_nodes.add((node.label, node.type))

    def add_relationship(self, relationship: Edge):
        self.all_relationships.add((relationship.source, relationship.target, relationship.type))

    def add_central_idea(self, central_idea: CentralIdeaNode):
        if (central_idea.label, central_idea.type) not in self.unique_central_ideas:
            self.central_ideas[central_idea.label] = central_idea
            self.unique_central_ideas.add((central_idea.label, central_idea.type))

    def add_central_idea_data(self, data: Dict[str, Any]):
        if not data:
            return None

        central_idea_data = {**data}.get('central_idea', {})
        central_idea = CentralIdeaNode(**central_idea_data)

        # If central idea already exists, return the existing one
        # else create a new one
        self.add_central_idea(central_idea)
        self.add_node(central_idea)
        central_idea = self.central_ideas.get(central_idea.label)
        if central_idea is None:
            raise ValueError("Central idea should have been added.")

        for branch_data in data.get('main_topics', []):
            main_topic = MainTopicNode(**branch_data)
            central_idea.add_main_topic(main_topic)
            self.add_node(main_topic)
            self.all_relationships.add((central_idea.label, main_topic.label, ''))

            for sub_branch_data in branch_data.get('sub_topics', []):
                sub_topic = SubTopicNode(**sub_branch_data)
                main_topic.add_sub_topic(sub_topic)
                self.add_node(sub_topic)
                self.all_relationships.add((main_topic.label, sub_topic.label, ''))

                for keyword_data in sub_branch_data.get('keywords', []):
                    keyword = KeywordNode(**keyword_data)
                    sub_topic.add_keyword(keyword)
                    self.add_node(keyword)
                    self.all_relationships.add((sub_topic.label, keyword.label, ''))

        for relationship_data in data.get('relationships', []):
            self.all_relationships.add((relationship_data.get('source'), relationship_data.get('target'), relationship_data.get('type')))

    def to_json(self):
        """
        {
            "nodes": [
                {"id": "A", "label": "Node A"},
                {"id": "B", "label": "Node B"},
                {"id": "C", "label": "Node C"}
            ],
            "edges": [
                {"source": "A", "target": "B", "weight": 1},
                {"source": "B", "target": "C", "weight": 2}
            ]
        }
        """
        return {
            "nodes": [{"id": node_label, "label": node_label} for node_label, node_type in self.all_nodes],
            "edges": [{"source": edge_source, "target": edge_target, "weight": 1} for edge_source, edge_target, edge_type in self.all_relationships]
        }