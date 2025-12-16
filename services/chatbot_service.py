import asyncio
from datetime import datetime
from typing import List, Optional
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from graphiti_core.nodes import EpisodeType

from prompts import ImmigrationChatbotPrompts
from scripts.neo4j_integration_uscode import traverse_law_graph


class ChatbotService:
    
    
    def __init__(self, graphiti_client, llm_api_key: str, dbURI: str="", dbUser: str="", dbPassword:str="", model_name: str = "gemini-pro", full_search=False, db_name="neo4j"):
        self.graphiti = graphiti_client
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=llm_api_key,
            temperature=0.15
        )
        
        self.prompts = ImmigrationChatbotPrompts
        self.search_cont=full_search

        self.dbURI = dbURI
        self.dbUser = dbUser
        self.dbPassword = dbPassword
        
    async def search_knowledge_graph(self, query: str, limit: int = 15, law_nodes_limit=10):
        try:
            if self.search_cont:
               
                results = await self.graphiti.searchGraphFull(query=query, limit=limit)

                
                facts = []
                if results.edges:
                    facts = [edge.fact for edge in results.edges]

              
                response = {'facts': facts}

                statute_references = []
                if results.nodes:
                    nodes=[]
                    for node in results.nodes:
                        node_info ={'name': node.name if hasattr(node, 'name') else None,
                                    'label': " ".join(node.labels) if hasattr(node, 'labels') else None,
                                    'summary': node.summary if hasattr(node, 'summary') else None
                                    } 
                        nodes.append(node_info)

                        if "StatuteReference" in node.labels:
                            statute_references.append(node)
                        
                    response['nodes'] = nodes

                if statute_references:
                    response['law_nodes'] = traverse_law_graph(self.dbURI, self.dbUser, self.dbPassword, statute_references)[:law_nodes_limit]

                
                return response

            else:
                
                results = await self.graphiti.searchGraph(query, limit=limit)

                if results and len(results) > 0:
                    facts = [edge.fact for edge in results]
                    return {'facts': facts}
                else:
                    return {'facts': []}
                
        except Exception as e:
            print(f"Error searching knowledge graph: {e}")
            return {'facts': []}

    
    async def generate_response(self,user_message: str,conversation_history: List[BaseMessage],include_nodes: bool=False, limit=10, law_node_limit=10):
        facts = await self.search_knowledge_graph(user_message, limit, law_node_limit)
        
        
        knowledge_context = self.prompts.format_knowledge_context( facts=facts['facts'],nodes=facts.get('nodes', []), law_nodes=facts.get('law_nodes', []))
        
        sys_prompt = self.prompts.build_system_prompt(knowledge_context)
        
        queries = [SystemMessage(content=sys_prompt),*conversation_history,HumanMessage(content=user_message)
        ]
        
        res = await self.llm.ainvoke(queries)
        
        return res.content, knowledge_context
    
    async def _persist_to_graph(self, user_message: str, assistant_response: str):
        try:
            episode_body = f"User asked: {user_message}\n\nAssistant responded: {assistant_response}"
            
            await self.graphiti.addEpisodes([
            {
                "name": "Immigration Consultation",
                "episode_body": episode_body,
                "source": EpisodeType.message,
                "reference_time": datetime.now(),
                "source_description": "Legal Chatbot Conversation"
            }
        ])
        except Exception as e:
            print(f"Error persisting to graph: {e}")
