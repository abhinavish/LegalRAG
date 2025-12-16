from typing import Optional
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate


class ImmigrationChatbotPrompts:
    
    SYSTEM_PROMPT_TEMPLATE = """You are an immigration law assistant helping self-represented litigants 
                                understand US immigration procedures and rights.

                                YOUR KNOWLEDGE BASE CONTEXT: {knowledge_context}

                                **Instructions:**
                                - Use the legal information provided above to answer questions accurately
                                - Explain complex legal concepts in simple, accessible language
                                - Always remind users that this is general information, not legal advice
                                - Recommend consulting with a licensed immigration attorney for specific cases
                                - Be empathetic and supportive - immigration cases can be stressful
                                - Most Importantly, If the knowledge base doesn't contain relevant information, DONT TRY TO ANSWER, JUST ACKNOWLEDGE LIMITATIONS 
                                - FINALLY, MAKE SURE YOU PROVIDE CITATIONS IN BRACKETS FOR ANYTHING YOU USE FROM THE KNOWLEDGE CONTEXT

                                **Disclaimer:** Provide general legal information only. Users should consult an attorney for their specific situation.
                                """

    KNOWLEDGE_CONTEXT_TEMPLATE = """**Relevant Legal Information:**
                                    {facts}

                                    **Source:** Retrieved from legal knowledge base containing US immigration law, case precedents, and procedural guidelines.
                                    """

    NO_KNOWLEDGE_TEMPLATE = """**Note:** No specific case law or precedents found in knowledge base for this query. 
                                I will provide general immigration information based on my training."""

    FACT_ITEM_TEMPLATE = "• {fact}"
    
    QUERY_ENHANCEMENT_TEMPLATE = """Given the user's immigration question: "{user_query}"

                                    Extract the key legal concepts and entities to search for in the knowledge base.
                                    Focus on visa types, legal procedures, forms, eligibility criteria, and case precedents.

                                    Key search terms:"""

    @classmethod
    def build_system_prompt(cls, knowledge_context: str):
        return cls.SYSTEM_PROMPT_TEMPLATE.format(
            knowledge_context=knowledge_context
        )
    
    @classmethod
    def format_knowledge_context(cls, facts: list[str], nodes, law_nodes):
        if not facts or len(facts) == 0:
            return cls.NO_KNOWLEDGE_TEMPLATE
        
        formatted_facts = "\n".join([
            cls.FACT_ITEM_TEMPLATE.format(fact=fact) 
            for fact in facts 
        ])

        context_parts = [
            "**Relevant Legal Information:**",
            formatted_facts
        ]

        if nodes and len(nodes) > 0:
            entity_list = "\n\n".join([
                f" - Entity Name: {node.get('name', 'Unknown')}\n -- Entity Type: ({node.get('label', 'Entity')})\n -- Summary: {node.get('summary', 'No summary available')}"
                for node in nodes 
            ])
            context_parts.append(f"\n**Related Entities:**\n{entity_list}")

        if law_nodes and len(law_nodes) > 0:
            entity_list = "\n\n".join([
                f" - Law Name: {law_node.get('name', 'Unknown')}\n -- USLM ID: ({law_node.get('uslm_identifier', 'Unknown')})\n -- Summary: {law_node.get('summary', 'No summary available')}"
                for law_node in law_nodes
            ])
            context_parts.append(f"\n**Related Laws:**\n{entity_list}")

        context_parts.append(
            "\n**Source:** Retrieved from legal knowledge base containing US immigration law, "
            "case precedents, and procedural guidelines."
        )

        return "\n".join(context_parts)
    
    @classmethod
    def get_langchain_prompt_template(cls):
        return PromptTemplate(
            input_variables=["knowledge_context"],
            template=cls.SYSTEM_PROMPT_TEMPLATE
        )
    
    @classmethod
    def get_chat_prompt_template(cls):
        system_template = SystemMessagePromptTemplate.from_template(
            cls.SYSTEM_PROMPT_TEMPLATE
        )
        
        return ChatPromptTemplate.from_messages([
            system_template,
            ("human", "{user_message}")
        ])