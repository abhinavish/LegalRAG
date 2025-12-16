
import os
import asyncio
from dotenv import load_dotenv
from clients.graphiti_client import graphitiClient
from services.chatbot_service import ChatbotService
from langchain_core.messages import HumanMessage, AIMessage
import textwrap

load_dotenv()

neo4j_uri = os.getenv('NEO4J_URI')
neo4j_user = os.getenv('NEO4J_USER')
neo4j_password = os.getenv('NEO4J_PASSWORD')
db_name=os.getenv('DB_NAME', "neo4j")
llm_provider = os.getenv('LLM_PROVIDER')
gemini_key = os.getenv('PROVIDER_API_KEY')
llm_model = os.getenv('INGESTION_LLM')
chat_model=os.getenv('CHAT_MODEL')
embedding_model = os.getenv('INGESTION_EMBEDDING')
reranker_model = os.getenv('INGESTION_RERANKER')


async def test_single_query():
    FULL_SEARCH=True
    print("=" * 70)
    print("Testing Immigration Legal Chatbot - Single Query")
    print("=" * 70)
    
    
    print("\n[1/4] Initializing Graphiti client...")
    graphiti = await graphitiClient(
        neo4j_uri, neo4j_user, neo4j_password, 
        gemini_key, llm_model, embedding_model, 
        reranker_model, llm_provider, 
        use_huggingface_embedding=True
    )
    print("Graphiti client initialized")
    
    
    print("\n[2/4] Initializing Chatbot Service...")
    chatbot = ChatbotService(dbURI=neo4j_uri, dbUser=neo4j_user, dbPassword=neo4j_password,
        graphiti_client=graphiti,
        llm_api_key=gemini_key,
        model_name=chat_model,
        full_search=FULL_SEARCH
    )
    print("Chatbot service initialized")
    
    while True:
        query = input("\n👤 You: ").strip()
        
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
    
        

        if not query:
            continue

        try:
            response, knowledge = await chatbot.generate_response(
                query, 
                [],
                include_nodes=FULL_SEARCH
            )
            
            print(f"\nAssistant: {response}")
            
            if knowledge:
                print("\n" + "─" * 70)
                print("Context:")
                print(knowledge)
                print("─" * 70)
            
        except Exception as e:
            print(f"\nError: {str(e)}")
    
    return chatbot, graphiti


async def test_conversation():
    """Test chatbot with multi-turn conversation"""
    print("\n" + "=" * 70)
    print("Testing Immigration Legal Chatbot - Multi-Turn Conversation")
    print("=" * 70)
    
    
    print("\n[1/2] Initializing services...")
    graphiti = await graphitiClient(
        neo4j_uri, neo4j_user, neo4j_password, 
        gemini_key, llm_model, embedding_model, 
        reranker_model, llm_provider, 
        use_huggingface_embedding=True
    )
    
    chatbot = ChatbotService(
        graphiti_client=graphiti,
        llm_api_key=gemini_key,
        model_name=llm_model
    )
    print("Services initialized")
    
   
    conversation_history = []
    
    
    test_queries = [
        "My spouse is a US citizen. How does that help my immigration case?",
        "What forms do I need to file?",
        "How long does the process typically take?"
    ]
    
    print("\n[2/2] Running conversation test...")
    print("=" * 70)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"Turn {i}/{len(test_queries)}")
        print(f"{'='*70}")
        print(f"\n👤 USER: {query}")
        
        
        response = await chatbot.generate_response(query, conversation_history)
        print(f"\n🤖 ASSISTANT: {response}")
        
        
        conversation_history.append(HumanMessage(content=query))
        conversation_history.append(AIMessage(content=response))
        
        
        await asyncio.sleep(1)
    
    print("\n" + "=" * 70)
    print("Conversation test completed!")
    print("=" * 70)


async def test_knowledge_retrieval():
    """Interactive test allowing user to choose search mode"""
    print("\n" + "=" * 70)
    print("Testing Knowledge Graph Retrieval - Interactive")
    print("=" * 70)
    
    
    print("\n[1/3] Initializing Graphiti client...")
    graphiti = await graphitiClient(
        neo4j_uri, neo4j_user, neo4j_password, 
        gemini_key, llm_model, embedding_model, 
        reranker_model, llm_provider, 
        use_huggingface_embedding=True
    )
    
    
    print("Client initialized")
    
    
    print("\n[2/3] Select search mode:")
    print("  1. Partial search (edges only - fast)")
    print("  2. Full search with entities (edges + nodes)")
    
    choice = input("\nEnter choice (1-2, default=1): ").strip() or "1"
    
    
    if choice == "1":
        full_search = False
        include_nodes = False
        mode_name = "PARTIAL (edges only)"
    elif choice == "2":
        full_search = True
        mode_name = "FULL (edges + nodes)"
    else:
        full_search = False
        mode_name = "PARTIAL (edges only - default)"
    
    
    
    chatbot = ChatbotService(
        graphiti_client=graphiti,
        llm_api_key=gemini_key,
        model_name=llm_model,
        full_search=full_search,
        dbURI=neo4j_uri,
        dbUser=neo4j_user,
        dbPassword=neo4j_password,
        db_name=db_name
    )

    query=''
    while True:
        query=input("You: ")

        if query == 'quit':
            break
        results = await chatbot.search_knowledge_graph(query, limit=20, law_nodes_limit=10)

        
        print(f"\nRetrieved {len(results['facts'])} facts", end="")
        if full_search:
            print(f", {len(results.get('nodes', []))} nodes and {len(results.get('law_nodes', []))} law nodes")
        else:
            print()

        print("\n" + "-" * 70)

        
        if results['facts']:
            print("\nFacts:")
            for i, fact in enumerate(results['facts'], 1):
                print(f"\n{i}. {fact}")
        else:
            print("\nNo facts found in knowledge graph")
            print("   Make sure you've ingested immigration law data first!")

        
        if results.get('nodes'):
            print("\n" + "=" * 70)
            print("Entities:")
            print("=" * 70)
            for i, node in enumerate(results['nodes'], 1):
                name = node.get('name', 'Unknown')
                label = node.get('label', 'Entity')
                summary = node.get('summary', 'No summary available')
                print(f"\n{i}. {name} ({label})")
                print(f"\nSummary: {summary}")

        
        if results.get('law_nodes'):
            print("\n" + "=" * 70)
            print("Law Entities:")
            print("=" * 70)
            for i, node in enumerate(results['law_nodes'], 1):
                name = node.get('name', 'Unknown')
                uslm_id = node.get('uslm_identifier', 'n/a')
                summary = node.get('summary', 'No summary available')
                print(f"\n{i}. {name} ({uslm_id})")
                print(f"\nSummary: {summary}")

        print("\n" + "=" * 70)

async def main():
    """Main test runner"""
    print("\nImmigration Legal Chatbot - Test Suite")
    print("=" * 70)
    
    tests = {
        "1": ("Single Query Test", test_single_query),
        "2": ("Multi-Turn Conversation Test", test_conversation),
        "3": ("Knowledge Retrieval Test", test_knowledge_retrieval),
        "4": ("Run All Tests", None)
    }
    
    print("\nAvailable Tests:")
    for key, (name, _) in tests.items():
        print(f"  {key}. {name}")
    
    choice = input("\nSelect test to run (1-4): ").strip()
    
    if choice == "4":
        
        await test_knowledge_retrieval()
        await test_single_query()
        await test_conversation()
    elif choice in tests and tests[choice][1] is not None:
        await tests[choice][1]()
    else:
        print("Invalid choice. Running default single query test...")
        await test_single_query()
    
    print("\nTesting complete!")

if __name__ == "__main__":
    asyncio.run(main())
    
   
   
