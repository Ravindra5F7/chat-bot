import json
from langchain_ollama import ChatOllama
from kg_utils import load_knowledge_graph, create_kg_embeddings, find_relevant_nodes, format_kg_context

# Configuration (adjust these if your setup differs)
OLLAMA_BASE_URL = "http://172.180.9.187:11434/"  # Replace with your Ollama server URL
EMBEDDINGS_MODEL = "mxbai-embed-large" # Ensure this model is available in your Ollama instance
# IMPORTANT: Replace "your-working-llm-model" with the actual name of a working LLM model in your Ollama instance
LLM_MODEL = "llama2"
 # <-- Make sure you replace this with your actual model name
KG_FILE_PATH = "../../KG/kg1.json" # <-- Corrected path
TOP_K_NODES = 5 # Number of relevant KG nodes to include in the prompt context

def main():
    # Check if a valid LLM model is configured
    if LLM_MODEL == "your-working-llm-model":
        print("Error: Please update LLM_MODEL in main_app.py with the name of a working Ollama model.")
        return

    print("Loading and processing Knowledge Graph...")
    kg_data = load_knowledge_graph(KG_FILE_PATH)
    if not kg_data:
        print("Failed to load knowledge graph. Exiting.")
        return

    embedded_kg = create_kg_embeddings(kg_data, EMBEDDINGS_MODEL, OLLAMA_BASE_URL)
    if not embedded_kg:
        print("Failed to create knowledge graph embeddings. Exiting.")
        return

    print(f"Initializing Ollama Chat Model with model: {LLM_MODEL}...")
    try:
        llm = ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        # Optional: A quick test to ensure LLM is reachable
        # llm.invoke("Hello")
        print("Ollama Chat Model initialized.")
    except Exception as e:
        print(f"Error initializing Ollama Chat Model: {e}")
        print("Please ensure Ollama is running and the specified model is available.")
        return


    print("\\nKnowledge Graph Integration Ready!")
    print("Type your query or 'quit' to exit.")

    while True:
        user_query = input("\\nYour Query: ")
        if user_query.lower() == 'quit':
            break

        # 1. Retrieve relevant KG info
        relevant_nodes = find_relevant_nodes(user_query, embedded_kg, EMBEDDINGS_MODEL, OLLAMA_BASE_URL, TOP_K_NODES)
        kg_context = format_kg_context(relevant_nodes)

        # 2. Construct the prompt for Ollama
        # We instruct Ollama to use the provided context
        prompt = f"Using the following knowledge graph information, answer the question accurately. If the information is not sufficient, state that you cannot fully answer based on the provided context.\\n\\n{kg_context}\\n\\nQuestion: {user_query}\\n\\nAnswer:"

        # 3. Get response from Ollama
        print("Sending query to Ollama...")
        try:
            response = llm.invoke(prompt)
            print("\\nOllama's Response:")
            print(response.content) # Assuming response is a Langchain message object
        except Exception as e:
            print(f"Error getting response from Ollama: {e}")

    print("Exiting.")

if __name__ == "__main__":
    main()
