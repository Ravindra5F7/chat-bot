import json
from langchain_ollama import OllamaEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_knowledge_graph(kg_path="KG/kg1.json"):
    """Loads the knowledge graph from a JSON file."""
    try:
        with open(kg_path, 'r') as f:
            kg_data = json.load(f)
        print(f"Successfully loaded knowledge graph from {kg_path}")
        return kg_data
    except FileNotFoundError:
        print(f"Error: Knowledge graph file not found at {kg_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {kg_path}")
        return None

def create_kg_embeddings(kg_data, embeddings_model_name, ollama_base_url):
    """Creates embeddings for the knowledge graph nodes."""
    if not kg_data or "nodes" not in kg_data:
        print("No knowledge graph data or nodes found to embed.")
        return []

    print(f"Creating embeddings for {len(kg_data['nodes'])} KG nodes using {embeddings_model_name}...")
    embeddings = OllamaEmbeddings(
        model=embeddings_model_name,
        base_url=ollama_base_url
    )

    embedded_kg_data = []
    for node in kg_data["nodes"]:
        # Create a text representation of the node
        text_representation = f"Node ID: {node.get('id', 'N/A')}. Description: {node.get('description', 'No description available.')}"
        try:
            node_embedding = embeddings.embed_query(text_representation)
            embedded_kg_data.append({
                "node": node,
                "embedding": node_embedding
            })
        except Exception as e:
            print(f"Error creating embedding for node {node.get('id', 'N/A')}: {e}")

    print("Finished creating KG embeddings.")
    return embedded_kg_data

def find_relevant_nodes(query, embedded_kg_data, embeddings_model_name, ollama_base_url, top_k=3):
    """Finds the most relevant knowledge graph nodes based on a query using embeddings."""
    if not embedded_kg_data:
        print("No embedded knowledge graph data available for search.")
        return []

    embeddings = OllamaEmbeddings(
        model=embeddings_model_name,
        base_url=ollama_base_url
    )

    try:
        query_embedding = embeddings.embed_query(query)
    except Exception as e:
        print(f"Error creating embedding for query: {e}")
        return []

    # Calculate cosine similarity between query embedding and all node embeddings
    # Reshape query_embedding for cosine_similarity function
    query_embedding_reshaped = np.array(query_embedding).reshape(1, -1)
    node_embeddings = np.array([item["embedding"] for item in embedded_kg_data])

    # Handle case where node_embeddings might be empty or have wrong shape
    if node_embeddings.shape[0] == 0 or node_embeddings.shape[1] != query_embedding_reshaped.shape[1]:
         print("Error: Mismatch in embedding dimensions or no node embeddings.")
         return []


    similarities = cosine_similarity(query_embedding_reshaped, node_embeddings)[0]

    # Get indices of top_k most similar nodes
    top_k_indices = similarities.argsort()[-top_k:][::-1]

    # Retrieve the original node data for the top_k indices
    relevant_nodes = [embedded_kg_data[i]["node"] for i in top_k_indices]

    return relevant_nodes

def format_kg_context(relevant_nodes):
    """Formats the relevant KG nodes into a string for the LLM prompt."""
    if not relevant_nodes:
        return "No specific knowledge graph information found."

    context = "Knowledge Graph Information:\\n"
    for i, node in enumerate(relevant_nodes):
        context += f"{i+1}. Node ID: {node.get('id', 'N/A')}\\n"
        if node.get('description'):
             context += f"   Description: {node['description']}\\n"
        # You could add parent/child info here if desired
        # if node.get('parent'):
        #     context += f"   Parent: {node['parent']}\\n"
        context += "\\n" # Add a newline for separation

    return context.strip() # Remove trailing newline/space
