# FusedChatbot/server/ai_core_service/app.py
import os
import sys
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

try:
    from . import config
    from . import file_parser
    from . import faiss_handler
    from . import llm_handler 
except ImportError as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s')
    logging.error(f"ImportError: {e}. Attempting to adjust sys.path...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    import config 
    import file_parser
    import faiss_handler
    import llm_handler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

def create_error_response(message, status_code=500):
    logger.error(f"API Error Response ({status_code}): {message}")
    return jsonify({"error": message, "status": "error"}), status_code

@app.route('/health', methods=['GET'])
def health_check():
    logger.info("\n--- Received request at /health ---")
    status_details = {
        "status": "error", 
        "embedding_model_type": config.EMBEDDING_TYPE,
        "embedding_model_name": config.EMBEDDING_MODEL_NAME,
        "embedding_dimension": None, 
        "sentence_transformer_load": "Unknown",
        "default_index_loaded": False,
        "default_index_vectors": 0,
        "default_index_dim": None,
        "gemini_configured": False, 
        "ollama_base_url_configured": False, 
        "ollama_default_model": None, 
        "groq_configured": False,
        "message": ""
    }
    http_status_code = 503 
    try:
        model = faiss_handler.embedding_model
        if model is None:
            status_details["message"] = "Embedding model could not be initialized."
            status_details["sentence_transformer_load"] = "Failed"
            raise RuntimeError(status_details["message"])
        else:
            status_details["sentence_transformer_load"] = "OK"
            try: status_details["embedding_dimension"] = faiss_handler.get_embedding_dimension(model)
            except Exception as dim_err: status_details["embedding_dimension"] = f"Error: {dim_err}"

        if config.DEFAULT_INDEX_USER_ID in faiss_handler.loaded_indices:
            status_details["default_index_loaded"] = True
            default_index = faiss_handler.loaded_indices[config.DEFAULT_INDEX_USER_ID]
            if hasattr(default_index, 'index') and default_index.index:
                status_details["default_index_vectors"] = default_index.index.ntotal
                status_details["default_index_dim"] = default_index.index.d
        else:
            try:
                default_index = faiss_handler.load_or_create_index(config.DEFAULT_INDEX_USER_ID)
                status_details["default_index_loaded"] = True
                if hasattr(default_index, 'index') and default_index.index:
                    status_details["default_index_vectors"] = default_index.index.ntotal
                    status_details["default_index_dim"] = default_index.index.d
            except Exception as index_load_err:
                status_details["message"] = f"Failed to load default index: {index_load_err}"
                status_details["default_index_loaded"] = False
        
        status_details["gemini_configured"] = llm_handler.gemini_sdk_configured
        status_details["ollama_base_url_configured"] = bool(llm_handler.OLLAMA_BASE_URL and llm_handler.ollama_available)
        status_details["ollama_default_model"] = llm_handler.DEFAULT_OLLAMA_MODEL
        status_details["groq_configured"] = bool(llm_handler.groq_client)

        if status_details["sentence_transformer_load"] == "OK" and status_details["default_index_loaded"]:
            status_details["status"] = "ok"; status_details["message"] = "AI Core service is running. Embeddings and default index OK."
            http_status_code = 200
            llm_notes = [note for configured, note in [
                (status_details["gemini_configured"], "Gemini not fully configured"),
                (status_details["ollama_base_url_configured"], "Ollama not fully configured"),
                (status_details["groq_configured"], "Groq not fully configured")
            ] if not configured]
            if llm_notes: status_details["message"] += " LLM Status: " + "; ".join(llm_notes) + "."
        else:
            status_details["status"] = "error"
            if not status_details["message"]: status_details["message"] = "AI Core service has issues with embeddings or default index."
            http_status_code = 503
        logger.info(f"Health check completed. Status: {status_details['status']}.")
    except Exception as e:
        logger.error(f"--- Health Check Critical Error ---", exc_info=True)
        if not status_details["message"]: status_details["message"] = f"Health check failed critically: {str(e)}"
        status_details["status"] = "error"; http_status_code = 503
    return jsonify(status_details), http_status_code

@app.route('/add_document', methods=['POST'])
def add_document():
    logger.info("\n--- Received request at /add_document ---")
    if not request.is_json:
        return create_error_response("Request must be JSON", 400)
    data = request.get_json()
    if data is None:
        return create_error_response("Invalid or empty JSON body", 400)
        
    user_id = data.get('user_id')
    file_path = data.get('file_path')
    original_name = data.get('original_name')

    missing_fields = []
    if not user_id: missing_fields.append("user_id")
    if not file_path: missing_fields.append("file_path")
    if not original_name: missing_fields.append("original_name")
    if missing_fields:
        error_msg = f"Missing required fields: {', '.join(missing_fields)}"
        return create_error_response(error_msg, 400)

    logger.info(f"Processing file: {original_name} for user: {user_id}")
    logger.debug(f"File path: {file_path}")

    if not os.path.exists(file_path):
        return create_error_response(f"File not found at path: {file_path}", 404)
    try:
        text_content = file_parser.parse_file(file_path)
        if text_content is None:
            return jsonify({"message": f"File type of '{original_name}' not supported or parsing failed.", "filename": original_name, "status": "skipped"}), 200
        if not text_content.strip():
            return jsonify({"message": f"No text content extracted from '{original_name}'.", "filename": original_name, "status": "skipped"}), 200
        documents = file_parser.chunk_text(text_content, original_name, user_id)
        if not documents:
            return jsonify({"message": f"No text chunks generated for '{original_name}'.", "filename": original_name, "status": "skipped"}), 200
        faiss_handler.add_documents_to_index(user_id, documents)
        return jsonify({"message": f"Document '{original_name}' processed and added to index.", "filename": original_name, "chunks_added": len(documents), "status": "added"}), 200
    except Exception as e:
        logger.error(f"--- Add Document Error for file '{original_name}' ---", exc_info=True)
        return create_error_response(f"Failed to process document '{original_name}': {str(e)}", 500)


@app.route('/query_rag_documents', methods=['POST'])
def query_rag_documents_route():
    logger.info("\n--- Received request at /query_rag_documents ---")
    if not request.is_json: return create_error_response("Request must be JSON", 400)
    data = request.get_json()
    if data is None: return create_error_response("Invalid or empty JSON body", 400)
    user_id = data.get('user_id'); query_text = data.get('query'); k = data.get('k', 5)
    if not user_id or not query_text: return create_error_response("Missing user_id or query", 400)
    logger.info(f"Querying RAG for user: {user_id} with k={k}"); logger.debug(f"Query text: '{query_text[:100]}...'")
    try:
        results = faiss_handler.query_index(user_id, query_text, k=k)
        formatted_results = [{"documentName": doc.metadata.get("documentName", "Unknown"), "score": float(score), "content": doc.page_content} for doc, score in results]
        return jsonify({"relevantDocs": formatted_results, "status": "success"}), 200
    except Exception as e:
        logger.error(f"--- RAG Query Error ---", exc_info=True)
        return create_error_response(f"Failed to query RAG index: {str(e)}", 500)

@app.route('/analyze_document', methods=['POST'])
def analyze_document_route():
    logger.info("\n--- Received request at /analyze_document ---")
    if not request.is_json:
        return create_error_response("Request must be JSON", 400)
    data = request.get_json()
    if data is None:
        return create_error_response("Invalid or empty JSON body", 400)

    user_id = data.get('user_id') 
    document_name = data.get('document_name') 
    analysis_type = data.get('analysis_type') 
    
    llm_provider = data.get('llm_provider', config.DEFAULT_LLM_PROVIDER)
    llm_model_name = data.get('llm_model_name', None)

    # --- Validate required fields ---
    missing_fields = []
    if not user_id: missing_fields.append("user_id") 
    if not document_name: missing_fields.append("document_name")
    if not analysis_type: missing_fields.append("analysis_type")
    
    # --- MODIFICATION for standalone testing: Expect 'file_path' for now ---
    file_path_for_analysis = data.get('file_path_for_analysis') 
    if not file_path_for_analysis:
         missing_fields.append("file_path_for_analysis (for standalone testing)")
    
    if missing_fields:
        return create_error_response(f"Missing required fields for analysis: {', '.join(missing_fields)}", 400)

    if analysis_type not in llm_handler.ANALYSIS_PROMPTS: 
        return create_error_response(f"Invalid analysis_type: '{analysis_type}'. Supported types are: {list(llm_handler.ANALYSIS_PROMPTS.keys())}", 400)

    if not os.path.exists(file_path_for_analysis):
        logger.error(f"Document file for analysis not found at specified path: {file_path_for_analysis}")
        return create_error_response(f"Document '{document_name}' not found at path for analysis.", 404)

    document_text = ""
    try:
        logger.info(f"Reading document content from: {file_path_for_analysis} for analysis type '{analysis_type}'")
        document_text = file_parser.parse_file(file_path_for_analysis)
        if document_text is None:
            raise ValueError(f"Could not parse or extract text from file: {file_path_for_analysis}")
        if not document_text.strip():
            raise ValueError(f"Extracted text content is empty for file: {file_path_for_analysis}")
        logger.info(f"Successfully loaded text for '{document_name}' for analysis (length: {len(document_text)}).")
    except Exception as e:
        logger.error(f"Error reading or parsing document '{document_name}' from path '{file_path_for_analysis}' for analysis: {e}", exc_info=True)
        return create_error_response(f"Error accessing or parsing document '{document_name}': {e}", 500)

    # --- Perform Analysis ---
    try:
        logger.info(f"Performing '{analysis_type}' analysis on '{document_name}' using {llm_provider}...")
        
        analysis_result, thinking_content = llm_handler.perform_document_analysis(
            document_text=document_text,
            analysis_type=analysis_type,
            llm_provider=llm_provider,
            llm_model_name=llm_model_name
        )

        if analysis_result is None and thinking_content is None: 
             return create_error_response(f"Analysis for '{analysis_type}' failed to produce any output.", 500)


        response_payload = {
            "document_name": document_name,
            "analysis_type": analysis_type,
            "analysis_result": analysis_result,
            "thinking_content": thinking_content,
            "status": "success"
        }
        return jsonify(response_payload), 200

    except ValueError as ve: 
        logger.error(f"Value error during document analysis: {ve}", exc_info=True)
        return create_error_response(str(ve), 400)
    except ConnectionError as ce: 
        logger.error(f"Connection error with LLM service during analysis: {ce}", exc_info=True)
        return create_error_response(str(ce), 502) 
    except Exception as e:
        logger.error(f"--- Document Analysis Error ---", exc_info=True)
        return create_error_response(f"Failed to perform document analysis for '{analysis_type}': {str(e)}", 500)

# --- MODIFIED /generate_chat_response for CoT ---
@app.route('/generate_chat_response', methods=['POST'])
def generate_chat_response_route():
    logger.info("\n--- Received request at /generate_chat_response ---")
    if not request.is_json: return create_error_response("Request must be JSON", 400)
    data = request.get_json()
    if data is None: return create_error_response("Invalid or empty JSON body", 400)

    user_id = data.get('user_id')
    raw_chat_history = data.get('chat_history', []) 
    current_user_query = data.get('query') 
    
    llm_provider = data.get('llm_provider', config.DEFAULT_LLM_PROVIDER)
    llm_model_name = data.get('llm_model_name', None)
    system_prompt_from_request = data.get('system_prompt', None) 
    
    perform_rag = data.get('perform_rag', True)
    enable_multi_query = data.get('enable_multi_query', True) 
    num_sub_queries_to_gen = data.get('num_sub_queries', config.MULTI_QUERY_COUNT_CONFIG)
    rag_k_per_query = data.get('rag_k_per_query', config.DEFAULT_RAG_K_PER_SUBQUERY_CONFIG)

    if not user_id or not current_user_query:
        return create_error_response("Missing user_id or current_user_query in request", 400)
    if not isinstance(raw_chat_history, list):
        raw_chat_history = []

    chat_history_for_llm_call = list(raw_chat_history)

    retrieved_docs_for_context_build = [] 
    rag_references_for_client = []
    unique_retrieved_chunks_content = set()

    if perform_rag:
        logger.info(f"RAG enabled for user: {user_id} with query: '{current_user_query[:50]}...'")
        queries_to_search = [current_user_query]
        if enable_multi_query and num_sub_queries_to_gen > 0:
            try:
                logger.info(f"Attempting to generate {num_sub_queries_to_gen} sub-queries...")
                provider_available_for_subquery = True
                if llm_provider == "gemini" and not llm_handler.gemini_sdk_configured: provider_available_for_subquery = False
                elif llm_provider == "ollama" and not llm_handler.ollama_available: provider_available_for_subquery = False
                elif llm_provider == "groq_llama3" and not llm_handler.groq_client: provider_available_for_subquery = False
                
                if not provider_available_for_subquery:
                    logger.warning(f"LLM provider '{llm_provider}' for sub-query generation is not available/configured. Skipping sub-query generation.")
                else:
                    sub_queries = llm_handler.generate_sub_queries_via_llm(
                        original_query=current_user_query, llm_provider=llm_provider,
                        llm_model_name=llm_model_name, num_sub_queries=num_sub_queries_to_gen)
                    if sub_queries:
                        logger.info(f"Generated sub-queries: {sub_queries}")
                        queries_to_search.extend(sub_queries)
                    else: logger.info("No sub-queries were generated by the LLM.")
            except Exception as e: logger.error(f"Error during sub-query generation: {e}", exc_info=True)
        
        logger.info(f"Total queries for RAG search: {queries_to_search}")
        for i, query_item in enumerate(queries_to_search):
            logger.info(f"Performing RAG search for query {i+1}/{len(queries_to_search)}: '{query_item[:50]}...' (k={rag_k_per_query})")
            try:
                rag_results_for_item = faiss_handler.query_index(user_id, query_item, k=rag_k_per_query)
                if rag_results_for_item:
                    for doc, score in rag_results_for_item:
                        chunk_content_for_key = doc.page_content 
                        if chunk_content_for_key not in unique_retrieved_chunks_content:
                            unique_retrieved_chunks_content.add(chunk_content_for_key)
                            doc_name = doc.metadata.get("documentName", "Unknown")
                            retrieved_docs_for_context_build.append({"documentName": doc_name, "content": doc.page_content, "score": float(score)})
                            rag_references_for_client.append({"documentName": doc_name, "score": float(score), "preview_snippet": doc.page_content[:config.REFERENCE_SNIPPET_LENGTH] + "..."})
            except Exception as e: logger.error(f"Error during RAG search for query_item '{query_item[:50]}...': {e}", exc_info=True)
        
        if retrieved_docs_for_context_build: logger.info(f"Total unique RAG chunks collected for context: {len(retrieved_docs_for_context_build)}.")
        else: logger.info("No relevant documents found from RAG across all queries.")

    context_text_for_llm = "No relevant context was found in the available documents."
    if retrieved_docs_for_context_build:
        formatted_context_parts = []
        for i, doc_info in enumerate(retrieved_docs_for_context_build):
            context_str = f"[{i+1}] Source: {doc_info['documentName']}\n{doc_info['content']}"
            formatted_context_parts.append(context_str)
        context_text_for_llm = "\n\n---\n\n".join(formatted_context_parts)
    
    logger.debug(f"Context text for LLM (first 300 chars): {context_text_for_llm[:300]}...")

    try:
        logger.info(f"Calling LLM provider for CoT synthesis: {llm_provider} for user: {user_id}. Model: {llm_model_name or 'provider_default'}")
        
        final_answer, thinking_content = llm_handler.generate_response(
            llm_provider=llm_provider,
            query=current_user_query, 
            context_text=context_text_for_llm, 
            chat_history=chat_history_for_llm_call, 
            system_prompt_text=system_prompt_from_request, 
            relevant_docs=retrieved_docs_for_context_build, 
            llm_model_name=llm_model_name
        )
        
        response_payload = {
            "llm_response": final_answer,
            "references": rag_references_for_client,
            "thinking_content": thinking_content, 
            "status": "success"
        }
        return jsonify(response_payload), 200

    except ValueError as ve: return create_error_response(str(ve), 400)
    except ConnectionError as ce: return create_error_response(str(ce), 502)
    except Exception as e:
        logger.error(f"--- Chat Generation Error during CoT synthesis ---", exc_info=True)
        return create_error_response(f"Failed to generate chat response: {str(e)}", 500)

if __name__ == '__main__':
    try:
        logger.info("Initializing AI Core Service...")
        faiss_handler.ensure_faiss_dir()
        logger.info(f"FAISS index directory ensured at: {config.FAISS_INDEX_DIR}")
        faiss_handler.get_embedding_model()
        logger.info(f"Embedding model '{config.EMBEDDING_MODEL_NAME}' initialized successfully on startup.")
        logger.info(f"Expected Embedding Dimension: {faiss_handler.get_embedding_dimension(faiss_handler.embedding_model)}")
        faiss_handler.load_or_create_index(config.DEFAULT_INDEX_USER_ID)
        logger.info(f"Default index '{config.DEFAULT_INDEX_USER_ID}' loaded/checked/created successfully on startup.")
    except Exception as e:
        logger.critical(f"CRITICAL STARTUP FAILURE: {e}", exc_info=True)
        sys.exit(1)

    port = config.AI_CORE_SERVICE_PORT 
    host = '0.0.0.0'
    logger.info(f"--- Starting AI Core Service (Flask App) ---")
    logger.info(f"Service will be available at: http://{host}:{port}")
    logger.info(f"Using Embedding: {config.EMBEDDING_TYPE} ({config.EMBEDDING_MODEL_NAME})")
    logger.info(f"Gemini Configured: {llm_handler.gemini_sdk_configured}")
    logger.info(f"Ollama Available: {llm_handler.ollama_available} (URL: {llm_handler.OLLAMA_BASE_URL}, Default Model: {llm_handler.DEFAULT_OLLAMA_MODEL})")
    logger.info(f"Groq Client Initialized: {bool(llm_handler.groq_client)} (Default Model: {llm_handler.DEFAULT_GROQ_LLAMA3_MODEL})")
    logger.info(f"Default LLM Provider from config: {config.DEFAULT_LLM_PROVIDER}")
    logger.info(f"Multi-query count from config: {config.MULTI_QUERY_COUNT_CONFIG}, RAG K per sub-query from config: {config.DEFAULT_RAG_K_PER_SUBQUERY_CONFIG}")
    logger.info("---------------------------------------------")
    app.run(host=host, port=port, debug=os.getenv('FLASK_DEBUG', 'false').lower() == 'true')