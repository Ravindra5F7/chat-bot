# --- START OF FILE app.py ---

import os
import logging
import json
import uuid 
from flask import Flask, request, jsonify, render_template, send_from_directory, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from waitress import serve
from datetime import datetime, timezone
import threading # Import threading
import re # Import re for regex parsing

# Initialize Logging and Configuration
import config
config.setup_logging()
logger = logging.getLogger(__name__)

# Import Core Modules
import database
import ai_core # Your AI logic module
import utils

# ... (rest of your Flask app setup: template_folder, static_folder, CORS, UPLOAD_FOLDER config) ...
backend_dir = os.path.dirname(__file__)
template_folder = os.path.join(backend_dir, 'templates')
static_folder = os.path.join(backend_dir, 'static')

if not os.path.exists(template_folder): logger.error(f"Template folder not found: {template_folder}")
if not os.path.exists(static_folder): logger.error(f"Static folder not found: {static_folder}")

app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)

CORS(app, resources={r"/*": {"origins": "*"}})
logger.info("CORS configured to allow all origins ('*').")

app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024

# ... (initialize_app function and other global app state variables are fine) ...
app_db_ready = False
app_ai_ready = False
app_vector_store_ready = False
app_doc_cache_loaded = False

def initialize_app():
    global app_db_ready, app_ai_ready, app_vector_store_ready, app_doc_cache_loaded
    if hasattr(app, 'initialized') and app.initialized:
        return

    logger.info("--- Starting Application Initialization ---")
    # ... (your existing initialization logic for DB, AI, Vector Store, Doc Cache) ...
    try:
        database.init_db()
        app_db_ready = True
        logger.info("Database initialization successful.")
    except Exception as e:
        logger.critical(f"Database initialization failed: {e}. Chat history will be unavailable.", exc_info=True)
        app_db_ready = False
    
    logger.info("Initializing AI components (LLM, Embeddings)...")
    embed_instance, llm_instance = ai_core.initialize_ai_components()
    if embed_instance and llm_instance:
        app_ai_ready = True
        logger.info("AI components initialized successfully.")
        
        logger.info("Loading FAISS vector store...")
        if ai_core.load_vector_store():
            app_vector_store_ready = True
            index_size = 0
            if ai_core.vector_store and hasattr(ai_core.vector_store, 'index'):
                 index_size = getattr(ai_core.vector_store.index, 'ntotal', 0)
            logger.info(f"FAISS vector store status: Loaded. Index size: {index_size}")
        else:
            app_vector_store_ready = False
            logger.warning("Failed to load FAISS vector store. RAG might start with an empty/new index.")
    else:
        app_ai_ready = False
        app_vector_store_ready = False 
        logger.warning(f"AI components failed to initialize. RAG, Analysis, and KG-enhanced chat will be affected.")

    logger.info("Loading document texts into cache...")
    try:
        ai_core.load_all_document_texts() 
        app_doc_cache_loaded = True
        logger.info(f"Document text cache loaded. Cached {len(ai_core.document_texts_cache)} documents.")
    except Exception as e:
        logger.error(f"Error loading document texts: {e}. Analysis may require on-the-fly extraction.", exc_info=True)
        app_doc_cache_loaded = False
        
    app.initialized = True
    logger.info("--- Application Initialization Complete ---")


@app.before_request
def ensure_initialized():
    if not hasattr(app, 'initialized') or not app.initialized:
        initialize_app()

# --- Function to run KG generation in a background thread ---
def _trigger_kg_generation_background(filename_for_kg):
    """
    Wrapper to call ai_core.generate_knowledge_graph_from_pdf in a new thread.
    This function will run in the background.
    """
    # It's important that functions running in new threads, especially if they use Flask's
    # app context or request context, are handled carefully.
    # For ai_core.generate_knowledge_graph_from_pdf, it primarily uses its own
    # Ollama client and file operations, so it should be relatively safe.
    # If it needed app context, you'd use `with app.app_context():` here.
    logger.info(f"BACKGROUND_KG: Starting KG generation for '{filename_for_kg}' in a background thread.")
    try:
        # We need to ensure ai_core's global LLM/embeddings are available if KG gen relies on them
        # In your current setup, KG generation uses its own _kg_ollama_client.
        # If generate_knowledge_graph_from_pdf needed app_ai_ready components,
        # you might need to pass them or re-initialize them carefully in a thread.
        # However, your KG generation seems self-contained with _initialize_kg_ollama_client.

        # Check if AI components required for KG generation are ready (e.g., KG_MODEL via _kg_ollama_client)
        # The _initialize_kg_ollama_client in ai_core.py handles its own readiness.
        
        result = ai_core.generate_knowledge_graph_from_pdf(filename_for_kg)
        if result:
            logger.info(f"BACKGROUND_KG: Successfully completed KG generation for '{filename_for_kg}'.")
        else:
            logger.error(f"BACKGROUND_KG: KG generation for '{filename_for_kg}' failed or returned no data.")
    except Exception as e:
        logger.error(f"BACKGROUND_KG: Error during background KG generation for '{filename_for_kg}': {e}", exc_info=True)
# --- End background KG function ---


@app.route('/upload', methods=['POST'])
def upload_file():
    # ... (your existing checks for AI readiness, file part, filename, allowed_file) ...
    if not app_ai_ready:
        logger.error("Upload failed: Core AI components (LLM/Embeddings) not initialized.")
        return jsonify({"error": "Cannot process upload: AI components not ready."}), 503
    if not ai_core.embeddings: # Assuming ai_core.embeddings is the embedding model instance
        logger.error("Upload failed: Embeddings model not initialized.")
        return jsonify({"error": "Cannot process upload: Embeddings model not ready."}), 503

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if not file or not file.filename: # Check for file and filename presence
        return jsonify({"error": "No file selected or filename missing"}), 400
    
    original_secure_filename = secure_filename(file.filename)

    # Handle cases where secure_filename might return an empty string (e.g., filename was ".." or similar)
    if not original_secure_filename:
        file_ext_parts = file.filename.rsplit('.', 1)
        file_ext = file_ext_parts[1].lower() if len(file_ext_parts) > 1 else ""
        
        # Ensure the extension is valid, or use a default if config is available
        allowed_exts_no_dot = getattr(config, 'ALLOWED_EXTENSIONS_WITHOUT_DOT', ['pdf', 'txt', 'md'])
        if file_ext not in allowed_exts_no_dot:
            file_ext = "dat" # Fallback extension
        
        original_secure_filename = f"upload_{uuid.uuid4().hex}.{file_ext}"
        logger.warning(f"Original filename was insecure or empty. Generated new filename: {original_secure_filename}")

    if not utils.allowed_file(original_secure_filename): # Ensure utils.allowed_file exists and uses config.ALLOWED_EXTENSIONS
        allowed_exts_str = ', '.join(getattr(config, 'ALLOWED_EXTENSIONS', ['.pdf', '.txt']))
        return jsonify({"error": f"Invalid file type. Only {allowed_exts_str} files are allowed."}), 400

    # This will be the filename used for processing and storage.
    # It may be changed if there's a collision with a default PDF.
    filename_for_processing = original_secure_filename
    
    # Check for name collision with files in DEFAULT_PDFS_FOLDER
    if hasattr(config, 'DEFAULT_PDFS_FOLDER') and config.DEFAULT_PDFS_FOLDER:
        default_pdf_path_if_exists = os.path.join(config.DEFAULT_PDFS_FOLDER, original_secure_filename)
        if os.path.exists(default_pdf_path_if_exists):
            name_part, ext_part = os.path.splitext(original_secure_filename)
            unique_suffix = uuid.uuid4().hex[:8] # Use a short UUID to make it unique
            filename_for_processing = f"{name_part}_{unique_suffix}{ext_part}"
            logger.info(
                f"Uploaded file '{original_secure_filename}' has a name conflict with a file in DEFAULT_PDFS_FOLDER. "
                f"It will be processed as '{filename_for_processing}' to ensure the uploaded version from UPLOAD_FOLDER is used."
            )
    else:
        logger.debug("config.DEFAULT_PDFS_FOLDER not defined or empty. Skipping collision check with default PDFs.")

    # Use filename_for_processing (which might be same as original_secure_filename or a new unique one)
    final_filename = filename_for_processing
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], final_filename)

    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)
        logger.info(f"File '{final_filename}' saved to {filepath}")

        text = ai_core.extract_text_from_file(filepath)
        if not text:
            logger.error(f"Could not extract text from '{final_filename}'.")
            # Attempt to remove the problematic file to avoid clutter
            try:
                if os.path.exists(filepath): os.remove(filepath)
                logger.info(f"Removed '{filepath}' due to text extraction failure.")
            except OSError as e_rm:
                logger.error(f"Error removing '{filepath}' after text extraction failure: {e_rm}")
            return jsonify({"error": f"Could not read text from '{final_filename}'. Ensure it's a valid file."}), 400

        ai_core.document_texts_cache[final_filename] = text
        logger.info(f"Text extracted and cached for {final_filename}.")

        documents_for_rag = ai_core.create_chunks_from_text(text, final_filename)
        if not documents_for_rag:
            logger.error(f"Could not create chunks for '{final_filename}'.")
            return jsonify({"error": f"Could not process '{final_filename}' into searchable chunks."}), 500

        if not ai_core.add_documents_to_vector_store(documents_for_rag):
            logger.error(f"Failed to add '{final_filename}' to vector store.")
            return jsonify({"error": f"File '{final_filename}' processed but failed to update knowledge base."}), 500
        
        vector_count = -1
        if ai_core.vector_store and hasattr(ai_core.vector_store, 'index'):
            vector_count = getattr(ai_core.vector_store.index, 'ntotal', 0)
        logger.info(f"Successfully processed '{final_filename}' for RAG. Vector count: {vector_count}")

        # --- AUTOMATIC KG GENERATION TRIGGER ---
        logger.info(f"UPLOAD_SUCCESS: '{final_filename}' processed for RAG. Attempting to trigger background KG generation.")
        # Start KG generation in a new thread so the upload request returns quickly.
        # Pass the 'final_filename' (which is unique if collision occurred)
        kg_thread = threading.Thread(target=_trigger_kg_generation_background, args=(final_filename,))
        kg_thread.daemon = True # Allows main program to exit even if threads are still running
        kg_thread.start()
        logger.info(f"UPLOAD_SUCCESS: Background KG generation thread started for '{final_filename}'.")
        # --- END AUTOMATIC KG GENERATION TRIGGER ---

        response_payload = {
            "message": f"File '{original_secure_filename}' uploaded and processed for RAG (as '{final_filename}'). KG generation started in background.",
            "filename": final_filename, # This is the name to use for future requests for this file
            "vector_count": vector_count
        }
        if final_filename != original_secure_filename:
            response_payload["original_filename_on_upload"] = original_secure_filename
            response_payload["note"] = "Filename was changed to avoid conflict with a default system file."


        return jsonify(response_payload), 200 # Return 200 OK immediately

    except Exception as e:
        # Use final_filename if defined, otherwise original_secure_filename for logging
        current_fn_for_log = final_filename if 'final_filename' in locals() else original_secure_filename
        logger.error(f"Error processing upload '{current_fn_for_log}': {e}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred during upload: {type(e).__name__}."}), 500

@app.route('/generate_kg_service', methods=['POST'])
def generate_kg_service_endpoint():
    """
    Endpoint to trigger KG generation for a specific file.
    This is intended to be called by other services or manually.
    """
    if not app_ai_ready:
        logger.error("KG generation request failed: Core AI components (LLM/Embeddings) not initialized.")
        return jsonify({"error": "Cannot generate KG: AI components not ready."}), 503
    
    data = request.get_json()
    if not data or not isinstance(data, dict):
        logger.warning("KG service request missing JSON body or invalid format.")
        return jsonify({"error": "Invalid request: JSON body with 'filename' required."}), 400

    filename = data.get('filename')
    if not filename or not isinstance(filename, str) or \
       '/' in filename or '\\' in filename or filename.startswith('.'):
        logger.warning(f"Invalid filename for KG service: {filename}")
        return jsonify({"error": "Missing or invalid 'filename'. Must be a simple filename."}), 400

    logger.info(f"Received service request to generate knowledge graph for '{filename}'")
    try:
        # Start KG generation in a new thread
        kg_thread = threading.Thread(target=_trigger_kg_generation_background, args=(filename,))
        kg_thread.daemon = True 
        kg_thread.start()
        logger.info(f"KG service: Background KG generation thread started for '{filename}'.")

        return jsonify({
            "message": f"Knowledge graph generation for '{filename}' initiated in background.",
            "filename": filename
        }), 202 # Return 202 Accepted, as processing is in background
    except Exception as e:
        logger.error(f"Unexpected error during KG service request for '{filename}': {e}", exc_info=True)
        return jsonify({"error": f"An unexpected server error occurred: {type(e).__name__}."}), 500

@app.route('/get_kg_data', methods=['GET'])
def get_kg_data_endpoint():
    """
    Endpoint to retrieve knowledge graph data for a specific file.
    This is intended to be called by other services.
    """
    if not app_ai_ready:
        logger.error("KG data retrieval failed: Core AI components (LLM/Embeddings) not initialized.")
        return jsonify({"error": "Cannot retrieve KG data: AI components not ready."}), 503
    
    filename = request.args.get('filename')
    if not filename or not isinstance(filename, str) or \
       '/' in filename or '\\' in filename or filename.startswith('.'):
        logger.warning(f"Invalid filename for KG data retrieval: {filename}")
        return jsonify({"error": "Missing or invalid 'filename'. Must be a simple filename."}), 400

    logger.info(f"Received request to retrieve knowledge graph for '{filename}'")
    try:
        kg_data = ai_core.load_knowledge_graph(filename)
        if kg_data:
            logger.info(f"Successfully retrieved KG for '{filename}'.")
            return jsonify({"filename": filename, "kg_data": kg_data}), 200
        else:
            logger.warning(f"No KG data found for '{filename}'.")
            return jsonify({"filename": filename, "kg_data": None, "message": "No knowledge graph found for this file."}), 404
    except Exception as e:
        logger.error(f"Unexpected error retrieving KG data for '{filename}': {e}", exc_info=True)
        return jsonify({"error": f"An unexpected server error occurred: {type(e).__name__}."}), 500

@app.route('/')
def index():
    logger.debug("Serving index.html")
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return Response(status=204) # Using 204 No Content is a common way to handle favicon if you don't have one

@app.route('/status', methods=['GET'])
def get_status():
    vector_store_count = -1
    if app_ai_ready and app_vector_store_ready and ai_core.vector_store and hasattr(ai_core.vector_store, 'index'):
        try:
            vector_store_count = ai_core.vector_store.index.ntotal
        except AttributeError: 
            vector_store_count = 0 
        except Exception:
             vector_store_count = -2 # Indicates an error trying to get the count
    elif app_vector_store_ready and not ai_core.vector_store: # Store was supposed to be ready but instance is None
        vector_store_count = 0

    status_data = {
        "status": "ok" if app_db_ready and app_ai_ready else "degraded",
        "database_initialized": app_db_ready,
        "ai_components_loaded": app_ai_ready,
        "vector_store_loaded": app_vector_store_ready,
        "vector_store_entries": vector_store_count,
        "doc_cache_loaded": app_doc_cache_loaded,
        "cached_docs_count": len(ai_core.document_texts_cache) if app_doc_cache_loaded and ai_core.document_texts_cache is not None else 0,
        "ollama_model": getattr(config, 'OLLAMA_MODEL', 'N/A'),
        "embedding_model": getattr(config, 'OLLAMA_EMBED_MODEL', 'N/A'),
        "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    }
    return jsonify(status_data)

@app.route('/documents', methods=['GET'])
def get_documents():
    default_files = []
    uploaded_files = []
    error_messages = []

    def _list_files(folder_path, folder_name_log):
        files_list = []
        if not folder_path or not os.path.exists(folder_path): # Added check for folder_path itself
            msg = f"Folder not found or not configured: {folder_path} (for {folder_name_log})"
            error_messages.append(msg)
            logger.warning(msg)
            return files_list
        try:
            files_list = sorted([
                f for f in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, f)) and
                utils.allowed_file(f) and # Relies on utils.allowed_file and config.ALLOWED_EXTENSIONS
                not f.startswith('~') # Ignore temporary files (e.g. Word temporary files)
            ])
        except OSError as e:
            msg = f"Could not read folder {folder_path}: {e}"
            error_messages.append(msg)
            logger.error(msg, exc_info=True)
        return files_list

    # Get folder paths from config
    default_pdfs_folder = getattr(config, 'DEFAULT_PDFS_FOLDER', None)
    upload_folder = getattr(config, 'UPLOAD_FOLDER', None) # This is also app.config['UPLOAD_FOLDER']
    kg_output_folder = getattr(config, 'KG_OUTPUT_FOLDER', None)
    kg_filename_suffix = getattr(config, 'KG_FILENAME_SUFFIX', '.kg.json')

    default_files = _list_files(default_pdfs_folder, "Default Files")
    uploaded_files = _list_files(upload_folder, "Uploaded Files")
    
    available_kgs = []
    if kg_output_folder and os.path.exists(kg_output_folder):
        try:
            available_kgs = sorted([
                f for f in os.listdir(kg_output_folder)
                if f.endswith(kg_filename_suffix) # Use configured suffix
            ])
        except OSError as e:
            msg = f"Could not read KG folder {kg_output_folder}: {e}"
            error_messages.append(msg)
            logger.error(msg, exc_info=True)
    elif not kg_output_folder:
        logger.debug("KG_OUTPUT_FOLDER not configured, so not listing available KGs.")

    response_data = {
        "default_files": default_files,
        "uploaded_files": uploaded_files,
        "available_kgs": available_kgs, 
        "errors": error_messages if error_messages else None
    }
    logger.debug(f"Returning document lists: {len(default_files)} default, {len(uploaded_files)} uploaded, {len(available_kgs)} KGs.")
    return jsonify(response_data)

@app.route('/generate_kg', methods=['POST'])
def generate_knowledge_graph_route(): 
    data = request.get_json()
    if not data:
        logger.warning("KG request missing JSON body.")
        return jsonify({"error": "Invalid request: JSON body required."}), 400

    filename = data.get('filename')
    if not filename or not isinstance(filename, str) or \
       '/' in filename or '\\' in filename or filename.startswith('.'): # Basic security for filename
        logger.warning(f"Invalid filename for KG: {filename}")
        return jsonify({"error": "Missing or invalid 'filename'. Must be a simple filename."}), 400

    logger.info(f"Received MANUAL request to generate knowledge graph for '{filename}'")
    try:
        # ai_core.generate_knowledge_graph_from_pdf is expected to find 'filename'
        # in one of the configured locations (e.g., UPLOAD_FOLDER or DEFAULT_PDFS_FOLDER)
        # If 'filename' was a result of renaming in upload_file, the client must provide that renamed name.
        kg_result_data = ai_core.generate_knowledge_graph_from_pdf(filename) 
        
        if not kg_result_data: 
            logger.error(f"MANUAL KG generation failed for '{filename}' (returned None).")
            return jsonify({"error": f"Failed to generate knowledge graph for '{filename}'. Check server logs."}), 500

        # Assuming ai_core.get_kg_filepath constructs the path to the saved KG file
        kg_output_file = ai_core.get_kg_filepath(filename) 

        nodes_count = len(kg_result_data.get('nodes',[])) if isinstance(kg_result_data, dict) else 0
        edges_count = len(kg_result_data.get('edges',[])) if isinstance(kg_result_data, dict) else 0
        
        logger.info(f"MANUAL KG generated for '{filename}': {nodes_count} nodes, {edges_count} edges.")
        return jsonify({
            "message": f"Knowledge graph generation MANUALLY initiated and likely completed for '{filename}'.",
            "nodes_count": nodes_count,
            "edges_count": edges_count,
            "output_file": kg_output_file 
        }), 200
    except Exception as e: 
        logger.error(f"Unexpected error during MANUAL KG generation for '{filename}': {e}", exc_info=True)
        return jsonify({"error": f"An unexpected server error occurred: {type(e).__name__}."}), 500

@app.route('/analyze', methods=['POST'])
def analyze_document():
    if not app_ai_ready or not ai_core.llm: 
        logger.error("Analysis failed: LLM not initialized.")
        return jsonify({"error": "Analysis unavailable: AI model not ready.", "thinking": None}), 503
    data = request.get_json()
    if not data: 
        logger.warning("Analysis request missing JSON body.")
        return jsonify({"error": "Invalid request: JSON body required.", "thinking": None}), 400
    filename = data.get('filename')
    analysis_type = data.get('analysis_type')
    logger.info(f"Analysis request: type='{analysis_type}', file='{filename}'")

    if not filename or not isinstance(filename, str) or '/' in filename or '\\' in filename:
        logger.warning(f"Invalid filename for analysis: {filename}")
        return jsonify({"error": "Missing or invalid 'filename'.", "thinking": None}), 400
    
    analysis_prompts_config = getattr(config, 'ANALYSIS_PROMPTS', {})
    if not isinstance(analysis_prompts_config, dict) or not analysis_prompts_config:
        logger.error("config.ANALYSIS_PROMPTS is not defined, not a dict, or empty. Analysis types unknown.")
        return jsonify({"error": "Server configuration error for analysis types.", "thinking": None}), 500
    allowed_types = list(analysis_prompts_config.keys())

    if not analysis_type or analysis_type not in allowed_types:
        logger.warning(f"Invalid analysis_type: {analysis_type}")
        return jsonify({"error": f"Invalid 'analysis_type'. Must be one of: {', '.join(allowed_types)}", "thinking": None}), 400
    
    try:
        # ai_core.generate_document_analysis is expected to find 'filename'
        # (e.g., using its text from document_texts_cache, which would use the final_filename if renamed on upload)
        analysis_content, thinking_content = ai_core.generate_document_analysis(filename, analysis_type)
        
        if analysis_content is None: 
            logger.error(f"Analysis failed: Could not retrieve or process '{filename}'.")
            return jsonify({"error": f"Could not retrieve or process '{filename}'.", "thinking": thinking_content or "File or text not found."}), 404
        elif isinstance(analysis_content, str) and analysis_content.startswith("Error:"): 
            logger.error(f"Analysis failed for '{filename}' ({analysis_type}): {analysis_content}")
            status_code = 503 if "AI model" in analysis_content else 500
            return jsonify({"error": analysis_content, "thinking": thinking_content}), status_code
            
        logger.info(f"Analysis successful for '{filename}' ({analysis_type}).")
        return jsonify({"content": analysis_content, "thinking": thinking_content})
    except Exception as e:
        logger.error(f"Unexpected error in analysis for '{filename}' ({analysis_type}): {e}", exc_info=True)
        return jsonify({"error": f"An unexpected server error occurred: {type(e).__name__}.", "thinking": None}), 500


@app.route('/chat', methods=['POST'])
def chat():
    try:
        if not app_db_ready:
            logger.error("Chat failed: Database not initialized.")
            return jsonify({"error": "Chat unavailable: Database connection failed.", "answer": "Database unavailable.", "thinking": None, "references": [], "session_id": None}), 503
        if not app_ai_ready or not ai_core.llm or not ai_core.embeddings: 
            logger.error("Chat failed: Core AI components not initialized.")
            return jsonify({"error": "Chat unavailable: AI components not ready.", "answer": "AI components not ready.", "thinking": None, "references": [], "session_id": None}), 503
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid request: JSON body required."}), 400
        
        query = data.get('query', "").strip()
        session_id = data.get('session_id')

        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400

        is_new_session = False
        if session_id:
            try:
                uuid.UUID(session_id, version=4) 
            except (ValueError, TypeError):
                logger.warning(f"Invalid session_id '{session_id}' received. Generating new one.")
                session_id = str(uuid.uuid4())
                is_new_session = True
        else:
            session_id = str(uuid.uuid4())
            is_new_session = True
        
        logger.info(f"Processing chat query (Session: {session_id}, New: {is_new_session}): '{query[:100]}...'")

        try: 
            database.save_message(session_id, 'user', query, None, None)
        except Exception as e:
            logger.error(f"Database error saving user message for session {session_id}: {e}", exc_info=True)

        bot_answer_content = "Sorry, an issue occurred while processing your request."
        thinking_content = "" # Initialize empty for accumulation
        references_list = []
        
        # Fetch chat history for the ReAct loop
        messages_from_db = []
        try:
            messages_from_db = database.get_messages_by_session(session_id)
            # Format history for LLM: only 'user' and 'bot' roles, as simple text
            chat_history_for_llm = []
            for msg in messages_from_db:
                if msg['role'] in ['user', 'bot']:
                    chat_history_for_llm.append(f"{msg['role'].capitalize()}: {msg['content']}")
            chat_history_for_llm = "\\n".join(chat_history_for_llm)
        except Exception as e:
            logger.error(f"Database error fetching chat history for session {session_id}: {e}", exc_info=True)
            chat_history_for_llm = "" # Proceed without history if error

        tool_output = "" # Initialize tool output for the first turn

        MAX_REACT_ITERATIONS = 5 # Prevent infinite loops
        current_iteration = 0

        while current_iteration < MAX_REACT_ITERATIONS:
            current_iteration += 1
            logger.info(f"ReAct Iteration {current_iteration} for session {session_id}.")
            
            # Call LLM with current query, history, and tool output
            llm_response, current_thought = ai_core.synthesize_chat_response(
                query=query, 
                chat_history=chat_history_for_llm, 
                tool_output=tool_output
            )
            
            thinking_content += f"Iteration {current_iteration} Thought: {current_thought}\\n"
            
            if llm_response.startswith("<ACTION>"):
                action_text = llm_response.replace("<ACTION>", "").replace("</ACTION>", "").strip()
                logger.info(f"LLM proposed Action: {action_text}")
                
                # Parse the action
                action_match = re.match(r"(\w+)\((.*)\)", action_text)
                if action_match:
                    tool_name = action_match.group(1)
                    tool_args_str = action_match.group(2)
                    
                    tool_args = {}
                    try:
                        # Basic parsing for key=value, handle potential JSON in values
                        # This is a simplified parser, for more complex args, ast.literal_eval or a proper parser might be needed
                        arg_pairs = re.findall(r'(\w+)=(?:\"(.*?)\"|\'(.*?)\'|([^\,)]+))', tool_args_str)
                        for key, val1, val2, val3 in arg_pairs:
                            # Prioritize double-quoted, then single-quoted, then unquoted
                            tool_args[key.strip()] = (val1 or val2 or val3).strip()
                            # Attempt to parse as JSON if it looks like JSON
                            if tool_args[key.strip()].startswith('{') and tool_args[key.strip()].endswith('}'):
                                try:
                                    tool_args[key.strip()] = json.loads(tool_args[key.strip()])
                                except json.JSONDecodeError:
                                    pass # Not JSON, keep as string
                            elif tool_args[key.strip()].isdigit():
                                tool_args[key.strip()] = int(tool_args[key.strip()])

                    except Exception as e:
                        logger.error(f"Error parsing tool arguments '{tool_args_str}': {e}", exc_info=True)
                        tool_output = f"Error: Failed to parse arguments for {tool_name}. Argument parsing error: {e}"
                        continue # Continue to next iteration for LLM to re-evaluate

                    tool_output = "" # Reset tool output for next observation

                    if tool_name == "RAG_search":
                        search_query = tool_args.get('query')
                        if search_query:
                            try:
                                # ai_core.perform_rag_search already returns relevant docs with KG data if enabled
                                rag_context_docs, rag_context_text, rag_docs_map = ai_core.perform_rag_search(search_query)
                                
                                # Format relevantDocs for LLM consumption as Observation
                                formatted_rag_output = []
                                for i, doc in enumerate(rag_context_docs):
                                    doc_content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                                    doc_source = doc.metadata.get('source', 'Unknown Source')
                                    kg_info = ""
                                    if doc.metadata.get('kg_data'):
                                        kg_info = ai_core.format_kg_info_for_llm(doc.metadata['kg_data'], query) # Query might be used for relevancy in KG formatting
                                        if kg_info:
                                            kg_info = f"  KG Insights for '{doc_source}':\\n{kg_info}"

                                    formatted_rag_output.append(f"Document {i+1} (Source: {doc_source}):\\n{doc_content}\\n{kg_info}")
                                    
                                    # Also collect references for the final response
                                    references_list.append({"source": doc_source, "content": doc.page_content})

                                tool_output = f"RAG_search Result:\\n{'' if not formatted_rag_output else '\\n'.join(formatted_rag_output)}"
                                logger.info(f"RAG_search executed. {len(rag_context_docs)} docs found.")
                            except Exception as e:
                                logger.error(f"Error during RAG_search for query '{search_query}': {e}", exc_info=True)
                                tool_output = f"Error: RAG_search failed: {e}"
                        else:
                            tool_output = "Error: RAG_search requires a 'query' argument."
                    
                    elif tool_name == "KG_query":
                        kg_query_text = tool_args.get('query')
                        kg_document_name = tool_args.get('document_name')
                        if kg_query_text or kg_document_name:
                            try:
                                kg_data = None
                                if kg_document_name:
                                    kg_data = ai_core.load_knowledge_graph(kg_document_name)
                                # If no document_name or no KG found for it, could add a broader KG search here if available
                                
                                if kg_data:
                                    formatted_kg_output = ai_core.format_kg_info_for_llm(kg_data, kg_query_text or query)
                                    tool_output = f"KG_query Result for '{kg_document_name or 'general'}':\\n{formatted_kg_output}"
                                    logger.info(f"KG_query executed. KG data found for '{kg_document_name or 'general'}'.")
                                else:
                                    tool_output = f"KG_query Result: No knowledge graph data found for '{kg_document_name or 'the query'}'. "
                                    logger.info(f"KG_query executed. No KG data found for '{kg_document_name or 'the query'}'.")
                            except Exception as e:
                                logger.error(f"Error during KG_query for '{kg_document_name or kg_query_text}': {e}", exc_info=True)
                                tool_output = f"Error: KG_query failed: {e}"
                        else:
                            tool_output = "Error: KG_query requires 'query' or 'document_name' argument."
                    
                    elif tool_name == "Web_search":
                        # Placeholder for future web search integration
                        search_query = tool_args.get('query')
                        tool_output = f"Observation: Web_search is not yet implemented. Cannot search for '{search_query}'."
                        logger.warning(f"Web_search tool called but not implemented: {search_query}")

                    else:
                        tool_output = f"Error: Unknown tool '{tool_name}'."
                        logger.warning(f"Unknown tool requested by LLM: {tool_name}")
                else:
                    tool_output = "Error: Failed to parse action format."
                    logger.warning(f"LLM proposed action in unparseable format: {action_text[:200]}...")
                
                # Continue the loop for the next ReAct iteration
                continue 

            else: # LLM did not propose an action, means it's a 'Speak' or an error
                bot_answer_content = llm_response # This should be the final answer if not an action
                logger.info(f"LLM provided final answer. Iterations: {current_iteration}")
                break # Exit loop if not an action

        # If loop finishes without a 'Speak' (e.g., max iterations reached or error)
        if current_iteration >= MAX_REACT_ITERATIONS and not llm_response.startswith("<ACTION>"): # Ensure it's not still trying to act
            bot_answer_content = bot_answer_content or "Sorry, I reached the maximum number of reasoning steps and could not generate a complete answer. Please try rephrasing your query."
            logger.warning(f"Max ReAct iterations reached for session {session_id}.")

        if isinstance(bot_answer_content, str) and bot_answer_content.startswith("Error:"):
            logger.error(f"LLM synthesis failed for session {session_id}: {bot_answer_content}")
        
        # Save bot's final answer to DB
        try:
            # Deduplicate references for saving
            unique_references = []
            seen_sources = set()
            for ref in references_list:
                if ref['source'] not in seen_sources:
                    unique_references.append(ref)
                    seen_sources.add(ref['source'])
            
            database.save_message(session_id, 'bot', bot_answer_content, json.dumps(unique_references) if unique_references else None, thinking_content)
        except Exception as e:
            logger.error(f"Database error saving bot response for session {session_id}: {e}", exc_info=True)

        return jsonify({
            "answer": bot_answer_content,
            "session_id": session_id,
            "references": unique_references,
            "thinking": thinking_content
        }), 200

    except Exception as e: 
        logger.error(f"Unexpected error in chat processing for session {session_id}: {e}", exc_info=True)
        try:
            error_msg_for_db = f"Sorry, an unexpected server error occurred: {type(e).__name__}."
            if session_id: 
                database.save_message(session_id, 'bot', error_msg_for_db, None, f"Internal Error: {str(e)}")
        except Exception as db_err:
            logger.error(f"Failed to save error message to DB for session {session_id} after chat error: {db_err}")
        
        return jsonify({
            "error": "An unexpected server error occurred.",
            "answer": "Sorry, I encountered an unexpected problem. Please try again.", 
            "session_id": session_id or str(uuid.uuid4()), 
            "thinking": f"Error: {type(e).__name__}", 
            "references": []
        }), 500

@app.route('/history', methods=['GET'])
def get_history():
    if not app_db_ready:
        logger.error("History retrieval failed: Database not initialized.")
        return jsonify({"error": "History unavailable: Database connection failed."}), 503
    
    session_id = request.args.get('session_id')
    if not session_id:
        logger.warning("History request missing 'session_id' parameter.")
        return jsonify({"error": "Missing 'session_id' parameter"}), 400
    
    try: 
        uuid.UUID(session_id, version=4) # Validate format
    except (ValueError, TypeError):
        logger.warning(f"Invalid session_id format for history: {session_id}")
        return jsonify({"error": "Invalid session_id format."}), 400
        
    try:
        messages_from_db = database.get_messages_by_session(session_id) # Expect list of dict-like or Row objects
        if messages_from_db is None: 
            logger.error(f"History retrieval for session {session_id} failed at DB level (returned None).")
            return jsonify({"error": "Could not retrieve history due to a database error."}), 500
        
        processed_messages = []
        for msg_data in messages_from_db: 
            # Ensure msg is a mutable dictionary for modification
            msg = dict(msg_data) # Works if msg_data is a RowProxy, dict, or similar mapping

            if msg.get('role') == 'bot':
                references_json = msg.get('references')
                if references_json and isinstance(references_json, str):
                    try:
                        msg['references'] = json.loads(references_json)
                    except json.JSONDecodeError:
                        logger.warning(f"Could not decode references JSON for message ID {msg.get('id')} in session {session_id}: '{str(references_json)[:100]}...'")
                        msg['references'] = [] 
                elif not references_json: 
                     msg['references'] = []
            processed_messages.append(msg)

        logger.info(f"Retrieved {len(processed_messages)} messages for session {session_id}.")
        return jsonify(processed_messages)
    except Exception as e: 
        logger.error(f"Unexpected error retrieving history for session {session_id}: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected server error occurred: {type(e).__name__}."}), 500

if __name__ == '__main__':
    if not hasattr(app, 'initialized') or not app.initialized:
        initialize_app() 
    
    try:
        # Use getattr for safer access to config attributes with defaults
        port_default = getattr(config, 'FLASK_PORT_DEFAULT', 5000)
        port = int(os.getenv('FLASK_RUN_PORT', port_default))
        if not (1024 <= port <= 65535): 
            logger.warning(f"Configured port {port} is invalid. Defaulting to {port_default}.")
            port = port_default
    except ValueError:
        port_default_for_error = getattr(config, 'FLASK_PORT_DEFAULT', 5000)
        port = port_default_for_error
        logger.warning(f"Invalid FLASK_RUN_PORT environment variable. Defaulting to {port_default_for_error}.")
    
    host_default = getattr(config, 'FLASK_HOST_DEFAULT', '127.0.0.1') # Changed from 'localhost' to '127.0.0.1' for clarity
    host = os.getenv('FLASK_RUN_HOST', host_default)
    
    logger.info(f"--- Starting Waitress WSGI Server for Flask app '{app.name}' ---")
    logger.info(f"Serving on http://{host}:{port}")
    logger.info(f"Ollama Base URL: {getattr(config, 'OLLAMA_BASE_URL', 'http://172.180.9.187:11435/')}")
    logger.info(f"LLM Model: {getattr(config, 'OLLAMA_MODEL', 'deepseek-r1:14b')}, Embedding Model: {getattr(config, 'OLLAMA_EMBED_MODEL', 'mxbai-embed-large')}")
    
    db_status = 'Ready' if app_db_ready else 'Failed'
    ai_status = 'Ready' if app_ai_ready else 'Failed/Degraded'
    
    index_status_val = 'Not Loaded/Unavailable'
    if app_vector_store_ready:
        if ai_core.vector_store and hasattr(ai_core.vector_store, 'index') and hasattr(ai_core.vector_store.index, 'ntotal'):
            index_status_val = f"Ready ({ai_core.vector_store.index.ntotal} entries)"
        elif ai_core.vector_store: # Vector store object exists but index details might be missing
            index_status_val = "Ready (index status unknown or empty)"
        else: 
            index_status_val = "Ready (instance missing despite flag)" # Should ideally not happen
            
    cache_status_val = "Empty/Not Loaded"
    if app_doc_cache_loaded and ai_core.document_texts_cache is not None: 
        cache_status_val = f"{len(ai_core.document_texts_cache)} docs"
        
    logger.info(f"Initial Status: DB={db_status} | AI={ai_status} | VectorIndex={index_status_val} | DocCache={cache_status_val}")
    
    waitress_threads = getattr(config, 'WAITRESS_THREADS', 8)
    serve(app, host=host, port=port, threads=waitress_threads)

# --- END OF FILE app.py ---