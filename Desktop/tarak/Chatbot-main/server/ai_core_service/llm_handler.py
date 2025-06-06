# FusedChatbot/server/ai_core_service/llm_handler.py
import os
import logging
from dotenv import load_dotenv

# Attempt to load SDKs, with placeholders if not installed
try:
    import google.generativeai as genai
except ImportError:
    genai = None
    logging.warning("google.generativeai SDK not found. Gemini functionality will be unavailable.")

try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None
    logging.warning("langchain_ollama not found. Ollama functionality will be unavailable.")

try:
    from groq import Groq
except ImportError:
    Groq = None
    logging.warning("groq SDK not found. Groq functionality will be unavailable.")

try:
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage # For Ollama chat messages
except ImportError:
    HumanMessage, SystemMessage, AIMessage = None, None, None
    logging.warning("langchain_core.messages not found. Ollama functionality might be affected.")

# Need to import config here if ANALYSIS_MAX_CONTEXT_LENGTH is used from it
try:
    from . import config as service_config # Use an alias to avoid confusion if 'config' is a var name
except ImportError: # Fallback for direct script run or if structuring changes
    import config as service_config


# Load environment variables from .env file in the ai_core_service directory
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path)

logger = logging.getLogger(__name__)

# --- Analysis Prompts (Adapted from Chatbot A's config) ---
_ANALYSIS_THINKING_PREFIX_STR = """**STEP 1: THINKING PROCESS (Recommended):**
*   Before generating the analysis, briefly outline your plan in `<thinking>` tags. Example: `<thinking>Analyzing for FAQs. Will scan for key questions and answers presented in the text.</thinking>`
*   If you include thinking, place the final analysis *after* the `</thinking>` tag.

**STEP 2: ANALYSIS OUTPUT:**
*   Generate the requested analysis based **strictly** on the text provided below.
*   Follow the specific OUTPUT FORMAT instructions carefully.

--- START DOCUMENT TEXT ---
{doc_text_for_llm}
--- END DOCUMENT TEXT ---
"""

# If PromptTemplate is available, use it. Otherwise, store as strings.
try:
    from langchain.prompts import PromptTemplate # Or langchain_core.prompts
    ANALYSIS_PROMPTS = {
        "faq": PromptTemplate(
            input_variables=["doc_text_for_llm"],
            template=_ANALYSIS_THINKING_PREFIX_STR + """
**TASK:** Generate 5-7 Frequently Asked Questions (FAQs) with concise answers based ONLY on the text.
**OUTPUT FORMAT (Strict):**
*   Start directly with the first FAQ (after thinking, if used). Do **NOT** include preamble.
*   Format each FAQ as:
    Q: [Question derived ONLY from the text]
    A: [Answer derived ONLY from the text, concise]
*   If the text doesn't support an answer, don't invent one. Use Markdown for formatting if appropriate.
**BEGIN OUTPUT (Start with 'Q:' or `<thinking>`):**
"""
        ),
        "topics": PromptTemplate(
            input_variables=["doc_text_for_llm"],
            template=_ANALYSIS_THINKING_PREFIX_STR + """
**TASK:** Identify the 5-8 most important topics discussed. Provide a 1-2 sentence explanation per topic based ONLY on the text.
**OUTPUT FORMAT (Strict):**
*   Start directly with the first topic (after thinking, if used). Do **NOT** include preamble.
*   Format as a Markdown bulleted list:
    *   **Topic Name:** Brief explanation derived ONLY from the text content (1-2 sentences max).
**BEGIN OUTPUT (Start with '*   **' or `<thinking>`):**
"""
        ),
        "mindmap": PromptTemplate(
            input_variables=["doc_text_for_llm"],
            template=_ANALYSIS_THINKING_PREFIX_STR + """
**TASK:** Generate a mind map outline in Markdown list format representing key concepts and hierarchy ONLY from the text.
**OUTPUT FORMAT (Strict):**
*   Start directly with the main topic as the top-level item (using '-') (after thinking, if used). Do **NOT** include preamble.
*   Use nested Markdown lists ('-' or '*') with indentation (2 or 4 spaces) for hierarchy.
*   Focus **strictly** on concepts and relationships mentioned in the text. Be concise.
**BEGIN OUTPUT (Start with e.g., '- Main Topic' or `<thinking>`):**
"""
        )
    }
    logger.info("ANALYSIS_PROMPTS initialized using PromptTemplate objects.")
except ImportError:
    PromptTemplate = None
    # Fallback to string templates if PromptTemplate is not available
    _FAQ_TEMPLATE_STR = _ANALYSIS_THINKING_PREFIX_STR + """
**TASK:** Generate 5-7 Frequently Asked Questions (FAQs) with concise answers based ONLY on the text.
**OUTPUT FORMAT (Strict):**
*   Start directly with the first FAQ (after thinking, if used). Do **NOT** include preamble.
*   Format each FAQ as:
    Q: [Question derived ONLY from the text]
    A: [Answer derived ONLY from the text, concise]
*   If the text doesn't support an answer, don't invent one. Use Markdown for formatting if appropriate.
**BEGIN OUTPUT (Start with 'Q:' or `<thinking>`):**
"""
    _TOPICS_TEMPLATE_STR = _ANALYSIS_THINKING_PREFIX_STR + """
**TASK:** Identify the 5-8 most important topics discussed. Provide a 1-2 sentence explanation per topic based ONLY on the text.
**OUTPUT FORMAT (Strict):**
*   Start directly with the first topic (after thinking, if used). Do **NOT** include preamble.
*   Format as a Markdown bulleted list:
    *   **Topic Name:** Brief explanation derived ONLY from the text content (1-2 sentences max).
**BEGIN OUTPUT (Start with '*   **' or `<thinking>`):**
"""
    _MINDMAP_TEMPLATE_STR = _ANALYSIS_THINKING_PREFIX_STR + """
**TASK:** Generate a mind map outline in Markdown list format representing key concepts and hierarchy ONLY from the text.
**OUTPUT FORMAT (Strict):**
*   Start directly with the main topic as the top-level item (using '-') (after thinking, if used). Do **NOT** include preamble.
*   Use nested Markdown lists ('-' or '*') with indentation (2 or 4 spaces) for hierarchy.
*   Focus **strictly** on concepts and relationships mentioned in the text. Be concise.
**BEGIN OUTPUT (Start with e.g., '- Main Topic' or `<thinking>`):**
"""
    ANALYSIS_PROMPTS = {
        "faq": _FAQ_TEMPLATE_STR,
        "topics": _TOPICS_TEMPLATE_STR,
        "mindmap": _MINDMAP_TEMPLATE_STR
    }
    logger.warning("langchain.prompts.PromptTemplate not found. ANALYSIS_PROMPTS stored as strings. Manual formatting will be needed.")

# --- Gemini Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
gemini_sdk_configured = False

if genai and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("Google Generative AI SDK configured successfully.")
        gemini_sdk_configured = True
    except Exception as e:
        logger.error(f"Failed to configure Google Generative AI SDK: {e}", exc_info=True)
elif not genai:
    logger.debug("Gemini SDK (google.generativeai) not installed, Gemini features disabled.")
else:
     logger.warning("GEMINI_API_KEY not found in environment variables. Gemini functionality will be unavailable.")


DEFAULT_GEMINI_GENERATION_CONFIG = {"temperature": 0.7, "max_output_tokens": 4096}
DEFAULT_GEMINI_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
]

# --- Ollama Configuration ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
ollama_available = bool(ChatOllama and HumanMessage and SystemMessage and AIMessage)
if not ollama_available:
    logger.debug("Langchain Ollama or core messages not fully available, Ollama features might be limited or disabled.")

# --- Groq Configuration ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_GROQ_LLAMA3_MODEL = os.getenv("GROQ_LLAMA3_MODEL", "llama3-8b-8192")
groq_client = None
if Groq and GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq client configured successfully.")
    except Exception as e:
        logger.error(f"Failed to configure Groq client: {e}", exc_info=True)
        groq_client = None
elif not Groq:
    logger.debug("Groq SDK (groq) not installed, Groq features disabled.")
else:
    logger.warning("GROQ_API_KEY not found in environment variables. Groq functionality will be unavailable.")


# --- START: CoT (Chain-of-Thought) Modifications ---
_SYNTHESIS_PROMPT_TEMPLATE_STR = """You are a faculty member for engineering students with in-depth knowledge in all engineering subjects and am Expert for an academic audience, ranging from undergraduates to PhD scholars. Your goal is to answer the user's query based on the provided context document chunks, augmented with your general knowledge when necessary. You have to Provide detailed, technical, and well-structured responses suitable for this audience. Use precise terminology, include relevant concepts, algorithms, and applications, and organize your response with sections or bullet points where appropriate.

**TASK:** Respond to the user's query using the provided context and your general knowledge.

**USER QUERY:**
"{query}"

**PROVIDED CONTEXT:**
--- START CONTEXT ---
{context}
--- END CONTEXT ---

**INSTRUCTIONS:**

**STEP 1: THINKING PROCESS (MANDATORY):**
*   **CRITICAL:** Before writing the final answer, first articulate your step-by-step reasoning process for how you will arrive at the answer. Explain how you will use the context and potentially supplement it with general knowledge.
*   Use a step-by-step Chain of Thought (CoT) approach to arrive at a logical and accurate answer, and include your reasoning in a <thinking> tag. Enclose this entire reasoning process *exclusively* within `<thinking>` and `</thinking>` tags.
*   Example: `<thinking>The user asks about X. Context [1] defines X. Context [3] gives an example Z. Context [2] seems less relevant. The context doesn't cover aspect Y, so I will synthesize information from [1] and [3] and then add general knowledge about Y, clearly indicating it's external information.</thinking>`
*   **DO NOT** put any text before `<thinking>` or after `</thinking>` except for the final answer.

**STEP 2: FINAL ANSWER (After the `</thinking>` tag):**
*   Provide a comprehensive and helpful answer to the user query.
*   **Prioritize Context:** Base your answer **primarily** on information within the `PROVIDED CONTEXT`.
*   **Cite Sources:** When using information *directly* from a context chunk, **you MUST cite** its number like [1], [2], [1][3]. Cite all relevant sources for each piece of information derived from the context.
*   **Insufficient Context:** If the context does not contain information needed for a full answer, explicitly state what is missing (e.g., "The provided documents don't detail the specific algorithm used...").
*   **Integrate General Knowledge:** *Seamlessly integrate* your general knowledge to fill gaps, provide background, or offer broader explanations **after** utilizing the context. Clearly signal when you are using general knowledge (e.g., "Generally speaking...", "From external knowledge...", "While the documents focus on X, it's also important to know Y...").
*   **Be a Tutor:** Explain concepts clearly. Be helpful, accurate, and conversational. Use Markdown formatting (lists, bolding, code blocks) for readability.
*   **Accuracy:** Do not invent information not present in the context or verifiable general knowledge. If unsure, state that.

**BEGIN RESPONSE (Start *immediately* with the `<thinking>` tag):**
<thinking>"""

def _parse_thinking_and_answer(full_llm_response: str) -> tuple[str, str | None]:
    thinking_content = None
    answer = full_llm_response
    think_start_tag = "<thinking>"
    think_end_tag = "</thinking>"
    start_index = full_llm_response.find(think_start_tag)
    if start_index != -1:
        end_index = full_llm_response.find(think_end_tag, start_index + len(think_start_tag))
        if end_index != -1:
            thinking_content = full_llm_response[start_index + len(think_start_tag):end_index].strip()
            answer = full_llm_response[end_index + len(think_end_tag):].strip()
            logger.debug(f"Parsed thinking content (length: {len(thinking_content) if thinking_content else 0}). Answer starts with: '{answer[:50]}...'")
        else:
            logger.warning("Found '<thinking>' tag but no matching '</thinking>' tag in LLM response.")
    if not answer.strip() and thinking_content: # Only thinking, no answer after
        logger.warning("Parsed answer is empty after removing thinking block. LLM might have only outputted thinking content.")
        answer = "[AI response primarily contained reasoning. See thinking process.]"
    elif not answer.strip() and not thinking_content and full_llm_response.strip(): # Parsing failed to extract anything meaningful
        logger.warning("LLM response parsing resulted in empty answer and no thinking, but original response was not empty.")
        answer = full_llm_response # Fallback to original if parsing yields nothing.
    elif not full_llm_response.strip(): # LLM gave a totally empty response
        logger.warning("LLM response was empty, resulting in empty answer and no thinking.")
        answer = "[AI provided an empty response.]"
    return answer, thinking_content
# --- END: CoT Modifications ---


# --- Sub-Query Generation ---
_SUB_QUERY_TEMPLATE_STR = """You are an AI assistant skilled at decomposing user questions into effective search queries for a vector database.
Given the user's query, generate {num_queries} distinct search queries targeting different specific aspects, keywords, or concepts within the original query.
Focus on creating queries that are likely to retrieve relevant text chunks individually.
Output ONLY the generated search queries, each on a new line. Do not include numbering, labels, explanations, or any other text.

User Query: "{query}"

Generated Search Queries:"""

def generate_sub_queries_via_llm(original_query: str,
                                 llm_provider: str,
                                 llm_model_name: str = None,
                                 num_sub_queries: int = 3) -> list[str]:
    if num_sub_queries <= 0: return []
    prompt_for_sub_queries = _SUB_QUERY_TEMPLATE_STR.format(query=original_query, num_queries=num_sub_queries)
    logger.info(f"Generating {num_sub_queries} sub-queries for: '{original_query[:100]}...' using {llm_provider} (model: {llm_model_name or 'provider_default'})")
    logger.debug(f"Sub-query generation prompt: {prompt_for_sub_queries}")
    raw_llm_response_text = ""
    try:
        if llm_provider == "gemini":
            if not gemini_sdk_configured or not genai: raise ConnectionError("Gemini not configured for sub-query generation.")
            model = genai.GenerativeModel(llm_model_name or GEMINI_MODEL_NAME)
            response = model.generate_content(prompt_for_sub_queries)
            raw_llm_response_text = response.text if hasattr(response, 'text') else ""
        elif llm_provider == "groq_llama3":
            if not groq_client: raise ConnectionError("Groq client not configured for sub-query generation.")
            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt_for_sub_queries}],
                model=llm_model_name or DEFAULT_GROQ_LLAMA3_MODEL)
            raw_llm_response_text = chat_completion.choices[0].message.content
        elif llm_provider == "ollama":
            if not ollama_available: raise ConnectionError("Ollama not available for sub-query generation.")
            ollama_llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=llm_model_name or DEFAULT_OLLAMA_MODEL)
            response = ollama_llm.invoke([HumanMessage(content=prompt_for_sub_queries)])
            raw_llm_response_text = response.content if hasattr(response, 'content') else ""
        else:
            logger.error(f"Unsupported llm_provider '{llm_provider}' for sub-query generation."); return []
        logger.debug(f"Raw LLM response for sub-queries: {raw_llm_response_text}")
        sub_queries = [q.strip() for q in raw_llm_response_text.strip().split('\n') if q.strip()]
        if sub_queries: logger.info(f"Successfully generated {len(sub_queries)} sub-queries."); return sub_queries[:num_sub_queries]
        else: logger.warning("LLM did not generate any valid sub-queries."); return []
    except Exception as e:
        logger.error(f"Error generating sub-queries with {llm_provider}: {e}", exc_info=True); return []


# --- Document Analysis Function ---
def perform_document_analysis(document_text: str,
                              analysis_type: str,
                              llm_provider: str,
                              llm_model_name: str = None) -> tuple[str | None, str | None]: # (analysis_result, thinking_content)
    """
    Performs a specific analysis (FAQ, topics, mindmap) on the given document text
    using the specified LLM provider.
    """
    logger.info(f"Performing document analysis: type='{analysis_type}' using {llm_provider} (model: {llm_model_name or 'provider_default'})")

    if not document_text.strip():
        logger.warning("Document text for analysis is empty. Cannot perform analysis.")
        return "Error: Document content is empty, cannot perform analysis.", None

    # --- Step 1: Prepare Text for LLM (Truncation) ---
    # Use ANALYSIS_MAX_CONTEXT_LENGTH from our ai_core_service config
    max_len = service_config.ANALYSIS_MAX_CONTEXT_LENGTH
    doc_text_for_llm = document_text
    original_length = len(document_text)

    if original_length > max_len:
        logger.warning(f"Document text too long ({original_length} chars), truncating to {max_len} for '{analysis_type}' analysis.")
        doc_text_for_llm = document_text[:max_len]
        doc_text_for_llm += "\n\n... [CONTENT TRUNCATED DUE TO LENGTH LIMIT FOR ANALYSIS]"
    else:
        logger.debug(f"Using full document text ({original_length} chars) for analysis '{analysis_type}'.")

    # --- Step 2: Get Analysis Prompt ---
    prompt_obj = ANALYSIS_PROMPTS.get(analysis_type)
    if not prompt_obj:
        logger.error(f"Invalid analysis type '{analysis_type}' or prompt not found in ANALYSIS_PROMPTS.")
        return f"Error: Invalid analysis type '{analysis_type}'.", None

    final_prompt_str = ""
    try:
        if PromptTemplate and isinstance(prompt_obj, PromptTemplate): # Check if PromptTemplate was imported and used
            final_prompt_str = prompt_obj.format(doc_text_for_llm=doc_text_for_llm)
        elif isinstance(prompt_obj, str): # Fallback if prompts are just strings
            final_prompt_str = prompt_obj.format(doc_text_for_llm=doc_text_for_llm) # Assumes .format() works
        else:
            logger.error(f"Prompt for analysis type '{analysis_type}' is not a PromptTemplate object or a format-able string.")
            return "Error: Internal prompt configuration error for analysis.", None
        logger.debug(f"Analysis prompt for '{analysis_type}' (start): {final_prompt_str[:300]}...")
    except KeyError as e:
        logger.error(f"Error formatting ANALYSIS_PROMPTS['{analysis_type}']: Missing key {e}. Check prompt definition.")
        return f"Error: Internal prompt configuration issue for {analysis_type}.", None
    except Exception as e:
        logger.error(f"Error creating analysis prompt for {analysis_type}: {e}", exc_info=True)
        return f"Error: Could not prepare request for {analysis_type} analysis.", None

    # --- Step 3: Call LLM and Parse Response ---
    raw_llm_response_text = ""
    try:
        if llm_provider == "gemini":
            if not gemini_sdk_configured or not genai:
                raise ConnectionError("Gemini not configured for document analysis.")
            gemini_model_to_use = llm_model_name or GEMINI_MODEL_NAME
            model = genai.GenerativeModel(gemini_model_to_use)
            response = model.generate_content(final_prompt_str) # Direct generation
            raw_llm_response_text = response.text if hasattr(response, 'text') else ""
        
        elif llm_provider == "groq_llama3":
            if not groq_client:
                raise ConnectionError("Groq client not configured for document analysis.")
            groq_model_to_use = llm_model_name or DEFAULT_GROQ_LLAMA3_MODEL
            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": final_prompt_str}], # Treat entire analysis prompt as user input
                model=groq_model_to_use)
            raw_llm_response_text = chat_completion.choices[0].message.content
            
        elif llm_provider == "ollama":
            if not ollama_available or not ChatOllama or not HumanMessage: # Ensure HumanMessage is available
                raise ConnectionError("Ollama or Langchain HumanMessage not available for document analysis.")
            ollama_model_to_use = llm_model_name or DEFAULT_OLLAMA_MODEL
            ollama_llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=ollama_model_to_use)
            response = ollama_llm.invoke([HumanMessage(content=final_prompt_str)]) # Send as HumanMessage
            raw_llm_response_text = response.content if hasattr(response, 'content') else ""
            
        else:
            logger.error(f"Unsupported llm_provider '{llm_provider}' for document analysis.")
            return f"Error: Unsupported LLM provider '{llm_provider}' for analysis.", None

        logger.debug(f"Raw LLM response for analysis '{analysis_type}': {raw_llm_response_text[:300]}...")
        
        # Use the same parsing logic as for CoT synthesis
        analysis_result, thinking_content = _parse_thinking_and_answer(raw_llm_response_text)

        if analysis_result:
            logger.info(f"Successfully performed document analysis: '{analysis_type}'.")
            return analysis_result, thinking_content
        else:
            logger.warning(f"Document analysis '{analysis_type}' resulted in empty content after parsing, though LLM call succeeded.")
            # Return thinking content if it exists, otherwise a generic message
            return thinking_content or "[Analysis resulted in empty content]", thinking_content

    except Exception as e:
        logger.error(f"Error during document analysis '{analysis_type}' with {llm_provider}: {e}", exc_info=True)
        return f"Error performing document analysis '{analysis_type}': {e}", None


# --- Main Response Generation Functions (MODIFIED FOR CoT) ---

def get_gemini_response(query: str, context_text: str, system_prompt_text: str = None, # Note: query and context_text are now primary for the formatted prompt
                        generation_config=None, safety_settings=None, model_name=None,
                        chat_history=None # chat_history is now less directly used if prompt is monolithic
                        ) -> tuple[str, str | None]: # Returns (answer, thinking_content)
    if not gemini_sdk_configured: raise ConnectionError("Gemini not configured.")
    if not genai: raise ImportError("google.generativeai SDK is not available.")

    effective_model_name = model_name or GEMINI_MODEL_NAME
    
    final_llm_prompt = _SYNTHESIS_PROMPT_TEMPLATE_STR.format(query=query, context=context_text)

    model_options = {
        "model_name": effective_model_name,
        "generation_config": generation_config or DEFAULT_GEMINI_GENERATION_CONFIG,
        "safety_settings": safety_settings or DEFAULT_GEMINI_SAFETY_SETTINGS,
    }
    if system_prompt_text:
        logger.debug(f"Separate system_prompt_text provided for Gemini: '{system_prompt_text}'. It's not directly used if _SYNTHESIS_PROMPT_TEMPLATE_STR is the main prompt.")


    model = genai.GenerativeModel(**model_options)
    logger.info(f"Calling Gemini for synthesis. Model: {effective_model_name}.")
    logger.debug(f"Full prompt for Gemini (start): {final_llm_prompt[:300]}...")

    try:
        response = model.generate_content(final_llm_prompt)
        full_response_text = response.text if hasattr(response, 'text') else ""
        
        candidate = response.candidates[0] if response.candidates else None
        if candidate and not (candidate.finish_reason.name == "STOP" or candidate.finish_reason.name == "MAX_TOKENS"):
            finish_reason_name = candidate.finish_reason.name if candidate.finish_reason else "UNKNOWN"
            safety_ratings_info = candidate.safety_ratings
            block_message = f"Gemini response generation issue. Reason: {finish_reason_name}."
            logger.warning(f"{block_message} SafetyRatings: {safety_ratings_info}")

        return _parse_thinking_and_answer(full_response_text)

    except Exception as e:
        logger.error(f"Gemini API Call Error during synthesis: {e}", exc_info=True)
        raise ConnectionError(f"Failed to get synthesis response from Gemini: {e}")


def get_ollama_response(query: str, context_text: str, system_prompt_text: str = None,
                        model_name=None, chat_history=None) -> tuple[str, str | None]:
    if not ollama_available: raise ImportError("Ollama not available.")

    effective_model_name = model_name or DEFAULT_OLLAMA_MODEL
    final_llm_prompt = _SYNTHESIS_PROMPT_TEMPLATE_STR.format(query=query, context=context_text)

    logger.info(f"Calling Ollama for synthesis. Model: {effective_model_name}. Base URL: {OLLAMA_BASE_URL}.")
    logger.debug(f"Full prompt for Ollama (start): {final_llm_prompt[:300]}...")

    try:
        ollama_llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=effective_model_name)
        response = ollama_llm.invoke([HumanMessage(content=final_llm_prompt)])
        full_response_text = response.content if hasattr(response, 'content') else ""
        return _parse_thinking_and_answer(full_response_text)
    except Exception as e:
        logger.error(f"Ollama API Call Error during synthesis: {e}", exc_info=True)
        raise ConnectionError(f"Failed to get synthesis response from Ollama: {e}")


def get_groq_llama3_response(query: str, context_text: str, system_prompt_text: str = None,
                             model_name=None, chat_history=None) -> tuple[str, str | None]:
    if not groq_client: raise ConnectionError("Groq client not initialized.")

    effective_model_name = model_name or DEFAULT_GROQ_LLAMA3_MODEL
    final_llm_prompt = _SYNTHESIS_PROMPT_TEMPLATE_STR.format(query=query, context=context_text)

    logger.info(f"Calling Groq API for synthesis. Model: {effective_model_name}.")
    logger.debug(f"Full prompt for Groq (start): {final_llm_prompt[:300]}...")
    
    messages_for_groq = []
    if system_prompt_text:
        messages_for_groq.append({"role": "system", "content": system_prompt_text}) 
        logger.debug("Using separate system_prompt_text for Groq in addition to CoT prompt.")
    
    messages_for_groq.append({"role": "user", "content": final_llm_prompt})


    try:
        chat_completion = groq_client.chat.completions.create(
            messages=messages_for_groq,
            model=effective_model_name,
        )
        full_response_text = chat_completion.choices[0].message.content
        return _parse_thinking_and_answer(full_response_text)
    except Exception as e:
        logger.error(f"Groq LLaMA3 API Call Error during synthesis: {e}", exc_info=True)
        error_message = f"Failed to get synthesis response from Groq LLaMA3: {e}"
        if hasattr(e, 'response') and hasattr(e.response, 'json'):
            try: error_message += f" - Detail: {e.response.json()}"
            except: pass
        raise ConnectionError(error_message)


def generate_response(llm_provider: str, query: str, context_text: str,
                      chat_history, 
                      system_prompt_text=None, 
                      relevant_docs=None, 
                      llm_model_name=None) -> tuple[str, str | None]:
    """
    Main function to generate response from the chosen LLM provider using a CoT prompt.
    `query` and `context_text` are primary inputs for the CoT prompt.
    `chat_history` and `system_prompt_text` might be used differently by providers or ignored if CoT prompt is monolithic.
    """
    logger.info(f"Synthesizing response using LLM Provider: {llm_provider}, Model: {llm_model_name or 'provider_default'} with CoT prompt.")
    
    if llm_provider == "gemini":
        return get_gemini_response(
            query=query, context_text=context_text, system_prompt_text=system_prompt_text,
            model_name=llm_model_name, chat_history=chat_history
        )
    elif llm_provider == "ollama":
        return get_ollama_response(
            query=query, context_text=context_text, system_prompt_text=system_prompt_text,
            model_name=llm_model_name, chat_history=chat_history
        )
    elif llm_provider == "groq_llama3":
        return get_groq_llama3_response(
            query=query, context_text=context_text, system_prompt_text=system_prompt_text,
            model_name=llm_model_name, chat_history=chat_history
        )
    else:
        logger.error(f"Unsupported LLM provider for synthesis: {llm_provider}")
        raise ValueError(f"Unsupported LLM provider for CoT synthesis: {llm_provider}")