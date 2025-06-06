// server/routes/chat.js
const express = require('express');
const axios = require('axios'); // Make sure axios is installed (npm install axios)
const { tempAuth } = require('../middleware/authMiddleware'); // Assuming this correctly sets req.user
const ChatHistory = require('../models/ChatHistory');
const { v4: uuidv4 } = require('uuid');
// const { generateContentWithHistory } = require('../services/geminiService'); // <<<--- REMOVE OLD GEMINI SERVICE

const router = express.Router();

// Get Python AI Core Service URL from environment variables
const PYTHON_AI_SERVICE_URL = process.env.PYTHON_AI_CORE_SERVICE_URL;
if (!PYTHON_AI_SERVICE_URL) {
    console.error("FATAL ERROR: PYTHON_AI_CORE_SERVICE_URL is not set. AI features will not work.");
    // You might want to throw an error here to prevent the app from starting misconfigured
}

// --- OLD /api/chat/rag ENDPOINT - TO BE REMOVED/DEPRECATED ---
// We will integrate RAG directly into the /message endpoint's call to the new Python service
router.post('/rag', tempAuth, async (req, res) => {
    console.warn(">>> WARNING: /api/chat/rag is deprecated. RAG is now handled by /api/chat/message.");
    return res.status(410).json({ message: "This RAG endpoint is deprecated. Please use the main chat message endpoint." });
});


// --- @route   POST /api/chat/message ---
// This endpoint will now handle everything: RAG (if enabled), multi-query, LLM choice, CoT
router.post('/message', tempAuth, async (req, res) => {
    const { 
        message,                // User's current message/query
        history,                // Array of previous messages: [{role: 'user'/'model', parts: [{text: '...'}]}]
        sessionId, 
        systemPrompt,           // System prompt text from client
        isRagEnabled,           // Boolean from client: whether to perform RAG
        llmProvider,            // String from client: "gemini", "ollama", "groq_llama3" (or user preference)
        llmModelName,           // Optional: specific model name for the provider
        enableMultiQuery        // Optional boolean from client: whether to use multi-query RAG
        // relevantDocs is NO LONGER expected from client for this endpoint
    } = req.body;
    
    const userId = req.user._id.toString(); // req.user should be populated by tempAuth

    // --- Input Validations ---
    if (!message || typeof message !== 'string' || message.trim() === '') {
        return res.status(400).json({ message: 'Message text required.' });
    }
    if (!sessionId || typeof sessionId !== 'string') {
        return res.status(400).json({ message: 'Session ID required.' });
    }
    if (!Array.isArray(history)) {
        return res.status(400).json({ message: 'Invalid history format.'});
    }

    // Default values for new parameters if not provided by client
    const performRagRequest = !!isRagEnabled; // Ensure boolean
    const selectedLlmProvider = llmProvider || process.env.DEFAULT_LLM_PROVIDER_NODE || 'groq_llama3'; // Fallback
    const selectedLlmModel = llmModelName || null;
    const useMultiQuery = enableMultiQuery === undefined ? true : !!enableMultiQuery; // Default to true if not specified

    console.log(`>>> POST /api/chat/message: User=${userId}, Session=${sessionId}, RAG=${performRagRequest}, Provider=${selectedLlmProvider}, MultiQuery=${useMultiQuery}`);

    try {
        if (!PYTHON_AI_SERVICE_URL) {
            console.error("Python AI Core Service URL is not configured in Node.js environment.");
            throw new Error("AI Service communication error.");
        }

        // --- Prepare Payload for Python AI Core Service ---
        const pythonPayload = {
            user_id: userId,
            query: message.trim(),
            chat_history: history, // Pass existing history
            llm_provider: selectedLlmProvider,
            llm_model_name: selectedLlmModel, // Can be null
            system_prompt: systemPrompt,     // Can be null
            perform_rag: performRagRequest,
            enable_multi_query: useMultiQuery,
            // num_sub_queries and rag_k_per_query can also be passed if you want client control
            // otherwise Python service uses its config defaults.
        };

        console.log(`   Calling Python AI Core Service at ${PYTHON_AI_SERVICE_URL}/generate_chat_response`);
        // console.debug("   Payload for Python:", JSON.stringify(pythonPayload, null, 2)); // Uncomment for deep debugging

        // --- Call Python AI Core Service ---
        const pythonResponse = await axios.post(
            `${PYTHON_AI_SERVICE_URL}/generate_chat_response`,
            pythonPayload,
            { timeout: 60000 } // 60 second timeout (adjust as needed, RAG + LLM can take time)
        );

        if (!pythonResponse.data || pythonResponse.data.status !== 'success') {
            console.error("   Error or unexpected response from Python AI Core Service:", pythonResponse.data);
            throw new Error(pythonResponse.data?.error || "Failed to get valid response from AI service.");
        }

        const { 
            llm_response: aiReplyText, 
            references: retrievedReferences, // Array of {documentName, score, preview_snippet}
            thinking_content: thinkingContent // String or null
        } = pythonResponse.data;

        // --- Prepare Response for Client ---
        const modelResponseMessage = {
            role: 'model',
            parts: [{ text: aiReplyText || "[No response text from AI]" }],
            timestamp: new Date(),
            references: retrievedReferences || [],    // NEW: Pass references to client
            thinking: thinkingContent || null      // NEW: Pass thinking content to client
        };
        
        // The Python service now returns everything needed.
        // The old contextString construction and direct Gemini call are removed.

        console.log(`<<< POST /api/chat/message successful for session ${sessionId}.`);
        res.status(200).json({ reply: modelResponseMessage });

    } catch (error) {
        // --- Error Handling ---
        console.error(`!!! Error processing chat message for session ${sessionId}:`, error.response?.data || error.message || error);
        let statusCode = error.response?.status || 500;
        let clientMessage = "Failed to get response from AI service.";

        if (error.response?.data?.error) {
            clientMessage = error.response.data.error; // Use error from Python service if available
        } else if (error.message) {
            clientMessage = error.message;
        }
        
        // Avoid overly detailed internal errors to client for 500s
        if (statusCode === 500 && clientMessage.toLowerCase().includes("python")) {
            clientMessage = "An internal error occurred while communicating with the AI service.";
        }


        res.status(statusCode).json({ message: clientMessage });
    }
});


// --- Chat History Routes (keep as is for now, but note the model changes) ---
// When saving history, you'll now also want to save 'references' and 'thinking'
// associated with the model's messages. This requires ChatHistory.js model update.
router.post('/history', tempAuth, async (req, res) => {
    const { sessionId, messages } = req.body;
    const userId = req.user._id;
    if (!sessionId) return res.status(400).json({ message: 'Session ID required to save history.' });
    if (!Array.isArray(messages)) return res.status(400).json({ message: 'Invalid messages format.' });

    try {
        // Ensure messages include new fields if present (references, thinking)
        const validMessages = messages.map(m => ({
            role: m.role,
            parts: m.parts, // parts is an array of objects like [{text: "..."}]
            timestamp: m.timestamp,
            references: m.role === 'model' ? (m.references || []) : undefined,
            thinking: m.role === 'model' ? (m.thinking || null) : undefined,
        })).filter(m =>
            m && typeof m.role === 'string' &&
            Array.isArray(m.parts) && m.parts.length > 0 &&
            typeof m.parts[0].text === 'string' &&
            m.timestamp
        );

        if (validMessages.length === 0 && messages.length > 0) {
             console.warn(`Session ${sessionId}: No valid messages to save after filtering.`);
             // Still proceed to give a new session ID if requested
        }
        
        const newSessionId = uuidv4(); // Always generate a new session ID for the next interaction

        if (validMessages.length === 0) {
            console.log(`Session ${sessionId}: No valid messages to save. Client likely clearing history.`);
            return res.status(200).json({
                message: 'No history saved (empty or invalid messages). New session ID provided.',
                savedSessionId: null, // No session was actually saved or updated with content
                newSessionId: newSessionId
            });
        }

        const savedHistory = await ChatHistory.findOneAndUpdate(
            { sessionId: sessionId, userId: userId },
            { $set: { userId: userId, sessionId: sessionId, messages: validMessages, updatedAt: Date.now() } },
            { new: true, upsert: true, setDefaultsOnInsert: true }
        );
        
        console.log(`History saved/updated for session ${savedHistory.sessionId}. New session ID for client: ${newSessionId}`);
        res.status(200).json({
            message: 'Chat history saved successfully.',
            savedSessionId: savedHistory.sessionId,
            newSessionId: newSessionId // For the client to start the next session
        });
    } catch (error) {
        console.error(`Error saving chat history for session ${sessionId}:`, error);
        res.status(500).json({ message: 'Failed to save chat history.' });
    }
});

// GET /api/chat/sessions (no changes needed for now)
router.get('/sessions', tempAuth, async (req, res) => {
    // ... your existing code ...
    const userId = req.user._id;
    try {
        const sessions = await ChatHistory.find({ userId: userId })
            .sort({ updatedAt: -1 })
            .select('sessionId createdAt updatedAt messages') // Keep messages to get preview
            .lean();

        const sessionSummaries = sessions.map(session => {
            const firstUserMessage = session.messages?.find(m => m.role === 'user');
            let preview = 'Chat Session';
            if (firstUserMessage?.parts?.[0]?.text) {
                preview = firstUserMessage.parts[0].text.substring(0, 75);
                if (firstUserMessage.parts[0].text.length > 75) preview += '...';
            }
            return {
                sessionId: session.sessionId,
                createdAt: session.createdAt,
                updatedAt: session.updatedAt,
                messageCount: session.messages?.length || 0,
                preview: preview
            };
        });
        res.status(200).json(sessionSummaries);
    } catch (error) {
        console.error(`Error fetching chat sessions for user ${userId}:`, error);
        res.status(500).json({ message: 'Failed to retrieve chat sessions.' });
    }
});

// GET /api/chat/session/:sessionId (no changes needed for now, but client will need to handle new message fields)
router.get('/session/:sessionId', tempAuth, async (req, res) => {
    // ... your existing code ...
    const userId = req.user._id;
    const { sessionId } = req.params;
    if (!sessionId) return res.status(400).json({ message: 'Session ID parameter is required.' });
    try {
        const session = await ChatHistory.findOne({ sessionId: sessionId, userId: userId }).lean();
        if (!session) return res.status(404).json({ message: 'Chat session not found or access denied.' });
        res.status(200).json(session); // This will now include messages with 'references' and 'thinking'
    } catch (error) {
        console.error(`Error fetching chat session ${sessionId} for user ${userId}:`, error);
        res.status(500).json({ message: 'Failed to retrieve chat session details.' });
    }
});

module.exports = router;