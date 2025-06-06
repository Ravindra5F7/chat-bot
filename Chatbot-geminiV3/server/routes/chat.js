// server/routes/chat.js
const express = require('express');
const axios = require('axios');
const { protect } = require('../middleware/authMiddleware');
const ChatHistory = require('../models/ChatHistory');
const { v4: uuidv4 } = require('uuid');
// Assuming getKnowledgeGraphData and formatKGDataForContext are still relevant if KG_query tool is used.
const { getKnowledgeGraphData, formatKGDataForContext } = require('../services/kgService');

// --- START OF HELPER FUNCTIONS ---
// Ensure these are defined at the top, before 'router' is used.

async function queryPythonRagService(userId, query, k = 5) {
    const pythonServiceUrl = process.env.PYTHON_RAG_SERVICE_URL; // This should be your RAG service (e.g., port 5002)
    if (!pythonServiceUrl) {
        console.error("PYTHON_RAG_SERVICE_URL is not set. Cannot query RAG service.");
        throw new Error("RAG service configuration error.");
    }
    const queryUrl = `${pythonServiceUrl}/query`; // Ensure your RAG service has a /query endpoint
    console.log(`[NodeJS] Querying Python RAG service for User ${userId} at ${queryUrl} with query "${query}" and k=${k}`);
    try {
        const response = await axios.post(queryUrl, {
            user_id: userId,
            query: query,
            k: k
        }, { timeout: 30000 });

        if (response.data && Array.isArray(response.data.relevantDocs)) {
            console.log(`[NodeJS] Python RAG service returned ${response.data.relevantDocs.length} results for query "${query}".`);
            // Your existing logic for relevantDocsWithKg
            const relevantDocsWithKg = response.data.relevantDocs.map(doc => {
                if (doc && doc.metadata && doc.metadata.kg_data) {
                    return { ...doc, kg_data: doc.metadata.kg_data };
                }
                return doc;
            });
            return relevantDocsWithKg;
        } else {
            console.warn(`[NodeJS] Python RAG service returned unexpected data structure for query "${query}":`, response.data);
            return [];
        }
    } catch (error) {
        console.error(`[NodeJS] Error querying Python RAG service for User ${userId}, query "${query}":`);
        if (error.response) {
            console.error('   RAG Error Data:', error.response.data);
            console.error('   RAG Error Status:', error.response.status);
        } else if (error.request) {
            console.error('   RAG Error Request:', error.request);
            console.error('   No response received from Python RAG service.');
        } else {
            console.error('   RAG Error Message:', error.message);
        }
        return [];
    }
}

async function queryPythonKgService(query, documentName = null) {
    // This PYTHON_NOTEBOOK_SERVICE_URL should point to your Flask app (app.py)
    const pythonNotebookServiceUrl = `http://localhost:${process.env.NOTEBOOK_BACKEND_PORT || 5000}`;
    const endpoint = `/get_kg_data`; // From your app.py
    let queryUrl = `${pythonNotebookServiceUrl}${endpoint}`;
    if (documentName) {
        queryUrl += `?filename=${encodeURIComponent(documentName)}`;
    } else if (query) { // If your /get_kg_data can handle a general query param
        // queryUrl += `?query=${encodeURIComponent(query)}`; // This endpoint in app.py only takes filename
        console.warn("[NodeJS] queryPythonKgService called with query but no documentName. The current /get_kg_data in app.py primarily uses filename.");
        // For now, if no documentName, we can't do much with just a query to this endpoint
        if (!documentName) return null;
    }


    console.log(`[NodeJS] Querying Python KG service for document: '${documentName || 'general query (unsupported for now)'}' at ${queryUrl}`);
    try {
        const response = await axios.get(queryUrl, { timeout: 30000 });
        if (response.data && response.data.kg_data) {
            console.log(`[NodeJS] Python KG service returned KG data for ${documentName}`);
            return response.data.kg_data;
        } else {
            console.warn(`[NodeJS] Python KG service returned no data or unexpected structure for ${documentName}:`, response.data);
            return null;
        }
    } catch (error) {
        console.error(`[NodeJS] Error querying Python KG service for ${documentName}:`);
        // ... (detailed error logging as above) ...
        return null;
    }
}

async function executePythonCode(code) {
    const pythonExecutorUrl = process.env.PYTHON_EXECUTOR_SERVICE_URL; // Separate service if you have one
    if (!pythonExecutorUrl) {
        console.error("PYTHON_EXECUTOR_SERVICE_URL is not set.");
        // Fallback to notebook service if it has an /execute_code endpoint
        const pythonNotebookServiceUrl = `http://localhost:${process.env.NOTEBOOK_BACKEND_PORT || 5000}`;
        if (true) { // Assuming your main app.py will handle /execute_code
             // Modify if your app.py doesn't have /execute_code
            console.warn("PYTHON_EXECUTOR_SERVICE_URL not set, attempting to use NOTEBOOK_BACKEND_PORT for /execute_code (this needs to be implemented in app.py).");
            // return { output: "", error: "Python Executor service URL not configured." };
        } else {
           // throw new Error("Python Executor service configuration error.");
        }
    }
    const executeUrl = `${pythonExecutorUrl || `http://localhost:${process.env.NOTEBOOK_BACKEND_PORT || 5000}`}/execute_code`; // Modify if different
    console.log(`[NodeJS] Calling Python Executor service at ${executeUrl}`);
    try {
        const response = await axios.post(executeUrl, { code: code }, { timeout: 60000 });
        if (response.data) {
            console.log(`[NodeJS] Python Executor service returned: Output: ${response.data.output}, Error: ${response.data.error}`);
            return response.data;
        } else {
            console.warn(`[NodeJS] Python Executor service returned unexpected data structure:`, response.data);
            return { output: "", error: "Unexpected response from executor." };
        }
    } catch (error) {
        console.error(`[NodeJS] Error calling Python Executor service:`);
        // ... (detailed error logging) ...
        return { output: "", error: `Execution failed: ${error.message}` };
    }
}
// --- END OF HELPER FUNCTIONS ---

const router = express.Router();

// --- @route POST /api/chat/rag ---
// This route is called by ChatPage.js if RAG is enabled before sending the main message
router.post('/rag', protect, async (req, res) => {
    const { message } = req.body;
    const userId = req.user._id.toString();

    if (!message || typeof message !== 'string' || message.trim() === '') {
        return res.status(400).json({ message: 'RAG query message text required.' });
    }
    console.log(`[NodeJS] >>> POST /api/chat/rag: User=${userId}, Query="${message}"`);
    try {
        const kValue = 5;
        const relevantDocs = await queryPythonRagService(userId, message.trim(), kValue); // This should now work
        console.log(`[NodeJS] <<< POST /api/chat/rag successful for User ${userId}. Found ${relevantDocs.length} docs for query "${message}".`);
        res.status(200).json({ relevantDocs });
    } catch (error) {
        console.error(`[NodeJS] !!! Error in /api/chat/rag route for User ${userId}, Query="${message}":`, error);
        res.status(500).json({ message: "Failed to retrieve relevant documents for RAG." });
    }
});


// --- @route POST /api/chat/message ---
router.post('/message', protect, async (req, res) => {
    // Frontend ChatPage.js sends: message, history, sessionId, systemPrompt, isRagEnabled, relevantDocs, llm_preference
    const {
        message,
        history,        // from frontend (ChatPage.js messages state)
        sessionId,
        systemPrompt,   // from frontend (ChatPage.js editableSystemPromptText)
        isRagEnabled,   // from frontend (ChatPage.js isRagEnabled state)
        relevantDocs,   // from frontend (result of /api/chat/rag call if isRagEnabled)
        llm_preference  // from frontend (ChatPage.js selectedLlm state)
    } = req.body;
    const userId = req.user._id.toString();

    if (!message || typeof message !== 'string' || message.trim() === '') return res.status(400).json({ message: 'Message text required.' });
    if (!sessionId || typeof sessionId !== 'string') return res.status(400).json({ message: 'Session ID required.' });

    // Log the received llm_preference
    console.log(`[NodeJS] >>> POST /api/chat/message: User=${userId}, Session=${sessionId}, LLM Pref RECV: ${llm_preference}`);

    try {
        await ChatHistory.findOneAndUpdate(
            { userId: userId, sessionId: sessionId },
            {
                $push: { messages: { role: 'user', parts: [{ text: message.trim() }], timestamp: new Date() } },
                $set: { updatedAt: new Date() }, $setOnInsert: { createdAt: new Date() }
            },
            { upsert: true, new: true, setDefaultsOnInsert: true }
        );
        console.log(`[NodeJS] User message saved to ChatHistory for session ${sessionId}.`);

        // Prepare payload for Python Flask app (app.py) /chat endpoint
        // Your app.py /chat endpoint expects: query, session_id, chat_history (formatted), tool_output
        // It does NOT directly take systemPrompt, isRagEnabled, relevantDocs from this Node.js payload.
        // The ReAct loop within app.py will call RAG_search tool if needed.

        const historyForPython = history.map(msg => ({ // Convert to {role, content} format if needed by Python
            role: msg.role,
            content: msg.parts.map(part => part.text || '').join(' ') // Assuming Python wants 'content'
        }));

        const llmPayloadForPython = {
            query: message.trim(), // This is the 'content' for the LLM
            session_id: sessionId,
            // chat_history: historyForPython, // Your app.py fetches history from its DB, so this might be redundant or for overriding
                                            // Based on app.py, it uses messages_from_db then formats to chat_history_for_llm
                                            // So, sending history from Node.js might not be used by app.py's ReAct loop.
                                            // Let's send it just in case, app.py can decide.
            chat_history: historyForPython, 
            tool_output: "", // Initial tool_output for the first turn in ReAct loop
            // llm_preference IS NOT directly used by app.py /chat endpoint in its initial payload.
            // llm_preference from Node.js would need to be passed into ai_core.synthesize_chat_response if you want to select LLM there.
            // For now, let's assume ai_core.py uses a configured default or handles selection.
            // If you want to pass llm_preference, modify ai_core.synthesize_chat_response to accept it.
        };
        // Add llm_preference if your Python /chat endpoint or ai_core.py can use it:
        // llmPayloadForPython.llm_preference = llm_preference;


        console.log(`[NodeJS]   Payload to Python Flask app /chat:`, JSON.stringify(llmPayloadForPython, null, 2));

        const pythonChatServiceUrl = `http://localhost:${process.env.NOTEBOOK_BACKEND_PORT || 5000}`;
        const responseFromPython = await axios.post(`${pythonChatServiceUrl}/chat`, llmPayloadForPython, { timeout: 90000 });

        const pythonResponseData = responseFromPython.data;

        if (pythonResponseData.error) {
            console.error(`[NodeJS] Python /chat service returned an error:`, pythonResponseData.error);
            // Construct a valid model reply structure even for errors from Python
            const errorReply = {
                role: 'model',
                parts: [{ text: pythonResponseData.answer || pythonResponseData.error || "Error from AI service." }],
                references: pythonResponseData.references || [],
                thinking: pythonResponseData.thinking || "Error occurred in Python service.",
                llm_used: pythonResponseData.llm_used || llm_preference || 'default', // Use preference if Python doesn't specify
                timestamp: new Date()
            };
            await ChatHistory.findOneAndUpdate(
                { userId: userId, sessionId: sessionId },
                { $push: { messages: errorReply }, $set: { updatedAt: new Date() } }
            );
            return res.status(200).json({ reply: errorReply, thinking: errorReply.thinking }); // Send 200 but with error message in reply
        }

        // Assuming pythonResponseData contains: answer, session_id (can ignore), references, thinking
        const modelResponseMessage = {
            role: 'model',
            parts: [{ text: pythonResponseData.answer || "No response content." }],
            references: pythonResponseData.references || [],
            thinking: pythonResponseData.thinking || "",
            llm_used: pythonResponseData.llm_used || llm_preference || 'default',
            timestamp: new Date()
        };

        await ChatHistory.findOneAndUpdate(
            { userId: userId, sessionId: sessionId },
            { $push: { messages: modelResponseMessage }, $set: { updatedAt: new Date() } }
        );
        console.log(`[NodeJS] Model response saved to ChatHistory for session ${sessionId}.`);

        console.log(`[NodeJS] <<< POST /api/chat/message successful for session ${sessionId}.`);
        res.status(200).json({
            reply: modelResponseMessage,
            thinking: modelResponseMessage.thinking
        });

    } catch (error) {
        console.error(`[NodeJS] !!! Error in POST /api/chat/message for session ${sessionId}:`);
        let clientMessage = "Failed to get response due to a server error.";
        let thinkingOnError = "";

        if (error.response) { // Axios error (error from Python service)
            console.error('   Error Data from Python service:', error.response.data);
            console.error('   Error Status from Python service:', error.response.status);
            clientMessage = error.response.data.answer || error.response.data.error || error.response.data.message || "Error from AI sub-service.";
            thinkingOnError = error.response.data.thinking || `Python service responded with status ${error.response.status}.`;
        } else if (error.request) { // Axios error (no response from Python service)
            console.error('   Error Request (no response from Python):', error.request);
            clientMessage = "AI sub-service did not respond.";
            thinkingOnError = "Connection to Python service failed or timed out.";
        } else { // Other errors (e.g., in Node.js logic itself)
            console.error('   Error Message:', error.message);
            clientMessage = error.message; // Be cautious about exposing internal error messages
            if (error instanceof ReferenceError) {
                clientMessage = "An internal variable error occurred in Node.js. Please check server logs.";
            }
        }
        
        // Construct a valid model reply structure for errors
        const errorReplyForClient = {
            role: 'model',
            parts: [{ text: clientMessage }],
            references: [],
            thinking: thinkingOnError,
            llm_used: llm_preference || 'default',
            timestamp: new Date()
        };
         try {
            await ChatHistory.findOneAndUpdate(
                { userId: userId, sessionId: sessionId },
                { $push: { messages: errorReplyForClient }, $set: { updatedAt: new Date() } }
            );
        } catch (dbErr) {
            console.error("[NodeJS] Failed to save error message to DB:", dbErr);
        }

        res.status(500).json({ // Send 500 for Node.js level errors or if Python returns 500
            reply: errorReplyForClient, // Send the structured error back
            thinking: thinkingOnError,
            message: clientMessage // Also include a top-level message for ChatPage's error display
        });
    }
});

// ... (rest of your /history, /sessions, /session/:sessionId routes - they seem okay) ...
// Make sure to include them from your previous correct version. Example:
router.post('/history', protect, async (req, res) => { /* Your existing history save logic */ });
router.get('/sessions', protect, async (req, res) => { /* Your existing sessions list logic */ });
router.get('/session/:sessionId', protect, async (req, res) => { /* Your existing single session fetch logic */ });


module.exports = router;