// FusedChatbot/server/routes/analysis.js
const express = require('express');
const axios = require('axios');
const path = require('path');
const fs = require('fs');
const { tempAuth } = require('../middleware/authMiddleware'); // Assuming tempAuth sets req.user

const router = express.Router();

const PYTHON_AI_SERVICE_URL = process.env.PYTHON_AI_CORE_SERVICE_URL;

// Function to sanitize username for directory name (should match upload.js)
const sanitizeForPath = (name) => name.replace(/[^a-zA-Z0-9_-]/g, '_');

// Helper to determine subfolder based on original filename extension (simplified)
// This is a simplified version. A more robust solution would store file type/path during upload.
const determineFileTypeSubfolder = (originalFilename) => {
    const ext = path.extname(originalFilename).toLowerCase();
    // This mapping should ideally align with allowedMimeTypes in upload.js or be more robust
    if (['.pdf', '.docx', '.doc', '.pptx', '.ppt', '.txt'].includes(ext)) return 'docs';
    if (['.py', '.js', '.md', '.html', '.xml', '.json', '.csv', '.log'].includes(ext)) return 'code'; // Treating MD, JSON, CSV also as 'code' for this example
    if (['.jpg', '.jpeg', '.png', '.bmp', '.gif'].includes(ext)) return 'images'; // Though analysis on images isn't typical here
    return 'others'; // Default
};

/**
 * @route   POST /api/analysis/document
 * @desc    Trigger document analysis (FAQ, Topics, Mindmap) via Python AI Core service
 * @access  Private (requires auth via tempAuth)
 *
 * Expected req.body:
 * {
 *   "documentName": "original_filename.pdf", // Original name of the document
 *   "serverFilename": "timestamped_unique_server_filename.pdf", // The filename as stored on the server
 *   "analysisType": "faq" | "topics" | "mindmap",
 *   "llmProvider"?: "gemini" | "ollama" | "groq_llama3", // Optional
 *   "llmModelName"?: string // Optional
 * }
 */
router.post('/document', tempAuth, async (req, res) => {
    const {
        documentName,     // Original filename, primarily for user display and context
        serverFilename,   // The unique filename stored on the server by Multer
        analysisType,
        llmProvider,      // Optional: let Python service use its default if not provided
        llmModelName      // Optional
    } = req.body;

    const userId = req.user._id.toString();
    const sanitizedUsername = sanitizeForPath(req.user.username); // Use username for path as in upload.js

    // --- Input Validations ---
    if (!documentName || !serverFilename || !analysisType) {
        return res.status(400).json({ message: 'Missing required fields: documentName, serverFilename, or analysisType.' });
    }
    if (!['faq', 'topics', 'mindmap'].includes(analysisType)) {
        return res.status(400).json({ message: 'Invalid analysisType specified.' });
    }
    if (!PYTHON_AI_SERVICE_URL) {
        console.error("PYTHON_AI_CORE_SERVICE_URL is not configured.");
        return res.status(500).json({ message: "AI Service communication error." });
    }

    // --- Determine the file path on the server ---
    // This part is CRITICAL and must match how files are stored by upload.js.
    // We need to know the 'fileTypeSubfolder' (docs, code, images, others) for the given serverFilename.
    // For now, we'll try to infer it from the original documentName's extension.
    // A more robust solution would be to store this subfolder or the full path when the file is uploaded,
    // and the client would send the serverFilename which is unique.
    
    const fileTypeSubfolder = determineFileTypeSubfolder(documentName); // Infer from original name
    const absoluteFilePath = path.resolve(
        __dirname, // Current directory (routes)
        '..',      // Up to 'server' directory
        'assets',
        sanitizedUsername,
        fileTypeSubfolder,
        serverFilename // The unique name used on the server
    );

    logger.info(`Analysis request for User: ${userId}, Doc: ${documentName} (Server: ${serverFilename}), Type: ${analysisType}`);
    logger.debug(`Constructed file path for analysis: ${absoluteFilePath}`);

    if (!fs.existsSync(absoluteFilePath)) {
        logger.error(`File not found for analysis at path: ${absoluteFilePath}`);
        return res.status(404).json({ message: `Document '${documentName}' (server file: ${serverFilename}) not found on server. Path reconstruction might be incorrect or file deleted.` });
    }

    // --- Prepare Payload for Python AI Core Service ---
    const pythonPayload = {
        user_id: userId, // Python service might use this for context or if it also needs user-scoped data
        document_name: documentName, // The original name for context
        analysis_type: analysisType,
        file_path_for_analysis: absoluteFilePath, // Crucial: the absolute path Python needs
        llm_provider: llmProvider, // Can be null, Python uses default
        llm_model_name: llmModelName // Can be null
    };

    try {
        console.log(`Node.js: Calling Python /analyze_document for ${documentName}. Payload:`, 
            // Avoid logging full file_path_for_analysis if too long or sensitive in production logs
            {...pythonPayload, file_path_for_analysis: `...${path.basename(absoluteFilePath)}`} 
        );

        const pythonResponse = await axios.post(
            `${PYTHON_AI_SERVICE_URL}/analyze_document`,
            pythonPayload,
            { timeout: 180000 } // 3 minutes timeout for analysis, adjust as needed
        );

        if (!pythonResponse.data || pythonResponse.data.status !== 'success') {
            logger.error("Error or unexpected response from Python /analyze_document:", pythonResponse.data);
            return res.status(500).json({ 
                message: pythonResponse.data?.error || "Failed to get valid analysis from AI service." 
            });
        }

        logger.info(`Node.js: Successfully received analysis for ${documentName} from Python service.`);
        res.status(200).json({
            documentName: pythonResponse.data.document_name,
            analysisType: pythonResponse.data.analysis_type,
            analysisResult: pythonResponse.data.analysis_result,
            thinkingContent: pythonResponse.data.thinking_content,
            status: 'success'
        });

    } catch (error) {
        logger.error(`!!! Node.js: Error calling Python /analyze_document for ${documentName}:`, error.response?.data || error.message || error);
        let statusCode = error.response?.status || 500;
        let clientMessage = "Failed to perform document analysis.";

        if (error.response?.data?.error) {
            clientMessage = error.response.data.error;
        } else if (error.message) {
            clientMessage = error.message;
        }
        
        if (statusCode === 500 && clientMessage.toLowerCase().includes("python")) {
            clientMessage = "An internal error occurred while communicating with the AI analysis service.";
        }
        res.status(statusCode).json({ message: clientMessage });
    }
});

// Placeholder for a logger if you don't have one globally available via `logger`
// If you have a shared logger module, import it. Otherwise, basic console logging:
const logger = {
    info: console.log,
    error: console.error,
    warn: console.warn,
    debug: console.log // Change to console.debug in actual dev if needed
};


module.exports = router;