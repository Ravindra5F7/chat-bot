// client/src/components/ChatPage.js
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
    sendMessage, saveChatHistory, getUserFiles, queryRagService, 
    getAnalysis, getKgDataForVisualization 
} from '../services/api';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { v4 as uuidv4 } from 'uuid';

import SystemPromptWidget, { availablePrompts, getPromptTextById } from './SystemPromptWidget';
import HistoryModal from './HistoryModal';
import FileUploadWidget from './FileUploadWidget';
import FileManagerWidget from './FileManagerWidget';
// import ChatHistory from './ChatHistory'; // Only if used directly
import FAQAnalysis from './FAQAnalysis';
import TopicAnalysis from './TopicAnalysis';
import MindmapAnalysis from './MindmapAnalysis';
import KnowledgeGraphViewer from './KnowledgeGraphViewer';
import CloseIcon from '@mui/icons-material/Close';

import './ChatPage.css';
import { 
    FormControl, InputLabel, Select, MenuItem, Button, Box, TextField, Typography, 
    CircularProgress, IconButton, AppBar, Toolbar, Drawer, List, ListItem, ListItemText, 
    Divider, Snackbar, Alert, Tooltip, Dialog, DialogTitle, DialogContent, DialogActions 
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import MenuIcon from '@mui/icons-material/Menu'; // Example, include icons you use
import HistoryIcon from '@mui/icons-material/History';
import LogoutIcon from '@mui/icons-material/Logout';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import ListIcon from '@mui/icons-material/List';
import BubbleChartIcon from '@mui/icons-material/BubbleChart';
import MapIcon from '@mui/icons-material/Map';


const ChatPage = ({ setIsAuthenticated }) => {
    const [messages, setMessages] = useState([]);
    const [inputText, setInputText] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [isRagLoading, setIsRagLoading] = useState(false);
    const [error, setError] = useState('');
    const [sessionId, setSessionId] = useState('');
    const [userId, setUserId] = useState('');
    const [username, setUsername] = useState('');
    const [currentSystemPromptId, setCurrentSystemPromptId] = useState('friendly');
    const [editableSystemPromptText, setEditableSystemPromptText] = useState(() => getPromptTextById('friendly'));
    const [isHistoryModalOpen, setIsHistoryModalOpen] = useState(false);
    const [fileRefreshTrigger, setFileRefreshTrigger] = useState(0);
    const [hasFiles, setHasFiles] = useState(false);
    const [isRagEnabled, setIsRagEnabled] = useState(false);
    const [activeSidebar, setActiveSidebar] = useState(null);
    const [openSnackbar, setOpenSnackbar] = useState(false);
    const [snackbarMessage, setSnackbarMessage] = useState('');
    const [snackbarSeverity, setSnackbarSeverity] = useState('info');
    const [selectedLlm, setSelectedLlm] = useState('gemini'); // Default LLM

    const [analysisData, setAnalysisData] = useState(null);
    const [analysisLoading, setAnalysisLoading] = useState(false);
    const [analysisError, setAnalysisError] = useState(null);
    const [selectedAnalysisFile, setSelectedAnalysisFile] = useState(null);

    const [kgData, setKgData] = useState(null);
    const [kgLoading, setKgLoading] = useState(false);
    const [kgError, setKgError] = useState(null);

    const messagesEndRef = useRef(null);
    const navigate = useNavigate();

    const toggleSidebar = (type) => {
        setActiveSidebar(prev => (prev === type ? null : type));
    };

    useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

    const performLogoutCleanup = useCallback(() => {
        console.log("[ChatPage] Performing logout cleanup...");
        localStorage.clear();
        setIsAuthenticated(false);
        setMessages([]); setSessionId(''); setUserId(''); setUsername('');
        setCurrentSystemPromptId('friendly');
        setEditableSystemPromptText(getPromptTextById('friendly')); setError('');
        setHasFiles(false); setIsRagEnabled(false);
        setActiveSidebar(null); setAnalysisData(null); setSelectedAnalysisFile(null); setKgData(null);
        requestAnimationFrame(() => {
            if (window.location.pathname !== '/login') navigate('/login', { replace: true });
        });
    }, [navigate, setIsAuthenticated]);

    useEffect(() => {
        const storedToken = localStorage.getItem('token');
        const storedUserId = localStorage.getItem('userId');
        const storedUsername = localStorage.getItem('username');
        let sessionIdFromStorage = localStorage.getItem('sessionId');

        if (!storedUserId || !storedToken || !storedUsername) {
            console.warn("[ChatPage] Mount: Missing essential auth info. Performing cleanup.");
            performLogoutCleanup();
        } else {
            console.log("[ChatPage] Mount: Auth info found. Setting user state.");
            setUserId(storedUserId);
            setUsername(storedUsername);
            if (sessionIdFromStorage) {
                setSessionId(sessionIdFromStorage);
                console.log("[ChatPage] Mount: Using existing sessionId:", sessionIdFromStorage);
            } else {
                const newClientSessionId = uuidv4();
                localStorage.setItem('sessionId', newClientSessionId);
                setSessionId(newClientSessionId);
                console.log("[ChatPage] Mount: Generated new client-side sessionId:", newClientSessionId);
            }
        }
    }, [performLogoutCleanup]); // Only performLogoutCleanup as dep as it includes navigate/setIsAuthenticated

    const handleLogout = useCallback((skipSave = false) => {
        if (!skipSave && messages.length > 0) {
            saveAndReset(true, performLogoutCleanup);
        } else {
            performLogoutCleanup();
        }
    }, [messages.length, performLogoutCleanup]); // Removed saveAndReset from here, will define it next

    const handlePromptSelectChange = useCallback((newId) => {
        setCurrentSystemPromptId(newId);
        setEditableSystemPromptText(getPromptTextById(newId));
        // Avoid clearing critical errors with transient messages
        setError(prev => (prev && (prev.includes("Session invalid") || prev.includes("Critical Error"))) ? prev : `Assistant mode changed.`);
        setTimeout(() => { setError(prev => prev === `Assistant mode changed.` ? '' : prev); }, 3000);
    }, []);

    const saveAndReset = useCallback(async (isLoggingOut = false, onCompleteCallback = null) => {
        const currentSessionIdFromStorage = localStorage.getItem('sessionId');
        const currentUserIdFromStorage = localStorage.getItem('userId');
        const messagesToSave = [...messages];

        if (!currentUserIdFromStorage || !currentSessionIdFromStorage) {
            console.error("[ChatPage] Save Error: Session ID or User ID missing from localStorage.");
            setError("Critical Error: Session info missing. Please re-login.");
            if (isLoggingOut) performLogoutCleanup();
            if (onCompleteCallback) onCompleteCallback();
            return;
        }
        if (isLoading || isRagLoading || messagesToSave.length === 0) {
            if (onCompleteCallback) onCompleteCallback();
            return;
        }

        setIsLoading(true);
        let newBackendSessionId = null;
        try {
            console.log(`[ChatPage] Saving history for session: ${currentSessionIdFromStorage} (User: ${currentUserIdFromStorage})`);
            const response = await saveChatHistory({ sessionId: currentSessionIdFromStorage, messages: messagesToSave });
            newBackendSessionId = response.data.newSessionId;
            if (!newBackendSessionId) throw new Error("Backend failed to provide new session ID for save.");

            localStorage.setItem('sessionId', newBackendSessionId);
            setSessionId(newBackendSessionId);
            setMessages([]);
            if (!isLoggingOut) {
                handlePromptSelectChange('friendly'); // Reset prompt
                setError(''); // Clear non-critical errors
            }
            console.log(`[ChatPage] History saved. New session ID from backend: ${newBackendSessionId}`);
        } catch (err) {
            const failErrorMsg = err.response?.data?.message || err.message || 'Failed to save/reset session.';
            console.error("[ChatPage] Save/Reset Error:", err.response || err);
            setError(`Session Error: ${failErrorMsg}`);
            if (err.response?.status === 401) {
                console.warn("[ChatPage] Received 401 saving history, logging out.");
                performLogoutCleanup();
            } else if (!newBackendSessionId && !isLoggingOut) {
                const clientFallbackSessionId = uuidv4();
                localStorage.setItem('sessionId', clientFallbackSessionId);
                setSessionId(clientFallbackSessionId);
                setMessages([]);
                handlePromptSelectChange('friendly');
                console.warn("[ChatPage] Save failed, generated new client-side session ID:", clientFallbackSessionId);
            }
        } finally {
            setIsLoading(false);
            if (onCompleteCallback) onCompleteCallback();
        }
    }, [messages, isLoading, isRagLoading, performLogoutCleanup, handlePromptSelectChange]);
    
    // Now handleLogout can be defined as it depends on saveAndReset
    // No, handleLogout was defined correctly before, its dependency on saveAndReset is fine.

    useEffect(() => {
        const checkUserFiles = async () => {
            if (!userId) { // Check component state
                console.log("[ChatPage] User files check skipped: No userId state.");
                setHasFiles(false); setIsRagEnabled(false); return;
            }
            console.log("[ChatPage] Checking user files for userId (from state):", userId);
            try {
                const response = await getUserFiles();
                const filesExist = response.data && response.data.length > 0;
                setHasFiles(filesExist);
                setIsRagEnabled(filesExist); // Default RAG to true if files exist
                console.log(`[ChatPage] User files check: ${filesExist ? `${response.data.length} files. RAG default: true` : 'No files. RAG default: false'}`);
            } catch (err) {
                console.error("[ChatPage] Error checking user files:", err);
                if (err.response?.status === 401 && !window.location.pathname.includes('/login')) {
                    console.warn("[ChatPage] Received 401 checking files, logging out.");
                    handleLogout(true); // Ensure handleLogout is stable
                } else {
                    setError("Could not check user files.");
                    setHasFiles(false); setIsRagEnabled(false);
                }
            }
        };
        if (userId) checkUserFiles();
    }, [userId, fileRefreshTrigger, handleLogout]);


    const triggerFileRefresh = useCallback(() => setFileRefreshTrigger(prev => prev + 1), []);
    const handlePromptTextChange = useCallback((newText) => {
        setEditableSystemPromptText(newText);
        const matchingPreset = availablePrompts.find(p => p.id !== 'custom' && p.prompt === newText);
        setCurrentSystemPromptId(matchingPreset ? matchingPreset.id : 'custom');
    }, []);
    const handleHistory = useCallback(() => setIsHistoryModalOpen(true), []);
    const closeHistoryModal = useCallback(() => setIsHistoryModalOpen(false), []);
    
    const handleLlmChange = useCallback((event) => {
        const newLlm = event.target.value;
        setSelectedLlm(newLlm);
        setSnackbarMessage(`LLM changed to: ${newLlm}`);
        setSnackbarSeverity('info');
        setOpenSnackbar(true);
        console.log("[ChatPage] LLM changed to:", newLlm);
    }, []);

    const fetchAnalysisData = useCallback(async (filename, analysisType) => {
        if (!filename || !analysisType) return;
        console.log(`[ChatPage] Fetching ${analysisType} analysis for ${filename} using ${selectedLlm}`);
        setAnalysisLoading(true); setAnalysisError(null); setAnalysisData(null);
        try {
            const response = await getAnalysis(filename, analysisType, selectedLlm); // Pass selectedLlm
            setAnalysisData(response.data);
        } catch (err) {
            console.error(`[ChatPage] Error fetching ${analysisType} analysis:`, err);
            setAnalysisError(err.response?.data?.message || err.message || `Failed to fetch ${analysisType} analysis`);
        } finally {
            setAnalysisLoading(false);
        }
    }, [selectedLlm]); // Add selectedLlm dependency

    const fetchKgData = useCallback(async (filename) => {
        if (!filename) return;
        console.log(`[ChatPage] Fetching KG data for ${filename}`);
        setKgLoading(true); setKgError(null); setKgData(null);
        try {
            const response = await getKgDataForVisualization(filename);
            setKgData(response.data);
        } catch (err) {
            console.error('[ChatPage] Error fetching KG data:', err);
            setKgError(err.response?.data?.message || err.message || 'Failed to fetch knowledge graph data');
        } finally {
            setKgLoading(false);
        }
    }, []);


    const handleFileSelectForAnalysis = useCallback((filename) => {
        console.log(`[ChatPage] File selected for analysis: ${filename}, current active sidebar: ${activeSidebar}`);
        setSelectedAnalysisFile(filename); // Set the selected file
        // Decide which data to fetch based on the currently active sidebar or default
        if (activeSidebar === 'faq') fetchAnalysisData(filename, 'faq');
        else if (activeSidebar === 'topics') fetchAnalysisData(filename, 'topics');
        else if (activeSidebar === 'mindmap') fetchAnalysisData(filename, 'mindmap');
        else if (activeSidebar === 'knowledgeGraph') fetchKgData(filename);
        else {
            // If no specific sidebar is active, maybe default to opening one and fetching data
            // For example, default to FAQ if a file is selected and no sidebar is open.
            // Or simply set selectedAnalysisFile and let sidebar toggle fetch data.
            console.log("[ChatPage] File selected, but no specific analysis sidebar active to auto-fetch.");
        }
        setSnackbarMessage(`File selected: ${filename}. Open an analysis tab to view.`);
        setSnackbarSeverity('info');
        setOpenSnackbar(true);
    }, [activeSidebar, fetchAnalysisData, fetchKgData]);


    const handleNewChat = useCallback(() => {
        if (!isLoading && !isRagLoading) saveAndReset(false);
    }, [isLoading, isRagLoading, saveAndReset]);

    const handleSendMessage = useCallback(async (e) => {
        if (e) e.preventDefault();
        const textToSend = inputText.trim();
        const currentStoredSessionId = localStorage.getItem('sessionId');
        const currentStoredUserId = localStorage.getItem('userId');

        if (!textToSend || isLoading || isRagLoading) return;
        if (!currentStoredSessionId || !currentStoredUserId) {
            console.error("[ChatPage] SendMessage Error: SessionId or UserId missing from localStorage.", { currentStoredSessionId, currentStoredUserId });
            setError("Session invalid. Please refresh or log in again.");
            if (!currentStoredUserId) {
                console.warn("[ChatPage] SendMessage: UserId missing, forcing logout.");
                handleLogout(true);
            }
            return;
        }

        const newUserMessage = { role: 'user', parts: [{ text: textToSend }], timestamp: new Date() };
        const currentMessagesBeforeSend = [...messages]; // For potential rollback
        setMessages(prev => [...prev, newUserMessage]);
        setInputText('');
        // Keep critical errors, clear transient ones
        setError(prev => (prev && (prev.includes("Session invalid") || prev.includes("Critical Error"))) ? prev : '');


        let ragDocsForPayload = []; // Renamed to avoid conflict with relevantDocs from API response
        let ragQueryError = null;

        if (isRagEnabled && hasFiles) { // Only attempt RAG if enabled AND files exist
            setIsRagLoading(true);
            try {
                console.log("[ChatPage] RAG Enabled: Querying backend /api/chat/rag...");
                const ragApiResponse = await queryRagService({ message: textToSend });
                ragDocsForPayload = ragApiResponse.data.relevantDocs || [];
                console.log(`[ChatPage] RAG Query returned ${ragDocsForPayload.length} documents.`);
            } catch (err) {
                console.error("[ChatPage] RAG Query Error:", err.response || err);
                ragQueryError = err.response?.data?.message || "Failed to retrieve documents for RAG.";
                if (err.response?.status === 401) {
                    console.warn("[ChatPage] Received 401 during RAG query, logging out.");
                    handleLogout(true); setIsRagLoading(false); return;
                }
            } finally {
                setIsRagLoading(false);
            }
        } else {
            console.log(`[ChatPage] RAG Skipping: isRagEnabled=${isRagEnabled}, hasFiles=${hasFiles}`);
        }

        setIsLoading(true);
        const systemPromptForApi = editableSystemPromptText;

        // ** ADDED LOGS FOR LLM PREFERENCE **
        console.log("[ChatPage] selectedLlm state before API call:", selectedLlm);

        const payloadToNodeJs = {
            message: textToSend,
            history: currentMessagesBeforeSend, // History before new user message
            sessionId: currentStoredSessionId,
            systemPrompt: systemPromptForApi,    // Pass system prompt
            isRagEnabled: isRagEnabled && hasFiles && ragDocsForPayload.length > 0, // Effective RAG status
            relevantDocs: ragDocsForPayload,     // Pass docs found by /api/chat/rag
            llm_preference: selectedLlm,         // Pass selected LLM
        };
        console.log(`[ChatPage] Sending payload to Node.js /api/chat/message:`, payloadToNodeJs);


        try {
            if (ragQueryError) { // Show RAG error if it occurred
                setError(prev => prev ? `${prev} | RAG Error: ${ragQueryError}` : `RAG Error: ${ragQueryError}`);
            }

            const apiResponse = await sendMessage(payloadToNodeJs);

            const modelReplyFromServer = apiResponse.data.reply;
            if (modelReplyFromServer?.role && modelReplyFromServer?.parts?.length > 0) {
                setMessages(prev => [...prev, modelReplyFromServer]);
                 // Clear only non-critical errors after successful message
                setError(prev => (prev && (prev.includes("Session invalid") || prev.includes("Critical Error"))) ? prev : '');
            } else {
                 console.warn("[ChatPage] Invalid reply structure from backend:", apiResponse.data);
                throw new Error("Invalid reply structure received from backend.");
            }

        } catch (err) {
            const errorMessage = err.response?.data?.message || err.response?.data?.reply?.parts?.[0]?.text || err.message || 'Failed to get response.';
            setError(prev => prev ? `${prev} | Chat Error: ${errorMessage}` : `Chat Error: ${errorMessage}`);
            console.error("[ChatPage] Send Message Error:", err.response || err);
            setMessages(currentMessagesBeforeSend); // Rollback UI
            if (err.response?.status === 401 && !window.location.pathname.includes('/login')) {
                console.warn("[ChatPage] Received 401 sending message, logging out.");
                handleLogout(true);
            }
        } finally {
            setIsLoading(false);
        }
    }, [
        inputText, isLoading, isRagLoading, messages, editableSystemPromptText, 
        isRagEnabled, hasFiles, handleLogout, selectedLlm 
    ]);

    const handleEnterKey = useCallback((e) => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSendMessage(); }
    }, [handleSendMessage]);

    const handleRagToggle = (event) => setIsRagEnabled(event.target.checked);
    const isProcessing = isLoading || isRagLoading;

    if (!userId) return <div className="loading-indicator"><span>Initializing authentication...</span></div>;

    return (
        // Keep your existing JSX structure from the previously provided full ChatPage.js
        // The key changes are in the logic, especially handleSendMessage.
        // Make sure the onClicks for sidebar icons correctly call fetchAnalysisData/fetchKgData
        // Example for KG icon:
        // <button onClick={() => { 
        //     toggleSidebar('knowledgeGraph'); 
        //     if (selectedAnalysisFile) fetchKgData(selectedAnalysisFile); 
        // }}>
        // ... your JSX for the page ...
        <div className="chat-page-container">
            <div className="sidebar-area">
                <SystemPromptWidget
                    selectedPromptId={currentSystemPromptId} promptText={editableSystemPromptText}
                    onSelectChange={handlePromptSelectChange} onTextChange={handlePromptTextChange} />
                <FileUploadWidget onUploadSuccess={triggerFileRefresh} userId={userId} /> {/* Pass userId if needed by widget */}
                <FileManagerWidget
                    refreshTrigger={fileRefreshTrigger}
                    onFileSelect={handleFileSelectForAnalysis} // This sets selectedAnalysisFile
                    userId={userId} // Pass userId if needed
                />
            </div>

            <div className={`chat-container ${activeSidebar ? 'with-sidebar' : ''}`}>
                <header className="chat-header">
                    <h1>Engineering Tutor</h1>
                    <div className="header-controls">
                        <span className="username-display">Hi, {username}!</span>
                        <FormControl variant="outlined" size="small" sx={{ m: 1, minWidth: 120 }}>
                            <InputLabel id="llm-select-label">LLM</InputLabel>
                            <Select
                                labelId="llm-select-label" id="llm-select"
                                value={selectedLlm} label="LLM"
                                onChange={handleLlmChange} disabled={isProcessing}
                            >
                                <MenuItem value="gemini">Gemini (Default)</MenuItem>
                                {/* Add other LLMs your Python backend supports by name */}
                                <MenuItem value="ollama_llama3">Ollama (Llama 3)</MenuItem>
                                <MenuItem value="ollama_deepseek">Ollama (DeepSeek)</MenuItem>
                            </Select>
                        </FormControl>
                        <Button onClick={handleHistory} disabled={isProcessing} startIcon={<HistoryIcon />}>History</Button>
                        <Button onClick={handleNewChat} disabled={isProcessing}>New Chat</Button>
                        <Button onClick={() => handleLogout(false)} disabled={isProcessing} startIcon={<LogoutIcon />}>Logout</Button>
                    </div>
                </header>

                <Button
                    className="export-pdf-button" /* Add appropriate styling */
                    variant="outlined"
                    size="small"
                    style={{ margin: '8px auto', display: 'block' }}
                    disabled={isProcessing || messages.length === 0}
                    onClick={async () => {
                        const currentPdfSessionId = localStorage.getItem('sessionId');
                        if (!currentPdfSessionId) {
                            alert('Session ID is missing, cannot export PDF.'); return;
                        }
                        try {
                            const response = await fetch(`/api/export_chat_pdf?sessionId=${currentPdfSessionId}&type=both`); // Use /api prefix
                            if (!response.ok) throw new Error(`Failed to download PDF: ${response.statusText}`);
                            const blob = await response.blob();
                            const url = window.URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url;
                            a.download = `chat_export_${currentPdfSessionId}.pdf`;
                            document.body.appendChild(a); a.click(); a.remove();
                            window.URL.revokeObjectURL(url);
                        } catch (pdfError) {
                            console.error('[ChatPage] Error downloading chat PDF:', pdfError);
                            setSnackbarMessage(`Failed to download PDF: ${pdfError.message}`);
                            setSnackbarSeverity('error'); setOpenSnackbar(true);
                        }
                    }}
                    title="Download Chat as PDF"
                >
                    Download Chat as PDF
                </Button>

                <div className="messages-area">
                    {messages.map((msg, index) => {
                        if (!msg?.role || !msg?.parts?.length || !msg.timestamp) {
                            console.warn("[ChatPage] Rendering invalid message structure at index", index, msg);
                            return <div key={`error-${index}`} className="message-error">Msg Error</div>;
                        }
                        const messageText = msg.parts[0]?.text || '';
                        // Add display for references if they exist on model messages
                        const references = msg.role === 'model' && msg.references && msg.references.length > 0
                            ? msg.references
                            : null;

                        return (
                            <div key={`${sessionId}-${index}-${msg.timestamp}`} className={`message ${msg.role}`}>
                                <div className="message-content">
                                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{messageText}</ReactMarkdown>
                                    {references && (
                                        <div className="message-references">
                                            <strong>Sources:</strong>
                                            <ul>
                                                {references.map((ref, refIdx) => (
                                                    <li key={refIdx} title={ref.content?.substring(0,200)}>
                                                        {ref.source || 'Unknown source'}
                                                    </li>
                                                ))}
                                            </ul>
                                        </div>
                                    )}
                                </div>
                                <span className="message-timestamp">
                                    {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                </span>
                            </div>
                        );
                    })}
                    <div ref={messagesEndRef} />
                </div>

                {isProcessing && <div className="loading-indicator"><span>{isRagLoading ? 'Searching documents...' : 'Thinking...'}</span></div>}
                {!isProcessing && error && <div className="error-indicator">{error}</div>}

                <footer className="input-area">
                    <TextField
                        fullWidth
                        variant="outlined"
                        size="small"
                        value={inputText}
                        onChange={(e) => setInputText(e.target.value)}
                        onKeyDown={handleEnterKey}
                        placeholder="Ask your tutor..."
                        disabled={isProcessing}
                        aria-label="Chat input"
                        multiline
                        maxRows={5}
                        InputProps={{
                            endAdornment: (
                                <IconButton onClick={handleSendMessage} disabled={isProcessing || !inputText.trim()} title="Send Message" aria-label="Send message">
                                    <SendIcon />
                                </IconButton>
                            )
                        }}
                    />
                    <Tooltip title={!hasFiles ? "Upload files to enable RAG" : (isRagEnabled ? "Disable RAG (Retrieval-Augmented Generation)" : "Enable RAG (Retrieval-Augmented Generation)")}>
                        <div className="rag-toggle-container"> {/* For proper layout with label */}
                            <input type="checkbox" id="rag-toggle" checked={isRagEnabled} onChange={handleRagToggle}
                                disabled={!hasFiles || isProcessing} aria-label="Enable RAG" />
                            <label htmlFor="rag-toggle" style={{ marginLeft: '4px', cursor: (!hasFiles || isProcessing) ? 'not-allowed' : 'pointer' }}>RAG</label>
                        </div>
                    </Tooltip>
                </footer>
            </div>

            <HistoryModal isOpen={isHistoryModalOpen} onClose={closeHistoryModal} />

            <div className="vertical-icon-bar" role="navigation" aria-label="Main navigation icons">
                {[
                    { label: 'FAQ', type: 'faq', icon: <HelpOutlineIcon style={{ fontSize: 28 }} />, action: () => fetchAnalysisData(selectedAnalysisFile, 'faq') },
                    { label: 'Topics', type: 'topics', icon: <ListIcon style={{ fontSize: 28 }} />, action: () => fetchAnalysisData(selectedAnalysisFile, 'topics') },
                    { label: 'Mindmap', type: 'mindmap', icon: <BubbleChartIcon style={{ fontSize: 28 }} />, action: () => fetchAnalysisData(selectedAnalysisFile, 'mindmap') },
                    { label: 'Knowledge Graph', type: 'knowledgeGraph', icon: <MapIcon style={{ fontSize: 28 }} />, action: () => fetchKgData(selectedAnalysisFile) }
                ].map(item => (
                    <Tooltip title={item.label} key={item.type}>
                        <IconButton
                            className={`icon-button ${activeSidebar === item.type ? 'active' : ''}`}
                            aria-label={item.label}
                            onClick={() => {
                                toggleSidebar(item.type);
                                if (selectedAnalysisFile && (activeSidebar !== item.type || !analysisData && !kgData)) { // Fetch if opening or if data is not loaded for this file
                                    item.action();
                                } else if (!selectedAnalysisFile) {
                                     setSnackbarMessage(`Please select a file from 'Your Uploaded Files' to generate ${item.label}.`);
                                     setSnackbarSeverity('warning'); setOpenSnackbar(true);
                                }
                            }}
                        >
                            {item.icon}
                        </IconButton>
                    </Tooltip>
                ))}
            </div>

            {activeSidebar && (
                <div className="right-sidebar">
                    <div className="sidebar-header">
                        <Typography variant="h6" component="span">{activeSidebar.toUpperCase()}</Typography>
                        <IconButton onClick={() => setActiveSidebar(null)}><CloseIcon /></IconButton>
                    </div>
                    <div className="sidebar-content">
                        {activeSidebar === 'faq' && <FAQAnalysis analysisData={analysisData} isLoading={analysisLoading && selectedAnalysisFile === selectedAnalysisFile} error={analysisError} />}
                        {activeSidebar === 'topics' && <TopicAnalysis analysisData={analysisData} isLoading={analysisLoading && selectedAnalysisFile === selectedAnalysisFile} error={analysisError} />}
                        {activeSidebar === 'mindmap' && <MindmapAnalysis analysisData={analysisData} isLoading={analysisLoading && selectedAnalysisFile === selectedAnalysisFile} error={analysisError} />}
                        {activeSidebar === 'knowledgeGraph' && <KnowledgeGraphViewer kgData={kgData} isLoading={kgLoading && selectedAnalysisFile === selectedAnalysisFile} error={kgError} />}
                    </div>
                </div>
            )}
            <Snackbar open={openSnackbar} autoHideDuration={6000} onClose={() => setOpenSnackbar(false)} anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}>
                <Alert onClose={() => setOpenSnackbar(false)} severity={snackbarSeverity} sx={{ width: '100%' }}>
                    {snackbarMessage}
                </Alert>
            </Snackbar>
        </div>
    );
};

export default ChatPage;
//     // client/src/components/ChatPage.js
//     import React, { useState, useEffect, useRef, useCallback } from 'react';
//     import { useNavigate } from 'react-router-dom';
// import { sendMessage, saveChatHistory, getUserFiles, queryRagService, getAnalysis, getKgDataForVisualization } from '../services/api';
//     import ReactMarkdown from 'react-markdown';
//     import remarkGfm from 'remark-gfm';
// import { v4 as uuidv4 } from 'uuid'; // Ensure this is imported
    
//     import SystemPromptWidget, { availablePrompts, getPromptTextById } from './SystemPromptWidget';
//     import HistoryModal from './HistoryModal';
//     import FileUploadWidget from './FileUploadWidget';
//     import FileManagerWidget from './FileManagerWidget';
// import ChatHistory from './ChatHistory'; // Assuming this is used somewhere, else remove
//     import FAQAnalysis from './FAQAnalysis';
//     import TopicAnalysis from './TopicAnalysis';
//     import MindmapAnalysis from './MindmapAnalysis';
// import KnowledgeGraphViewer from './KnowledgeGraphViewer'; // Assuming this is used, else remove
    
//     import './ChatPage.css';
//     import { FormControl, InputLabel, Select, MenuItem, Button, Box, TextField, Typography, CircularProgress, IconButton, AppBar, Toolbar, Drawer, List, ListItem, ListItemText, Divider, Snackbar, Alert, Tooltip, Dialog, DialogTitle, DialogContent, DialogActions } from '@mui/material';
// // Assuming icons are used, keep them
//     import SendIcon from '@mui/icons-material/Send';
//     import MenuIcon from '@mui/icons-material/Menu';
//     import ChatBubbleOutlineIcon from '@mui/icons-material/ChatBubbleOutline';
//     import HistoryIcon from '@mui/icons-material/History';
//     import LogoutIcon from '@mui/icons-material/Logout';
//     import AddCircleOutlineIcon from '@mui/icons-material/AddCircleOutline';
//     import UploadFileIcon from '@mui/icons-material/UploadFile';
//     import FolderOpenIcon from '@mui/icons-material/FolderOpen';
//     import FunctionsIcon from '@mui/icons-material/Functions';
//     import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
//     import LightbulbOutlinedIcon from '@mui/icons-material/LightbulbOutlined';
//     import AccountCircleIcon from '@mui/icons-material/AccountCircle';
//     import CloseIcon from '@mui/icons-material/Close';
//     import ListIcon from '@mui/icons-material/List';
//     import InsightsIcon from '@mui/icons-material/Insights';
//     import BubbleChartIcon from '@mui/icons-material/BubbleChart';
//     import MapIcon from '@mui/icons-material/Map';
    
//     const ChatPage = ({ setIsAuthenticated }) => {
//         const [messages, setMessages] = useState([]);
//         const [inputText, setInputText] = useState('');
//         const [isLoading, setIsLoading] = useState(false);
//         const [isRagLoading, setIsRagLoading] = useState(false);
//         const [error, setError] = useState('');
//     const [sessionId, setSessionId] = useState('');
//     const [userId, setUserId] = useState('');
//     const [username, setUsername] = useState('');
//         const [currentSystemPromptId, setCurrentSystemPromptId] = useState('friendly');
//         const [editableSystemPromptText, setEditableSystemPromptText] = useState(() => getPromptTextById('friendly'));
//         const [isHistoryModalOpen, setIsHistoryModalOpen] = useState(false);
//         const [fileRefreshTrigger, setFileRefreshTrigger] = useState(0);
//         const [hasFiles, setHasFiles] = useState(false);
//         const [isRagEnabled, setIsRagEnabled] = useState(false);
//     const [activeSidebar, setActiveSidebar] = useState(null);
//     const [drawerOpen, setDrawerOpen] = useState(false); // If not used, can be removed
//         const [openSnackbar, setOpenSnackbar] = useState(false);
//         const [snackbarMessage, setSnackbarMessage] = useState('');
//     const [snackbarSeverity, setSnackbarSeverity] = useState('info');
//     const [selectedLlm, setSelectedLlm] = useState('gemini');
    
//         const [analysisData, setAnalysisData] = useState(null);
//         const [analysisLoading, setAnalysisLoading] = useState(false);
//         const [analysisError, setAnalysisError] = useState(null);
//     const [selectedAnalysisFile, setSelectedAnalysisFile] = useState(null);
    
//         const [kgData, setKgData] = useState(null);
//         const [kgLoading, setKgLoading] = useState(false);
//         const [kgError, setKgError] = useState(null);
    
//         const messagesEndRef = useRef(null);
//         const navigate = useNavigate();
    
//         const toggleSidebar = (type) => {
//         setActiveSidebar(prev => (prev === type ? null : type));
//     };
    
//         useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);
    
//     // Centralized logout logic
//     const performLogoutCleanup = useCallback(() => {
//         console.log("Performing logout cleanup...");
//         localStorage.clear();
//         setIsAuthenticated(false);
//         setMessages([]); setSessionId(''); setUserId(''); setUsername('');
//         setCurrentSystemPromptId('friendly');
//         setEditableSystemPromptText(getPromptTextById('friendly')); setError('');
//         setHasFiles(false); setIsRagEnabled(false);
//         setActiveSidebar(null); setAnalysisData(null); setSelectedAnalysisFile(null); setKgData(null);
//         // Ensure navigation happens after state updates
//         requestAnimationFrame(() => {
//             if (window.location.pathname !== '/login') {
//                 navigate('/login', { replace: true });
//             }
//         });
//     }, [navigate, setIsAuthenticated]); // Added setIsAuthenticated

//     // Validate auth info on mount and establish sessionId
//         useEffect(() => {
//         const storedToken = localStorage.getItem('token');
//             const storedUserId = localStorage.getItem('userId');
//             const storedUsername = localStorage.getItem('username');
//         let sessionIdFromStorage = localStorage.getItem('sessionId');

//         if (!storedUserId || !storedToken || !storedUsername) {
//             console.warn("ChatPage Mount: Missing essential auth info. Performing cleanup and redirecting.");
//             performLogoutCleanup(); // Use centralized cleanup
//             } else {
//             console.log("ChatPage Mount: Auth info found. Setting user state.");
//                 setUserId(storedUserId);
//                 setUsername(storedUsername);

//             if (sessionIdFromStorage) {
//                 setSessionId(sessionIdFromStorage);
//                 console.log("ChatPage Mount: Using existing sessionId from localStorage:", sessionIdFromStorage);
//                     } else {
//                 const newClientSessionId = uuidv4();
//                 localStorage.setItem('sessionId', newClientSessionId);
//                 setSessionId(newClientSessionId);
//                 console.log("ChatPage Mount: No sessionId in localStorage. Generated new client-side sessionId:", newClientSessionId);
//             }
//             }
//         // eslint-disable-next-line react-hooks/exhaustive-deps
//     }, [performLogoutCleanup]); // Removed navigate and setIsAuthenticated as they are in performLogoutCleanup
    

//         const handlePromptSelectChange = useCallback((newId) => {
//             setCurrentSystemPromptId(newId); setEditableSystemPromptText(getPromptTextById(newId));
//             setError(prev => prev && (prev.includes("Session invalid") || prev.includes("Critical Error")) ? prev : `Assistant mode changed.`);
//             setTimeout(() => { setError(prev => prev === `Assistant mode changed.` ? '' : prev); }, 3000);
//         }, []);

    
//         const saveAndReset = useCallback(async (isLoggingOut = false, onCompleteCallback = null) => {
//         const currentSessionId = localStorage.getItem('sessionId');
//         const currentUserId = localStorage.getItem('userId');
//             const messagesToSave = [...messages];
    
//             if (!currentSessionId || !currentUserId) {
//              console.error("Save Error: Session ID or User ID missing from localStorage.");
//              setError("Critical Error: Session info missing. Please re-login.");
//              if (isLoggingOut) performLogoutCleanup(); // Force logout if critical info is missing
//                  if (onCompleteCallback) onCompleteCallback();
//                  return;
//             }
//             if (isLoading || isRagLoading || messagesToSave.length === 0) {
//                  if (onCompleteCallback) onCompleteCallback();
//                  return;
//             }
    
//             let newSessionId = null;
//         setIsLoading(true); // Consider a specific 'isSaving' state if needed
//             setError(prev => prev && (prev.includes("Session invalid") || prev.includes("Critical Error")) ? prev : '');
    
//             try {
//                 console.log(`Saving history for session: ${currentSessionId} (User: ${currentUserId})`);
//                 const response = await saveChatHistory({ sessionId: currentSessionId, messages: messagesToSave });
//                 newSessionId = response.data.newSessionId;
//                 if (!newSessionId) throw new Error("Backend failed to provide new session ID.");
    
//                 localStorage.setItem('sessionId', newSessionId);
//             setSessionId(newSessionId); // Update state
//                 setMessages([]);
//                 if (!isLoggingOut) {
//                 handlePromptSelectChange('friendly'); // Reset prompt
//                     setError('');
//                 }
//                 console.log(`History saved. New session ID: ${newSessionId}`);
    
//             } catch (err) {
//                 const failErrorMsg = err.response?.data?.message || err.message || 'Failed to save/reset session.';
//                 console.error("Save/Reset Error:", err.response || err);
//                 setError(`Session Error: ${failErrorMsg}`);
//             if (err.response?.status === 401) {
//                      console.warn("Received 401 saving history, logging out.");
//                  performLogoutCleanup();
//                      // Don't proceed with client-side session generation if auth failed
//             } else if (!newSessionId && !isLoggingOut) { // If save failed and not logging out, create client-side ID
//                      newSessionId = uuidv4();
//                      localStorage.setItem('sessionId', newSessionId);
//                  setSessionId(newSessionId); // Update state
//                      setMessages([]);
//                      handlePromptSelectChange('friendly');
//                      console.warn("Save failed, generated new client-side session ID:", newSessionId);
//                 } else if (isLoggingOut && !newSessionId) {
//                  console.error("Save failed during logout, but proceeding with logout.");
//                 }
//             } finally {
//                 setIsLoading(false);
//                 if (onCompleteCallback) onCompleteCallback();
//             }
//     }, [messages, isLoading, isRagLoading, handlePromptSelectChange, performLogoutCleanup]); // Added performLogoutCleanup

    
//         const handleLogout = useCallback((skipSave = false) => {
//         if (!skipSave && messages.length > 0) {
//              saveAndReset(true, performLogoutCleanup);
//         } else {
//              performLogoutCleanup();
//         }
//     }, [messages.length, saveAndReset, performLogoutCleanup]);

//     // Check for user files on mount and refresh, only if userId state is set
//     useEffect(() => {
//         const checkUserFiles = async () => {
//             if (!userId) { // Check against the component's userId STATE
//                 console.log("User files check skipped: No userId state available yet.");
//                 setHasFiles(false); setIsRagEnabled(false);
//                 return;
//             }
//             console.log("Checking user files for userId (from state):", userId);
//             try {
//                 const response = await getUserFiles();
//                 const filesExist = response.data && response.data.length > 0;
//                 setHasFiles(filesExist);
//                 setIsRagEnabled(filesExist);
//                 console.log("User files check successful:", filesExist ? `${response.data.length} files found. RAG default: ${filesExist}` : "No files found. RAG default: false");
//             } catch (err) {
//                 console.error("Error checking user files:", err);
//                 if (err.response?.status === 401 && !window.location.pathname.includes('/login')) {
//                      console.warn("Received 401 checking files, logging out.");
//                      handleLogout(true); // Use the main logout handler
//             } else {
//                     setError("Could not check user files.");
//                     setHasFiles(false); setIsRagEnabled(false);
//                 }
//             }
//         };
//         if (userId) { // This ensures it runs after userId state is populated
//             checkUserFiles();
//             }
//         // eslint-disable-next-line react-hooks/exhaustive-deps
//     }, [userId, fileRefreshTrigger, handleLogout]); // Added handleLogout for stability


//     const triggerFileRefresh = useCallback(() => setFileRefreshTrigger(prev => prev + 1), []);
//     const handlePromptTextChange = useCallback((newText) => {
//         setEditableSystemPromptText(newText);
//         const matchingPreset = availablePrompts.find(p => p.id !== 'custom' && p.prompt === newText);
//         setCurrentSystemPromptId(matchingPreset ? matchingPreset.id : 'custom');
//     }, []);
//     const handleHistory = useCallback(() => setIsHistoryModalOpen(true), []);
//     const closeHistoryModal = useCallback(() => setIsHistoryModalOpen(false), []);
//     const handleLlmChange = useCallback((event) => {
//         setSelectedLlm(event.target.value);
//         setSnackbarMessage(`LLM changed to: ${event.target.value}`);
//         setSnackbarSeverity('info');
//         setOpenSnackbar(true);
//     }, []);

//     const handleFileSelectForAnalysis = useCallback((filename) => {
//         setSelectedAnalysisFile(filename);
//         if (activeSidebar === 'faq') { // Changed from 'analysis' to 'faq' for consistency if that's the intent
//             fetchAnalysisData(filename, 'faq'); // Pass type
//         } else if (activeSidebar === 'topics') {
//             fetchAnalysisData(filename, 'topics'); // Pass type
//         } else if (activeSidebar === 'mindmap') {
//             fetchAnalysisData(filename, 'mindmap'); // Pass type
//         } else if (activeSidebar === 'knowledgeGraph') { // Assuming KG is a separate sidebar type
//             fetchKgData(filename);
//         }
//         setSnackbarMessage(`File selected for analysis/view: ${filename}`);
//         setSnackbarSeverity('info');
//         setOpenSnackbar(true);
//         // Default to FAQ if no specific analysis sidebar is open or to trigger initial load
//         if (!activeSidebar && (filename.endsWith('.txt') || filename.endsWith('.pdf'))) { // Example condition
//             setActiveSidebar('faq');
//             fetchAnalysisData(filename, 'faq');
//         }
//     }, [activeSidebar]); // Removed fetchAnalysisData, fetchKgData from deps, they are stable
    
//         const handleNewChat = useCallback(() => {
//             if (!isLoading && !isRagLoading) {
//                  saveAndReset(false);
//             }
//          }, [isLoading, isRagLoading, saveAndReset]);
    
//         const handleSendMessage = useCallback(async (e) => {
//             if (e) e.preventDefault();
//             const textToSend = inputText.trim();
//         // Get fresh IDs directly from localStorage at the point of sending
//         const currentStoredSessionId = localStorage.getItem('sessionId');
//         const currentStoredUserId = localStorage.getItem('userId');

//         if (!textToSend || isLoading || isRagLoading) {
//             return;
//         }

//         if (!currentStoredSessionId || !currentStoredUserId) {
//              console.error("SendMessage Error: SessionId or UserId missing from localStorage.", {currentStoredSessionId, currentStoredUserId});
//                      setError("Session invalid. Please refresh or log in again.");
//              // Optionally trigger logout if auth info is critically missing
//              if (!currentStoredUserId) {
//                 console.warn("SendMessage: UserId missing, forcing logout.");
//                 handleLogout(true); // Force logout
//                 }
//                 return;
//             }
    
//             const newUserMessage = { role: 'user', parts: [{ text: textToSend }], timestamp: new Date() };
//         const previousMessages = messages; // Keep a copy for potential rollback
//             setMessages(prev => [...prev, newUserMessage]);
//             setInputText('');
//             setError('');
    
//             let relevantDocs = [];
//             let ragError = null;
    
//             if (isRagEnabled) {
//                 setIsRagLoading(true);
//                 try {
//                     console.log("RAG Enabled: Querying backend /rag endpoint...");
//                 const ragResponse = await queryRagService({ message: textToSend }); // Interceptor adds user ID
//                     relevantDocs = ragResponse.data.relevantDocs || [];
//                     console.log(`RAG Query returned ${relevantDocs.length} documents.`);
//                 } catch (err) {
//                     console.error("RAG Query Error:", err.response || err);
//                     ragError = err.response?.data?.message || "Failed to retrieve documents for RAG.";
//                     if (err.response?.status === 401) {
//                          console.warn("Received 401 during RAG query, logging out.");
//                          handleLogout(true);
//                      setIsRagLoading(false); return;
//                     }
//                 } finally {
//                     setIsRagLoading(false);
//                 }
//             } else {
//                 console.log("RAG Disabled: Skipping RAG query.");
//             }
    
//             setIsLoading(true);
//         // const historyForAPI = previousMessages; // Send messages state before new user message
//             const systemPromptToSend = editableSystemPromptText;
    
//             try {
//                 if (ragError) {
//                      setError(prev => prev ? `${prev} | RAG Error: ${ragError}` : `RAG Error: ${ragError}`);
//                 }
    
//             console.log(`Sending message to backend /message. RAG Enabled: ${isRagEnabled}, Docs Found: ${relevantDocs.length}, SessionID: ${currentStoredSessionId}`);
//                 const sendMessageResponse = await sendMessage({
//                     message: textToSend,
//                 history: previousMessages, // Send the history *before* the current user message
//                 sessionId: currentStoredSessionId, // Use the one from localStorage
//                     systemPrompt: systemPromptToSend,
//                     isRagEnabled: isRagEnabled,
//                     relevantDocs: relevantDocs,
//                     llm_preference: selectedLlm,
//                 });
    
//                 const modelReply = sendMessageResponse.data.reply;
//                 if (modelReply?.role && modelReply?.parts?.length > 0) {
//                 setMessages(prev => [...prev, modelReply]); // Add model reply to current messages
//                 } else {
//                     throw new Error("Invalid reply structure received from backend.");
//                 }
//                 setError(prev => prev && (prev.includes("Session invalid") || prev.includes("Critical Error")) ? prev : '');
    
//             } catch (err) {
//                 const errorMessage = err.response?.data?.message || err.message || 'Failed to get response.';
//                 setError(prev => prev ? `${prev} | Chat Error: ${errorMessage}` : `Chat Error: ${errorMessage}`);
//                 console.error("Send Message Error:", err.response || err);
//             setMessages(previousMessages); // Rollback UI to state before user sent message
//                 if (err.response?.status === 401 && !window.location.pathname.includes('/login')) {
//                      console.warn("Received 401 sending message, logging out.");
//                      handleLogout(true);
//                 }
//             } finally {
//                 setIsLoading(false);
//             }
//     }, [inputText, isLoading, isRagLoading, messages, editableSystemPromptText, isRagEnabled, handleLogout, selectedLlm]);
    
//         const handleEnterKey = useCallback((e) => {
//             if (e.key === 'Enter' && !e.shiftKey) {
//                 e.preventDefault();
//                 handleSendMessage();
//             }
//         }, [handleSendMessage]);
    
//         const handleRagToggle = (event) => {
//             setIsRagEnabled(event.target.checked);
//         };
    
//         const isProcessing = isLoading || isRagLoading;
    
//     const fetchAnalysisData = async (filename, analysisType) => { // Added analysisType
//         if (!filename || !analysisType) return;
//             setAnalysisLoading(true);
//             setAnalysisError(null);
//             try {
//             // Make sure getAnalysis API expects analysisType
//             const response = await getAnalysis(filename, analysisType, selectedLlm);
//             setAnalysisData(response.data);
//             } catch (err) {
//             console.error(`Error fetching ${analysisType} analysis:`, err);
//             setAnalysisError(err.message || `Failed to fetch ${analysisType} analysis`);
//             setAnalysisData(null); // Clear data on error
//             } finally {
//                 setAnalysisLoading(false);
//             }
//         };
    
//         const fetchKgData = async (filename) => {
//         if (!filename) return;
//             setKgLoading(true);
//             setKgError(null);
//             try {
//             const response = await getKgDataForVisualization(filename);
//             setKgData(response.data);
//             } catch (err) {
//             console.error('Error fetching KG data:', err);
//             setKgError(err.message || 'Failed to fetch knowledge graph data');
//             setKgData(null); // Clear data on error
//             } finally {
//                 setKgLoading(false);
//             }
//         };
    
//     // Render loading or null if userId isn't set yet (critical for initial setup)
//     if (!userId) { // Check against component's state
//         return <div className="loading-indicator"><span>Initializing authentication...</span></div>;
//         }
    
//         return (
//         <div className="chat-page-container">
//                 <div className="sidebar-area">
//                     <SystemPromptWidget
//                         selectedPromptId={currentSystemPromptId} promptText={editableSystemPromptText}
//                         onSelectChange={handlePromptSelectChange} onTextChange={handlePromptTextChange} />
//                     <FileUploadWidget onUploadSuccess={triggerFileRefresh} />
//                     <FileManagerWidget 
//                         refreshTrigger={fileRefreshTrigger} 
//                         onFileSelect={handleFileSelectForAnalysis}
//                     />
//                 </div>
    
//                 <div className={`chat-container ${activeSidebar ? 'with-sidebar' : ''}`}>
//                     <header className="chat-header">
//                         <h1>Engineering Tutor</h1>
//                         <div className="header-controls">
//                             <span className="username-display">Hi, {username}!</span>
//                             <FormControl variant="outlined" size="small" sx={{ m: 1, minWidth: 120 }}>
//                                 <InputLabel id="llm-select-label">LLM</InputLabel>
//                                 <Select
//                                 labelId="llm-select-label" id="llm-select"
//                                 value={selectedLlm} label="LLM"
//                                 onChange={handleLlmChange} disabled={isProcessing}
//                                 >
//                                     <MenuItem value="gemini">Gemini</MenuItem>
//                                     <MenuItem value="ollama_deepseek">Ollama (DeepSeek)</MenuItem>
//                                     <MenuItem value="ollama_llama3">Ollama (Llama 3)</MenuItem>
//                                 </Select>
//                             </FormControl>
//                             <button onClick={handleHistory} className="header-button history-button" disabled={isProcessing}>History</button>
//                             <button onClick={handleNewChat} className="header-button newchat-button" disabled={isProcessing}>New Chat</button>
//                             <button onClick={() => handleLogout(false)} className="header-button logout-button" disabled={isProcessing}>Logout</button>
//                         </div>
//                     </header>
    
//                     <button
//                         className="header-button export-pdf-button"
//                     disabled={isProcessing || messages.length === 0} // Disable if no messages
//                         onClick={async () => {
//                         const currentPdfSessionId = localStorage.getItem('sessionId'); // Use current session for export
//                         if (!currentPdfSessionId) {
//                             alert('Session ID is missing, cannot export PDF.');
//                             return;
//                         }
//                             try {
//                             // Ensure your backend /export_chat_pdf uses sessionId from query param or header
//                             const response = await fetch(`/export_chat_pdf?sessionId=${currentPdfSessionId}&type=both`);
//                             if (!response.ok) throw new Error(`Failed to download PDF: ${response.statusText}`);
//                                 const blob = await response.blob();
//                                 const url = window.URL.createObjectURL(blob);
//                                 const a = document.createElement('a');
//                                 a.href = url;
//                             a.download = `chat_export_${currentPdfSessionId}.pdf`;
//                                 document.body.appendChild(a);
//                                 a.click();
//                                 a.remove();
//                                 window.URL.revokeObjectURL(url);
//                             } catch (error) {
//                                 console.error('Error downloading chat PDF:', error);
//                             alert(`Failed to download chat PDF: ${error.message}`);
//                             }
//                     }}
//                         title="Download Chat as PDF"
//                     >
//                         Download Chat as PDF
//                     </button>
    
//                     <div className="messages-area">
//                         {messages.map((msg, index) => {
//                             if (!msg?.role || !msg?.parts?.length || !msg.timestamp) {
//                                 console.warn("Rendering invalid message structure at index", index, msg);
//                                 return <div key={`error-${index}`} className="message-error">Msg Error</div>;
//                             }
//                             const messageText = msg.parts[0]?.text || '';
//                             return (
//                             <div key={`${sessionId}-${index}`} className={`message ${msg.role}`}> {/* Use sessionId state */}
//                                     <div className="message-content">
//                                         <ReactMarkdown remarkPlugins={[remarkGfm]}>
//                                             {messageText}
//                                         </ReactMarkdown>
//                                     </div>
//                                     <span className="message-timestamp">
//                                         {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
//                                     </span>
//                                 </div>
//                             );
//                         })}
//                         <div ref={messagesEndRef} />
//                     </div>
    
//                     {isProcessing && <div className="loading-indicator"><span>{isRagLoading ? 'Searching documents...' : 'Thinking...'}</span></div>}
//                     {!isProcessing && error && <div className="error-indicator">{error}</div>}
    
//                     <footer className="input-area">
//                         <textarea
//                             value={inputText} onChange={(e) => setInputText(e.target.value)} onKeyDown={handleEnterKey}
//                             placeholder="Ask your tutor..." rows="1" disabled={isProcessing} aria-label="Chat input" />
//                     <div className="rag-toggle-container" title={!hasFiles ? "Upload files to enable RAG" : (isRagEnabled ? "Disable RAG" : "Enable RAG")}>
//                             <input type="checkbox" id="rag-toggle" checked={isRagEnabled} onChange={handleRagToggle}
//                                 disabled={!hasFiles || isProcessing} aria-label="Enable RAG" />
//                             <label htmlFor="rag-toggle">RAG</label>
//                         </div>
//                         <button onClick={handleSendMessage} disabled={isProcessing || !inputText.trim()} title="Send Message" aria-label="Send message">
//                             <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="20" height="20">
//                                 <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
//                             </svg>
//                         </button>
//                     </footer>
//                 </div>
    
//                 <HistoryModal isOpen={isHistoryModalOpen} onClose={closeHistoryModal} />
    
//                 <div className="vertical-icon-bar" role="navigation" aria-label="Main navigation icons">
//                 <button className="icon-button" aria-label="FAQ" title="FAQ" onClick={() => { toggleSidebar('faq'); if (selectedAnalysisFile) fetchAnalysisData(selectedAnalysisFile, 'faq'); }}>
//                     <HelpOutlineIcon style={{ fontSize: 28 }} />
//                     </button>
//                 <button className="icon-button" aria-label="Topics" title="Topics" onClick={() => { toggleSidebar('topics'); if (selectedAnalysisFile) fetchAnalysisData(selectedAnalysisFile, 'topics'); }}>
//                     <ListIcon style={{ fontSize: 28 }} />
//                     </button>
//                 <button className="icon-button" aria-label="Mindmap" title="Mindmap" onClick={() => { toggleSidebar('mindmap'); if (selectedAnalysisFile) fetchAnalysisData(selectedAnalysisFile, 'mindmap'); }}>
//                     <BubbleChartIcon style={{ fontSize: 28 }} />
//                 </button>
//                  <button className="icon-button" aria-label="Knowledge Graph" title="Knowledge Graph" onClick={() => { toggleSidebar('knowledgeGraph'); if (selectedAnalysisFile) fetchKgData(selectedAnalysisFile); }}>
//                     <MapIcon style={{ fontSize: 28 }} /> {/* Example icon for KG */}
//                     </button>
//                 </div>

//                 {activeSidebar && (
//         <div className="right-sidebar">
//             <div className="sidebar-header">
//                 <span>{activeSidebar.toUpperCase()}</span>
//                 <button className="close-button" onClick={() => setActiveSidebar(null)}>✖</button>
//             </div>
//             <div className="sidebar-content">
//                 {activeSidebar === 'faq' && (
//                     <FAQAnalysis 
//                                 analysisData={analysisData} // Pass full data, component will pick what it needs
//                                 isLoading={analysisLoading && selectedAnalysisFile} // Only loading if a file is selected for this type
//                         error={analysisError}
//                     />
//                 )}
//                 {activeSidebar === 'topics' && (
//                     <TopicAnalysis
//                                 analysisData={analysisData}
//                                 isLoading={analysisLoading && selectedAnalysisFile}
//                         error={analysisError}
//                     />
//                 )}
//                 {activeSidebar === 'mindmap' && (
//                     <MindmapAnalysis
//                                 analysisData={analysisData}
//                                 isLoading={analysisLoading && selectedAnalysisFile}
//                         error={analysisError}
//                     />
//                 )}
//                         {activeSidebar === 'knowledgeGraph' && (
//                             <KnowledgeGraphViewer
//                                 kgData={kgData}
//                                 isLoading={kgLoading && selectedAnalysisFile}
//                                 error={kgError}
//                             />
//                         )}
//             </div>
//         </div>
//     )}
//              <Snackbar open={openSnackbar} autoHideDuration={4000} onClose={() => setOpenSnackbar(false)}>
//                 <Alert onClose={() => setOpenSnackbar(false)} severity={snackbarSeverity} sx={{ width: '100%' }}>
//                     {snackbarMessage}
//                 </Alert>
//             </Snackbar>
//             </div>
//         );
//     };
    
//     export default ChatPage;