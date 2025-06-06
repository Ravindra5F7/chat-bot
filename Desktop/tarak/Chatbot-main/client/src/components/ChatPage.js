// client/src/components/ChatPage.js
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
    sendMessage,
    saveChatHistory,
    getUserFiles
} from '../services/api';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { v4 as uuidv4 } from 'uuid';
import { LLM_OPTIONS } from '../config/constants'; // <<<--- IMPORTED FROM SHARED CONSTANTS

import SystemPromptWidget, { availablePrompts, getPromptTextById } from './SystemPromptWidget';
import HistoryModal from './HistoryModal';
import FileUploadWidget from './FileUploadWidget';
import FileManagerWidget from './FileManagerWidget';
import AnalysisResultModal from './AnalysisResultModal';

import './ChatPage.css';

// const LLM_OPTIONS = { ... }; // <<<--- LOCAL DEFINITION REMOVED

const ChatPage = ({ setIsAuthenticated }) => {
    const [messages, setMessages] = useState([]);
    const [inputText, setInputText] = useState('');
    const [isLoading, setIsLoading] = useState(false);
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

    const defaultProvider = 'groq_llama3';
    const [llmProvider, setLlmProvider] = useState(defaultProvider);
    const [llmModelName, setLlmModelName] = useState(
        LLM_OPTIONS[defaultProvider]?.models[0] || ''
    );
    const [enableMultiQuery, setEnableMultiQuery] = useState(true);

    // --- State for Analysis Modal ---
    const [isAnalysisModalOpen, setIsAnalysisModalOpen] = useState(false);
    const [analysisModalData, setAnalysisModalData] = useState(null);

    const messagesEndRef = useRef(null);
    const navigate = useNavigate();

    const performLogoutCleanup = useCallback(() => {
        localStorage.clear();
        setIsAuthenticated(false);
        setMessages([]); setSessionId(''); setUsername(''); setUserId('');
        setError(''); setHasFiles(false); setIsRagEnabled(false);
        setAnalysisModalData(null); setIsAnalysisModalOpen(false);
        requestAnimationFrame(() => {
            if (window.location.pathname !== '/login') navigate('/login', { replace: true });
        });
    }, [setIsAuthenticated, navigate]);


    useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

    useEffect(() => {
        const storedSessionId = localStorage.getItem('sessionId') || uuidv4();
        const storedUserId = localStorage.getItem('userId');
        const storedUsername = localStorage.getItem('username');

        if (!storedUserId || !storedUsername) {
            console.warn("ChatPage Mount: Missing auth info. Performing cleanup and redirect.");
            performLogoutCleanup();
        } else {
            setSessionId(storedSessionId);
            if (!localStorage.getItem('sessionId')) {
                localStorage.setItem('sessionId', storedSessionId);
            }
            setUserId(storedUserId);
            setUsername(storedUsername);
        }
    }, [performLogoutCleanup]);


    const handleLogout = useCallback((skipSave = false) => {
        const cleanupAndRedirect = () => performLogoutCleanup();

        if (!skipSave && messages.length > 0 && !isLoading) {
            saveAndReset(true, cleanupAndRedirect);
        } else {
            cleanupAndRedirect();
        }
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [messages.length, performLogoutCleanup, isLoading /*, saveAndReset */ ]);


    useEffect(() => {
        const checkUserFiles = async () => {
            const currentUserIdFromStorage = localStorage.getItem('userId');
            if (!currentUserIdFromStorage) {
                setHasFiles(false); setIsRagEnabled(false); return;
            }
            try {
                const response = await getUserFiles();
                const filesExist = response.data && response.data.length > 0;
                setHasFiles(filesExist);
                if (filesExist && messages.length === 0) {
                    setIsRagEnabled(true);
                } else if (!filesExist) {
                    setIsRagEnabled(false);
                }
            } catch (err) {
                console.error("Error checking user files:", err);
                setError(err.response?.data?.message || "Could not check user files.");
                setHasFiles(false); setIsRagEnabled(false);
            }
        };
        const currentUserIdFromStorage = localStorage.getItem('userId');
        if (currentUserIdFromStorage) checkUserFiles();
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [userId, fileRefreshTrigger, messages.length]);


    const triggerFileRefresh = useCallback(() => setFileRefreshTrigger(prev => prev + 1), []);

    const handlePromptSelectChange = useCallback((newId) => {
        setCurrentSystemPromptId(newId); setEditableSystemPromptText(getPromptTextById(newId));
        setError(prev => (prev && (prev.includes("Session invalid") || prev.includes("Critical Error"))) ? prev : `Assistant mode changed.`);
        setTimeout(() => { setError(prev => prev === `Assistant mode changed.` ? '' : prev); }, 3000);
    }, []);

    const handlePromptTextChange = useCallback((newText) => {
        setEditableSystemPromptText(newText);
        const matchingPreset = availablePrompts.find(p => p.id !== 'custom' && p.prompt === newText);
        setCurrentSystemPromptId(matchingPreset ? matchingPreset.id : 'custom');
    }, []);

    const handleHistory = useCallback(() => setIsHistoryModalOpen(true), []);
    const closeHistoryModal = useCallback(() => setIsHistoryModalOpen(false), []);

    const saveAndReset = useCallback(async (isLoggingOut = false, onCompleteCallback = null) => {
        const currentSessionIdToSave = localStorage.getItem('sessionId');
        const currentUserIdToSave = localStorage.getItem('userId');
        const messagesToSave = messages.map(m => ({ role: m.role, parts: m.parts, timestamp: m.timestamp }));

        if (!currentSessionIdToSave || !currentUserIdToSave || messagesToSave.length === 0 || isLoading) {
            if (onCompleteCallback) onCompleteCallback();
            return;
        }

        setIsLoading(true);
        setError(prev => (prev && (prev.includes("Session invalid") || prev.includes("Critical Error"))) ? prev : '');

        try {
            const response = await saveChatHistory({ sessionId: currentSessionIdToSave, messages: messagesToSave });
            const newSessionIdForStorage = response.data.newSessionId;

            if (!newSessionIdForStorage) {
                console.error("Backend failed to provide new session ID for storage. Using UUID.");
                throw new Error("Backend failed to provide new session ID for storage.");
            }

            localStorage.setItem('sessionId', newSessionIdForStorage);
            setSessionId(newSessionIdForStorage);
            setMessages([]);
            if (!isLoggingOut) {
                handlePromptSelectChange('friendly');
                setError('');
            }
            console.log(`History saved for ${currentSessionIdToSave}. New session ID for UI: ${newSessionIdForStorage}`);
        } catch (err) {
            const failErrorMsg = err.response?.data?.message || err.message || 'Failed to save/reset session.';
            setError(`Session Error: ${failErrorMsg}`);
            if (!isLoggingOut) {
                const fallbackNewSessionId = uuidv4();
                localStorage.setItem('sessionId', fallbackNewSessionId);
                setSessionId(fallbackNewSessionId);
                setMessages([]);
                handlePromptSelectChange('friendly');
                console.warn("Failed to save history to backend, but reset UI with new session ID:", fallbackNewSessionId);
            }
        } finally {
            setIsLoading(false);
            if (onCompleteCallback) onCompleteCallback();
        }
    }, [messages, isLoading, handlePromptSelectChange]);

    useEffect(() => {
      // This ensures handleLogout always has the latest saveAndReset
      // due to saveAndReset being in handleLogout's dependency array (implicitly via ESLint or explicitly)
    }, [saveAndReset]);


    const handleNewChat = useCallback(() => {
        if (!isLoading) saveAndReset(false, null);
    }, [isLoading, saveAndReset]);

    const handleSendMessage = useCallback(async (e) => {
        if (e) e.preventDefault();
        const textToSend = inputText.trim();
        const currentSessionIdFromStorage = localStorage.getItem('sessionId');
        const currentUserIdFromStorage = localStorage.getItem('userId');

        if (!textToSend) {
            setError("Message text cannot be empty.");
            return;
        }
        if (isLoading) return;

        if (!currentSessionIdFromStorage || !currentUserIdFromStorage) {
            setError("Critical Error: Session or User ID missing. Please try logging out and in again.");
            return;
        }

        const newUserMessage = {
            role: 'user',
            parts: [{ text: textToSend }],
            timestamp: new Date().toISOString(),
        };
        setMessages(prev => [...prev, newUserMessage]);
        setInputText('');
        setError('');
        setIsLoading(true);

        const historyForBackend = messages.map(m => ({
            role: m.role,
            parts: m.parts,
        }));

        const messageData = {
            message: textToSend,
            history: historyForBackend,
            sessionId: currentSessionIdFromStorage,
            systemPrompt: editableSystemPromptText,
            isRagEnabled: isRagEnabled,
            llmProvider: llmProvider,
            llmModelName: llmModelName || null,
            enableMultiQuery: enableMultiQuery,
        };

        console.log("ChatPage: Sending to Node.js /api/chat/message with payload:", messageData);

        try {
            const response = await sendMessage(messageData);
            const responseData = response.data;

            if (!responseData || !responseData.reply || !responseData.reply.parts || !responseData.reply.parts[0]) {
                console.error("Invalid response structure from backend:", responseData);
                throw new Error("Received an invalid or incomplete response from the AI service.");
            }

            const botMessage = {
                role: 'model',
                parts: responseData.reply.parts,
                references: responseData.reply.references || [],
                thinking: responseData.reply.thinking || null,
                timestamp: responseData.reply.timestamp || new Date().toISOString()
            };
            setMessages(prev => [...prev, botMessage]);

        } catch (err) {
            const errorMessage = err.response?.data?.message || err.message || 'Failed to get response.';
            setError(`Chat Error: ${errorMessage}`);
            console.error("ChatPage Send Message Error:", err);
            const systemErrorMessage = {
                role: 'model',
                parts: [{ text: `Error: ${errorMessage}` }],
                timestamp: new Date().toISOString(),
                isError: true
            };
            setMessages(prev => [...prev, systemErrorMessage]);
        } finally {
            setIsLoading(false);
        }
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [inputText, isLoading, messages, isRagEnabled, editableSystemPromptText, llmProvider, llmModelName, enableMultiQuery]);

    const handleEnterKey = useCallback((e) => {
        if (e.key === 'Enter' && !e.shiftKey && !isLoading) {
            e.preventDefault();
            handleSendMessage(null);
        }
    }, [handleSendMessage, isLoading]);

    const handleRagToggle = (event) => setIsRagEnabled(event.target.checked);
    const handleMultiQueryToggle = (event) => setEnableMultiQuery(event.target.checked);

    const handleLlmProviderChange = (event) => {
        const newProvider = event.target.value;
        setLlmProvider(newProvider);
        setLlmModelName(LLM_OPTIONS[newProvider]?.models[0] || '');
    };

    const handleLlmModelChange = (event) => {
        setLlmModelName(event.target.value);
    };

    const handleAnalysisComplete = useCallback((data) => {
        console.log("ChatPage: Received analysis complete data:", data);
        setAnalysisModalData(data);
        setIsAnalysisModalOpen(true);
    }, []);

    const closeAnalysisModal = useCallback(() => {
        setIsAnalysisModalOpen(false);
        setAnalysisModalData(null);
    }, []);

    const isProcessing = isLoading;

    const currentUserIdFromStorage = localStorage.getItem('userId');
    if (!currentUserIdFromStorage && window.location.pathname !== '/login') {
        return <div className="loading-indicator"><span>Session data missing. Redirecting...</span></div>;
    }


    return (
        <div className="chat-page-container">
            <div className="sidebar-area">
                 <SystemPromptWidget
                    selectedPromptId={currentSystemPromptId} promptText={editableSystemPromptText}
                    onSelectChange={handlePromptSelectChange} onTextChange={handlePromptTextChange}
                 />
                <div className="llm-settings-widget">
                    <h4>AI Settings</h4>
                    <div className="setting-item">
                        <label htmlFor="llm-provider-select">Provider:</label>
                        <select id="llm-provider-select" value={llmProvider} onChange={handleLlmProviderChange} disabled={isProcessing}>
                            {Object.keys(LLM_OPTIONS).map(key => (
                                <option key={key} value={key}>{LLM_OPTIONS[key].name}</option>
                            ))}
                        </select>
                    </div>
                    {LLM_OPTIONS[llmProvider] && LLM_OPTIONS[llmProvider].models.length > 0 && (
                        <div className="setting-item">
                            <label htmlFor="llm-model-select">Model:</label>
                            <select id="llm-model-select" value={llmModelName} onChange={handleLlmModelChange} disabled={isProcessing}>
                                {LLM_OPTIONS[llmProvider].models.map(model => (
                                    <option key={model} value={model}>{model}</option>
                                ))}
                                <option value="">Provider Default</option>
                            </select>
                        </div>
                    )}
                    <div className="setting-item rag-toggle-container" title="Enable Multi-Query for RAG">
                        <input type="checkbox" id="multi-query-toggle" checked={enableMultiQuery} onChange={handleMultiQueryToggle}
                               disabled={isProcessing || !isRagEnabled} aria-label="Enable Multi-Query RAG" />
                        <label htmlFor="multi-query-toggle">Multi-Query (RAG)</label>
                    </div>
                </div>
                <FileUploadWidget onUploadSuccess={triggerFileRefresh} />
                 <FileManagerWidget
                    refreshTrigger={fileRefreshTrigger}
                    onAnalysisComplete={handleAnalysisComplete}
                />
            </div>

            <div className="chat-container">
                 <header className="chat-header">
                    <h1>FusedChat: AI Core Integration</h1>
                    <div className="header-controls">
                        <span className="username-display">Hi, {username}!</span>
                        <button onClick={handleHistory} className="header-button history-button" disabled={isProcessing}>History</button>
                        <button onClick={handleNewChat} className="header-button newchat-button" disabled={isProcessing}>New Chat</button>
                        <button onClick={() => handleLogout(false)} className="header-button logout-button" disabled={isProcessing}>Logout</button>
                    </div>
                </header>

                 <div className="messages-area">
                    {messages.map((msg, index) => {
                         if (!msg || typeof msg.role !== 'string' || !Array.isArray(msg.parts) || !(msg.parts[0]?.text || msg.parts[0]?.text === '') ) {
                            console.warn("Rendering invalid message structure at index", index, msg);
                            return <div key={`error-${index}`} className="message-error">Invalid message data</div>;
                         }
                         const isErrorMessage = !!msg.isError;
                         return (
                            <div key={`${sessionId}-${index}-${msg.timestamp || index}`} className={`message ${msg.role.toLowerCase()}${isErrorMessage ? '-error-message' : ''}`}>
                                <div className="message-sender-icon">{msg.role.charAt(0).toUpperCase()}</div>
                                <div className="message-content-wrapper">
                                    <div className="message-sender-name">
                                        {msg.role === 'user' ? username : msg.role.toUpperCase()}
                                    </div>
                                    <div className="message-text">
                                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                            {msg.parts[0].text}
                                        </ReactMarkdown>
                                    </div>
                                    {msg.thinking && (
                                        <details className="message-thinking-trace">
                                            <summary>Thinking Process</summary>
                                            <pre>{typeof msg.thinking === 'string' ? msg.thinking : JSON.stringify(msg.thinking, null, 2)}</pre>
                                        </details>
                                    )}
                                    {msg.references && msg.references.length > 0 && (
                                        <div className="message-references">
                                            <strong>References:</strong>
                                            <ul>
                                                {msg.references.map((ref, i) => (
                                                    <li key={i} title={ref.preview_snippet || 'Reference details'}>
                                                        {ref.documentName || `Source ${i+1}`} (Score: {ref.score?.toFixed(2)})
                                                    </li>
                                                ))}
                                            </ul>
                                        </div>
                                    )}
                                </div>
                                {msg.timestamp && (
                                    <span className="message-timestamp">
                                        {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                    </span>
                                )}
                            </div>
                         );
                    })}
                    <div ref={messagesEndRef} />
                 </div>

                {isProcessing && <div className="loading-indicator"><span>Thinking...</span></div>}
                {!isProcessing && error && <div className="error-indicator">{error}</div>}

                <footer className="input-area">
                    <textarea
                        value={inputText} onChange={(e) => setInputText(e.target.value)} onKeyDown={handleEnterKey}
                        placeholder="Type your message..." rows="1" disabled={isProcessing} aria-label="Chat input"
                    />
                    <div className="rag-toggle-container" title={!hasFiles ? "Upload files to enable RAG" : (isRagEnabled ? "Disable RAG" : "Enable RAG")}>
                        <input type="checkbox" id="rag-toggle" checked={isRagEnabled} onChange={handleRagToggle}
                               disabled={!hasFiles || isProcessing} aria-label="Enable RAG" />
                        <label htmlFor="rag-toggle">RAG</label>
                    </div>
                    <button onClick={() => handleSendMessage(null)} disabled={isProcessing || !inputText.trim()} title="Send Message" aria-label="Send message">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="20" height="20">
                            <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
                        </svg>
                    </button>
                </footer>
            </div>

            <HistoryModal isOpen={isHistoryModalOpen} onClose={closeHistoryModal} />

            {analysisModalData && (
                <AnalysisResultModal
                    isOpen={isAnalysisModalOpen}
                    onClose={closeAnalysisModal}
                    analysisData={analysisModalData}
                />
            )}
        </div>
    );
};

// export const LLM_OPTIONS_FROM_CHATPAGE = LLM_OPTIONS; // <<<--- EXPORT REMOVED
export default ChatPage;