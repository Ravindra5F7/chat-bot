// client/src/components/FileManagerWidget.js
import React, { useState, useEffect, useCallback } from 'react';
import { getUserFiles, renameUserFile, deleteUserFile, analyzeDocument } from '../services/api';
import { LLM_OPTIONS } from '../config/constants';

// Helper Functions
const getFileIcon = (type) => {
  switch (type) {
    case 'docs': return '📄';
    case 'images': return '🖼️';
    case 'code': return '💻';
    default: return '📁';
  }
};
const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  if (typeof bytes !== 'number' || bytes < 0) return 'N/A';
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  const index = Math.max(0, Math.min(i, sizes.length - 1));
  return parseFloat((bytes / Math.pow(k, index)).toFixed(1)) + ' ' + sizes[index];
};

const FileManagerWidget = ({ refreshTrigger, onAnalysisComplete }) => {
  const [userFiles, setUserFiles] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [renamingFile, setRenamingFile] = useState(null);
  const [newName, setNewName] = useState('');

  const [analyzingFile, setAnalyzingFile] = useState(null);
  const [selectedAnalysisType, setSelectedAnalysisType] = useState('faq');
  const defaultAnalysisProvider = 'groq_llama3';
  const [analysisLlmProvider, setAnalysisLlmProvider] = useState(defaultAnalysisProvider);
  const [analysisLlmModel, setAnalysisLlmModel] = useState(
      LLM_OPTIONS[defaultAnalysisProvider]?.models[0] || ''
  );
  const [isAnalyzingInProgress, setIsAnalyzingInProgress] = useState(false);
  const [currentAnalysisError, setCurrentAnalysisError] = useState('');

  const fetchUserFiles = useCallback(async () => {
    const currentUserId = localStorage.getItem('userId');
    if (!currentUserId) { setUserFiles([]); return; }
    setIsLoading(true); setError('');
    try {
      const response = await getUserFiles();
      setUserFiles(response.data || []);
    } catch (err) {
      console.error("Error fetching user files:", err);
      setError(err.response?.data?.message || 'Failed to load files.');
      setUserFiles([]);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchUserFiles();
  }, [refreshTrigger, fetchUserFiles]);

  const handleRenameClick = (file) => { setRenamingFile(file.serverFilename); setNewName(file.originalName); setError(''); };
  const handleRenameCancel = () => { setRenamingFile(null); setNewName(''); setError(''); };

  const handleRenameSave = async () => {
    if (!renamingFile || !newName.trim()) { setError('New name cannot be empty.'); return; }
    if (newName.includes('/') || newName.includes('\\')) { setError('New name cannot contain slashes.'); return; }
    setError('');
    try {
      await renameUserFile(renamingFile, newName.trim());
      setRenamingFile(null); setNewName(''); fetchUserFiles();
    } catch (err) {
      console.error("Error renaming file:", err);
      setError(err.response?.data?.message || 'Failed to rename file.');
    }
  };

  const handleRenameInputKeyDown = (e) => {
      if (e.key === 'Enter') handleRenameSave();
      else if (e.key === 'Escape') handleRenameCancel();
  };

  const handleDeleteFile = async (serverFilename, originalName) => {
    if (!window.confirm(`Are you sure you want to delete "${originalName}"?`)) return;
    setError('');
    try {
      await deleteUserFile(serverFilename);
      fetchUserFiles();
    } catch (err) {
      console.error("Error deleting file:", err);
      setError(err.response?.data?.message || 'Failed to delete file.');
    }
  };

  const handleAnalyzeClick = (file) => {
    setAnalyzingFile({ serverFilename: file.serverFilename, originalName: file.originalName });
    setSelectedAnalysisType('faq');
    setCurrentAnalysisError('');
    setAnalysisLlmProvider(defaultAnalysisProvider);
    setAnalysisLlmModel(LLM_OPTIONS[defaultAnalysisProvider]?.models[0] || '');
  };

  const handleAnalysisTypeChange = (e) => setSelectedAnalysisType(e.target.value);
  const handleAnalysisProviderChange = (e) => {
    const newProvider = e.target.value;
    setAnalysisLlmProvider(newProvider);
    setAnalysisLlmModel(LLM_OPTIONS[newProvider]?.models[0] || '');
  };
  const handleAnalysisModelChange = (e) => setAnalysisLlmModel(e.target.value);

  const handleRunAnalysis = async () => {
    if (!analyzingFile) return;
    setIsAnalyzingInProgress(true);
    setCurrentAnalysisError('');
    try {
        const payload = {
            documentName: analyzingFile.originalName,
            serverFilename: analyzingFile.serverFilename,
            analysisType: selectedAnalysisType,
            llmProvider: analysisLlmProvider,
            llmModelName: analysisLlmModel || null,
        };
        const response = await analyzeDocument(payload); // This is the call using api.js

        // VVVV THESE ARE THE IMPORTANT LOGS VVVV
        console.log("FileManagerWidget: Received FULL response object from analyzeDocument API:", response);
        console.log("FileManagerWidget: Actual DATA from Node.js (response.data):", response.data);
        // VVVV END IMPORTANT LOGS VVVV

        if (response.data && response.data.status === 'success') {
            console.log("FileManagerWidget: Analysis success condition met. Passing to onAnalysisComplete. Raw data:", response.data);
            if (onAnalysisComplete) {
                onAnalysisComplete({
                    type: response.data.analysisType || selectedAnalysisType, // Fallback to selectedType
                    result: response.data.analysisResult,
                    thinking: response.data.thinkingContent || null, // Fallback to null
                    documentName: response.data.documentName || analyzingFile.originalName // Fallback
                });
            }
            setAnalyzingFile(null);
        } else {
            console.error("FileManagerWidget: Analysis API call did NOT meet success condition. Response.data was:", response.data);
            throw new Error(response.data?.message || "Analysis API call did not return expected success status.");
        }
    } catch (err) {
        const errorMsg = err.response?.data?.message || err.message || "Failed to perform analysis.";
        setCurrentAnalysisError(errorMsg);
        console.error("FileManager Analysis Error:", err);
    } finally {
        setIsAnalyzingInProgress(false);
    }
  };

  return (
    <div className="file-manager-widget"> {/* This class is styled by ChatPage.css & FileManagerWidgetCSS */}
      <div className="fm-header">
        <h4>Your Uploaded Files</h4>
        <button onClick={fetchUserFiles} disabled={isLoading || isAnalyzingInProgress} className="fm-refresh-btn" title="Refresh File List">🔄</button>
      </div>

      {error && <div className="fm-error">{error}</div>}

      <div className="fm-file-list-container">
        {isLoading && userFiles.length === 0 ? ( <p className="fm-loading">Loading files...</p> ) :
         userFiles.length === 0 && !isLoading ? ( <p className="fm-empty">No files uploaded yet.</p> ) : (
          <ul className="fm-file-list">
            {userFiles.map((file) => (
              <li key={file.serverFilename} className="fm-file-item">
                <span className="fm-file-icon">{getFileIcon(file.type)}</span>
                <div className="fm-file-details">
                  {renamingFile === file.serverFilename ? (
                    <div className="fm-rename-section">
                      <input type="text" value={newName} onChange={(e) => setNewName(e.target.value)} onKeyDown={handleRenameInputKeyDown} autoFocus className="fm-rename-input"/>
                      <button onClick={handleRenameSave} disabled={!newName.trim()} className="fm-action-btn fm-save-btn" title="Save Name">✔️</button>
                      <button onClick={handleRenameCancel} className="fm-action-btn fm-cancel-btn" title="Cancel Rename">❌</button>
                    </div>
                  ) : (
                    <>
                      <span className="fm-file-name" title={file.originalName}>{file.originalName}</span>
                      <span className="fm-file-size">{formatFileSize(file.size)}</span>
                    </>
                  )}
                </div>

                {renamingFile !== file.serverFilename && (
                  <div className="fm-file-actions">
                    <button onClick={() => handleAnalyzeClick(file)} disabled={!!renamingFile || isAnalyzingInProgress} className="fm-action-btn fm-analyze-btn" title="Analyze Document">🔬</button>
                    <button onClick={() => handleRenameClick(file)} disabled={!!renamingFile || isAnalyzingInProgress} className="fm-action-btn fm-rename-btn" title="Rename">✏️</button>
                    <button onClick={() => handleDeleteFile(file.serverFilename, file.originalName)} disabled={!!renamingFile || isAnalyzingInProgress} className="fm-action-btn fm-delete-btn" title="Delete">🗑️</button>
                  </div>
                )}

                {analyzingFile && analyzingFile.serverFilename === file.serverFilename && (
                    <div className="fm-analysis-options">
                        {currentAnalysisError && <div className="fm-error fm-analysis-error">{currentAnalysisError}</div>}
                        <select value={selectedAnalysisType} onChange={handleAnalysisTypeChange} disabled={isAnalyzingInProgress}>
                            <option value="faq">Generate FAQ</option>
                            <option value="topics">Identify Topics</option>
                            <option value="mindmap">Create Mindmap Outline</option>
                        </select>
                        <select value={analysisLlmProvider} onChange={handleAnalysisProviderChange} disabled={isAnalyzingInProgress}>
                            {Object.keys(LLM_OPTIONS).map(key => (
                                <option key={key} value={key}>{LLM_OPTIONS[key].name}</option>
                            ))}
                        </select>
                        {LLM_OPTIONS[analysisLlmProvider] && LLM_OPTIONS[analysisLlmProvider].models.length > 0 && (
                            <select value={analysisLlmModel} onChange={handleAnalysisModelChange} disabled={isAnalyzingInProgress}>
                                {LLM_OPTIONS[analysisLlmProvider].models.map(model => (
                                    <option key={model} value={model}>{model}</option>
                                ))}
                                <option value="">Provider Default</option>
                            </select>
                        )}
                        <button onClick={handleRunAnalysis} disabled={isAnalyzingInProgress} className="fm-action-btn">
                            {isAnalyzingInProgress ? 'Analyzing...' : 'Run'}
                        </button>
                        <button onClick={() => setAnalyzingFile(null)} disabled={isAnalyzingInProgress} className="fm-action-btn">Cancel</button>
                    </div>
                )}
              </li>
            ))}
          </ul>
        )}
         {isLoading && userFiles.length > 0 && <p className="fm-loading fm-loading-bottom">Processing...</p>}
      </div>
    </div>
  );
};

// --- CSS for FileManagerWidget ---
// MODIFIED: Styles adjusted to allow the entire sidebar to scroll.
// This means this widget will take its natural content height.
const FileManagerWidgetCSS = `
.file-manager-widget {
  /* This widget is a flex item in ChatPage.css's .sidebar-area (which is display:flex, flex-direction:column) */
  /* It should NOT have flex-grow:1 or overflow:hidden if the whole sidebar scrolls. */
  /* It will take its natural content height. */
  display: flex; /* Still useful for internal layout of header and list-container */
  flex-direction: column;
  /* min-height: 150px; /* Optional: can be kept or removed depending on desired minimum */
  padding: 10px 15px; /* Overall padding for the widget's content area */
  box-sizing: border-box;
}

.fm-header,
.fm-error,
.fm-loading,
.fm-empty {
  flex-shrink: 0;       /* ✅ These won't shrink and will take their content height */
  padding: 5px 0; /* Minimal vertical padding, horizontal handled by parent or specific rules */
  text-align: center;
}

.fm-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border-color, #3a3a3a);
    padding-bottom: 10px; /* Space below header text/button */
    margin-bottom: 10px; /* Space between header and file list or error message */
}
.fm-header h4 {
    margin: 0;
    font-size: 0.9em;
    color: var(--text-secondary, #a0a0a0);
}
.fm-refresh-btn {
    background: none; border: none;
    color: var(--text-secondary, #a0a0a0);
    font-size: 1.2em; cursor: pointer; padding: 5px;
}
.fm-refresh-btn:hover:not(:disabled) { color: var(--text-primary, #e0e0e0); }

.fm-error {
    color: var(--error-color, #f44747);
    background-color: var(--error-bg, rgba(244, 71, 71, 0.1));
    border: 1px solid var(--error-color, #f44747);
    border-radius: 4px;
    margin-bottom: 10px;
    padding: 8px; /* More visible padding for errors */
}
.fm-loading-bottom {
    padding: 5px;
    font-size: 0.8rem;
}

.fm-file-list-container {
  /* REMOVED: flex-grow: 1; */
  /* REMOVED: overflow-y: auto; */
  /* This container will now expand to the height of its content (the ul) */
  /* Padding can be removed if .file-manager-widget provides enough */
}

/* Detailed styles for list items and their contents */
.fm-file-list { list-style: none; padding: 0; margin: 0; }
.fm-file-item { display: flex; align-items: center; padding: 8px 0; border-bottom: 1px solid var(--border-color, #3a3a3a); flex-wrap: wrap; }
.fm-file-item:last-child { border-bottom: none; }
.fm-file-icon { margin-right: 10px; font-size: 1.1em; }
.fm-file-details { flex-grow: 1; display: flex; flex-direction: column; overflow: hidden; }
.fm-file-name { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: var(--text-primary, #e0e0e0); font-weight: 500; }
.fm-file-size { font-size: 0.75rem; color: var(--text-secondary, #a0a0a0); }
.fm-file-actions { margin-left: auto; display: flex; align-items: center; flex-shrink: 0; }
.fm-action-btn {
    background: none; border: none;
    color: var(--text-secondary, #a0a0a0);
    cursor: pointer; padding: 4px 6px;
    font-size: 0.9em;
    border-radius: 4px; margin-left: 5px;
    transition: color 0.2s, background-color 0.2s;
}
.fm-action-btn:hover:not(:disabled) { color: var(--text-primary, #e0e0e0); background-color: var(--bg-input, #2a2a2a); }
.fm-action-btn:disabled { opacity: 0.4; cursor: not-allowed; }
.fm-analyze-btn:hover:not(:disabled) { color: #72bcd4; }
.fm-rename-btn:hover:not(:disabled) { color: #ffe082; }
.fm-delete-btn:hover:not(:disabled) { color: #ff8a80; }

.fm-rename-section { display: flex; align-items: center; width: 100%; }
.fm-rename-input {
    flex-grow: 1; padding: 5px;
    border: 1px solid var(--border-color, #3a3a3a);
    background-color: var(--bg-input, #2a2a2a);
    color: var(--text-primary, #e0e0e0);
    border-radius: 3px; font-size: 0.85rem; margin-right: 5px;
}
.fm-save-btn, .fm-cancel-btn { padding: 5px 8px; }
.fm-save-btn:hover:not(:disabled) { background-color: var(--success-bg, rgba(76,175,80,0.1)); color: var(--success-color, #4caf50); }
.fm-cancel-btn:hover:not(:disabled) { background-color: var(--error-bg, rgba(244,71,71,0.1)); color: var(--error-color, #f44747); }

.fm-analysis-options {
    background-color: #303035; padding: 10px;
    margin-top: 8px; border-radius: 4px;
    display: flex; flex-direction: column; gap: 10px;
    width: 100%; box-sizing: border-box; border: 1px solid #404045;
}
.fm-analysis-options select, .fm-analysis-options button {
    padding: 7px 10px; border-radius: 4px;
    border: 1px solid var(--border-color, #555);
    background-color: var(--bg-input, #2c2c30);
    color: var(--text-primary, #e0e0e0); font-size: 0.9rem;
}
.fm-analysis-options button { cursor: pointer; transition: background-color 0.2s; }
.fm-analysis-options button:hover:not(:disabled) { background-color: var(--accent-blue, #007acc); color: white; }
.fm-analysis-options button:disabled { opacity: 0.6; cursor: not-allowed; }
.fm-analysis-error {
    font-size: 0.85rem; padding: 8px; margin-bottom: 8px;
    color: var(--error-color, #f44747);
    background-color: var(--error-bg, rgba(244, 71, 71, 0.1));
    border: 1px solid var(--error-color, #f44747);
    border-radius: 4px; text-align: left;
}
`;

// --- Inject CSS (keep your existing injection logic) ---
const styleTagFileManagerId = 'file-manager-widget-styles';
if (!document.getElementById(styleTagFileManagerId)) {
    const styleTag = document.createElement("style");
    styleTag.id = styleTagFileManagerId;
    styleTag.type = "text/css";
    styleTag.innerText = FileManagerWidgetCSS;
    document.head.appendChild(styleTag);
}

export default FileManagerWidget;