// client/src/components/AnalysisResultModal.js
import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './AnalysisResultModal.css'; // We'll add some basic CSS

const AnalysisResultModal = ({ isOpen, onClose, analysisData }) => {
    console.log("AnalysisResultModal: isOpen =", isOpen); // <<< ADDED THIS
    console.log("AnalysisResultModal: analysisData =", analysisData); // <<< ADDED THIS

    if (!isOpen || !analysisData) {
        return null;
    }

    const { type, result, thinking, documentName } = analysisData;

    return (
        <div className="analysis-modal-overlay" onClick={onClose}>
            <div className="analysis-modal-content" onClick={(e) => e.stopPropagation()}>
                <div className="analysis-modal-header">
                    <h3>{type ? type.toUpperCase() : 'Analysis'} Results for: {documentName || 'Document'}</h3>
                    <button onClick={onClose} className="analysis-modal-close-btn">×</button>
                </div>
                <div className="analysis-modal-body">
                    {thinking && (
                        <details className="analysis-thinking-details">
                            <summary>Thinking Process</summary>
                            <pre className="analysis-thinking-text">{thinking}</pre>
                        </details>
                    )}
                    <h4>Analysis Output:</h4>
                    <div className="analysis-result-text">
                        {/* If mindmap or topics, result might be Markdown. If FAQ, it's Q&A pairs. */}
                        {/* ReactMarkdown is good for rendering if the output is structured Markdown. */}
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {result || "No analysis result provided."}
                        </ReactMarkdown>
                    </div>
                </div>
                <div className="analysis-modal-footer">
                    <button onClick={onClose}>Close</button>
                </div>
            </div>
        </div>
    );
};

export default AnalysisResultModal;