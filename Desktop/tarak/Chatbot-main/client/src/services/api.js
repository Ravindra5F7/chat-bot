// client/src/services/api.js
import axios from 'axios';

// Dynamically determine API Base URL
const getApiBaseUrl = () => {
    const backendPort = process.env.REACT_APP_BACKEND_PORT || 5003; // Default to 5003
    const hostname = window.location.hostname;
    const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
    const backendHost = (hostname === 'localhost' || hostname === '127.0.0.1')
        ? 'localhost'
        : hostname;
    return `${protocol}//${backendHost}:${backendPort}/api`;
};

const API_BASE_URL = getApiBaseUrl();
console.log("API Base URL (api.js):", API_BASE_URL); // Matches desired log

const api = axios.create({
    baseURL: API_BASE_URL,
});

// --- Interceptor to add User ID header ---
api.interceptors.request.use(
    (config) => {
        const userId = localStorage.getItem('userId');
        if (userId) {
            config.headers['x-user-id'] = userId;
        } else if (!config.url.includes('/auth/')) {
             console.warn("API Interceptor: userId not found for non-auth request to", config.url);
        }

        // MODIFIED: Using the Content-Type logic from your provided "modification" block
        // Do not set Content-Type for FormData, let browser do it
        if (!(config.data instanceof FormData)) {
            config.headers['Content-Type'] = 'application/json';
        }
        return config;
    },
    (error) => {
        console.error("API Request Interceptor Error:", error);
        return Promise.reject(error);
    }
);

// --- Interceptor to handle 401 Unauthorized responses ---
api.interceptors.response.use(
    (response) => response, // Pass through successful responses
    (error) => {
        if (error.response && error.response.status === 401) {
            console.warn("API Interceptor: 401 Unauthorized. Clearing auth & redirecting.");
            localStorage.removeItem('sessionId');
            localStorage.removeItem('username');
            localStorage.removeItem('userId');
            // Check if already on login page to prevent redirection loop
            if (!window.location.pathname.includes('/login')) {
                 window.location.href = '/login?sessionExpired=true'; // Redirect to login page
            }
        }
        return Promise.reject(error); // Pass error along
    }
);
// --- End Interceptors ---


// --- NAMED EXPORTS for API functions ---

// Authentication
export const signupUser = (userData) => api.post('/auth/signup', userData);
export const signinUser = (userData) => api.post('/auth/signin', userData);

// Chat Interaction
// MODIFIED: Comment updated to reflect the payload structure from your "modification" block
// messageData should include:
// message: string (this is the current user text, was 'query' in backend payload before)
// history: array of { role: 'user'/'model', parts: [{text: '...'}] } (this is chatHistory for backend)
// sessionId: string,
// systemPrompt: string,
// isRagEnabled: boolean,
// llmProvider: string,
// llmModelName?: string,
// enableMultiQuery?: boolean
export const sendMessage = (messageData) => api.post('/chat/message', messageData); // Endpoint is correct

export const saveChatHistory = (historyData) => api.post('/chat/history', historyData);

// Chat History Retrieval
export const getChatSessions = () => api.get('/chat/sessions');
export const getSessionDetails = (sessionId) => api.get(`/chat/session/${sessionId}`);

// File Upload
// Pass FormData directly
export const uploadFile = (formData) => api.post('/upload', formData);

// File Management
export const getUserFiles = () => api.get('/files');
export const renameUserFile = (serverFilename, newOriginalName) => api.patch(`/files/${serverFilename}`, { newOriginalName });
export const deleteUserFile = (serverFilename) => api.delete(`/files/${serverFilename}`);

// Document Analysis --- ADDED THIS SECTION ---
// analysisData should include:
// document_text: string,
// analysis_type: string ('faq', 'topics', 'mindmap'),
// llm_provider: string,
// llm_model_name?: string (optional)
export const analyzeDocument = (analysisData) => api.post('/analysis/document', analysisData);


// --- DEFAULT EXPORT ---
// Export the configured Axios instance if needed for direct use elsewhere
export default api;