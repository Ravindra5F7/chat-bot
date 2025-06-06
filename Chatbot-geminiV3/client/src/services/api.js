// // client/src/services/api.js
// import axios from 'axios';

// // Dynamically determine API Base URL
// const getApiBaseUrl = () => {
//     // Use REACT_APP_BACKEND_PORT environment variable if set during build, otherwise default
//     // This allows overriding the port via build environment if needed.
//     const backendPort = process.env.REACT_APP_BACKEND_PORT || 4000;
//     const hostname = window.location.hostname; // Get hostname browser is accessing

//     // Use http protocol by default for local development
//     const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:';

//     // If hostname is localhost or 127.0.0.1, construct URL with localhost
//     // Otherwise, use the hostname the frontend was accessed with (e.g., LAN IP)
//     const backendHost = (hostname === 'localhost' || hostname === '127.0.0.1')
//         ? 'localhost'
//         : hostname;

//     return `${protocol}//${backendHost}:${backendPort}/api`;
// };

// const API_BASE_URL = getApiBaseUrl();
// console.log("API Base URL:", API_BASE_URL); // Log the dynamically determined URL

// // Create Axios instance
// const api = axios.create({
//     baseURL: API_BASE_URL,
// });

// // --- Interceptor to add User ID header (TEMP AUTH) ---
// api.interceptors.request.use(
//     (config) => {
//         const userId = localStorage.getItem('userId');
//         const token = localStorage.getItem('token'); // Get the JWT token

//         // Add Authorization header with Bearer token if token exists
//         if (token) {
//             config.headers['Authorization'] = `Bearer ${token}`;
//         } else if (!config.url.includes('/auth/')) {
//              // If no token and it's not an auth request, log a warning
//              console.warn("API Interceptor: No token found in localStorage for non-auth request to", config.url);
//              // Note: The 401 interceptor will handle redirection if the backend rejects the request.
//         }

//         // Add x-user-id header (keeping for now, might be used in some backend logic)
//         if (userId) {
//              config.headers['x-user-id'] = userId; // Keep adding userId header
//         } else if (!config.url.includes('/auth/')) {
//               console.warn("API Interceptor: userId not found in localStorage for non-auth request to", config.url);
//         }

//         // Handle FormData content type specifically
//         if (config.data instanceof FormData) {
//             // Let Axios set the correct 'multipart/form-data' header with boundary
//             // Deleting it ensures Axios handles it automatically.
//             delete config.headers['Content-Type'];
//         } else if (!config.headers['Content-Type']) {
//              // Set default Content-Type for other requests (like JSON) if not already set
//              config.headers['Content-Type'] = 'application/json';
//         }
//         // console.log("API Request Config:", config); // Debug: Log outgoing request config
//         return config;
//     },
//     (error) => {
//         console.error("API Request Interceptor Error:", error);
//         return Promise.reject(error);
//     }
// );

// // --- Interceptor to handle 401 Unauthorized responses ---
// api.interceptors.response.use(
//     (response) => {
//         // Any status code within the range of 2xx cause this function to trigger
//         return response;
//     },
//     (error) => {
//         // Any status codes outside the range of 2xx cause this function to trigger
//         if (error.response && error.response.status === 401) {
//             console.warn("API Response Interceptor: Received 401 Unauthorized. Clearing auth data and redirecting to login.");
//             // Clear potentially invalid auth tokens/user info
//             localStorage.removeItem('sessionId');
//             localStorage.removeItem('username');
//             localStorage.removeItem('userId');

//             // Use window.location to redirect outside of React Router context if needed
//             // Check if already on login page to prevent loop
//             if (!window.location.pathname.includes('/login')) {
//                  window.location.href = '/login?sessionExpired=true'; // Redirect to login page
//             }
//         }
//         // Return the error so that the calling code can handle it (e.g., display message)
//         return Promise.reject(error);
//     }
// );
// // --- End Interceptors ---


// // --- NAMED EXPORTS for API functions ---

// // Authentication
// export const signupUser = (userData) => api.post('/auth/signup', userData);
// export const signinUser = (userData) => api.post('/auth/signin', userData);

// // Chat Interaction
// // messageData includes { message, history, sessionId, systemPrompt, isRagEnabled, relevantDocs, llm_preference }
// export const sendMessage = (messageData) => api.post('/chat/message', messageData);
// export const saveChatHistory = (historyData) => api.post('/chat/history', historyData);

// // RAG Query
// // queryData includes { message }
// export const queryRagService = (queryData) => api.post('/chat/rag', queryData);

// // Chat History Retrieval
// export const getChatSessions = () => api.get('/chat/sessions');
// export const getSessionDetails = (sessionId) => api.get(`/chat/session/${sessionId}`);

// // Document Analysis
// export const getAnalysis = (filename, analysisType, llm_preference) => 
//     api.post('/analyze', { filename, analysis_type: analysisType, llm_preference });

// // Knowledge Graph Retrieval
// export const getKgDataForVisualization = (filename) => 
//     api.get(`/get_kg_data?filename=${encodeURIComponent(filename)}`);

// // File Upload
// // Pass FormData directly
// export const uploadFile = (formData) => api.post('/upload', formData);

// // File Management
// export const getUserFiles = () => api.get('/files');
// export const renameUserFile = (serverFilename, newOriginalName) => api.patch(`/files/${serverFilename}`, { newOriginalName });
// export const deleteUserFile = (serverFilename) => api.delete(`/files/${serverFilename}`);


// // --- DEFAULT EXPORT ---
// // Export the configured Axios instance if needed for direct use elsewhere
// export default api;

// client/src/services/api.js
import axios from 'axios';

// Dynamically determine API Base URL
const getApiBaseUrl = () => {
    const backendPort = process.env.REACT_APP_BACKEND_PORT || 4000; // Node.js backend port
    const hostname = window.location.hostname;
    const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
    const backendHost = (hostname === 'localhost' || hostname === '127.0.0.1') ? 'localhost' : hostname;
    return `${protocol}//${backendHost}:${backendPort}/api`;
};

const API_BASE_URL = getApiBaseUrl();
console.log("[api.js] API Base URL determined as:", API_BASE_URL); // Existing log

// Create Axios instance
const apiClient = axios.create({ // Renamed 'api' to 'apiClient' for clarity if 'api' is default export
    baseURL: API_BASE_URL,
});

// Request Interceptor: Add Auth Token and User ID
apiClient.interceptors.request.use(
    (config) => {
        const token = localStorage.getItem('token');
        const userId = localStorage.getItem('userId');

        if (token) {
            config.headers['Authorization'] = `Bearer ${token}`;
        } else if (!config.url.includes('/auth/')) { // Don't warn for auth endpoints
            console.warn("[api.js Interceptor] No token found for non-auth request to:", config.url);
        }

        if (userId) {
            config.headers['x-user-id'] = userId; // Continue sending if present
        } else if (!config.url.includes('/auth/')) { // Don't warn for auth endpoints
            console.warn("[api.js Interceptor] No userId found for non-auth request to:", config.url);
        }
        
        if (config.data instanceof FormData) {
            delete config.headers['Content-Type']; // Axios handles FormData Content-Type
        } else if (!config.headers['Content-Type']) {
            config.headers['Content-Type'] = 'application/json'; // Default for others
        }
        // console.log("[api.js Interceptor] Outgoing request config:", config); // Uncomment for deep debug
        return config;
    },
    (error) => {
        console.error("[api.js Interceptor] Request Error:", error);
        return Promise.reject(error);
    }
);

// Response Interceptor: Handle 401 Unauthorized
apiClient.interceptors.response.use(
    (response) => response, // Pass through successful responses
    (error) => {
        if (error.response && error.response.status === 401) {
            console.warn("[api.js Interceptor] Received 401 Unauthorized. Clearing auth data and redirecting to login.");
            localStorage.removeItem('token'); // Clear token first on 401
            localStorage.removeItem('sessionId');
            localStorage.removeItem('username');
            localStorage.removeItem('userId');
            // Ensure we are not already on /login to prevent redirect loops
            if (window.location.pathname !== '/login' && !window.location.pathname.startsWith('/login/')) {
                window.location.href = '/login?sessionExpired=true';
            }
        }
        return Promise.reject(error); // Important to reject so calling code can handle
    }
);

// --- API Functions (Named Exports) ---

// Authentication
export const signupUser = (userData) => apiClient.post('/auth/signup', userData);
export const signinUser = (userData) => apiClient.post('/auth/signin', userData);

// Chat Interaction
export const sendMessage = (messageData) => {
    // >>> ADDED/VERIFIED LOGGING <<<
    console.log("[api.js] sendMessage - payload to Node.js /chat/message:", messageData);
    return apiClient.post('/chat/message', messageData);
};
export const saveChatHistory = (historyData) => apiClient.post('/chat/history', historyData);

// RAG Query (Node.js proxies to Python RAG service if needed)
export const queryRagService = (queryData) => apiClient.post('/chat/rag', queryData);

// Chat History Retrieval
export const getChatSessions = () => apiClient.get('/chat/sessions');
export const getSessionDetails = (sessionId) => apiClient.get(`/chat/session/${sessionId}`);

// Document Analysis
export const getAnalysis = (filename, analysisType, llm_preference) => {
    // >>> ADDED/VERIFIED LOGGING <<<
    console.log(`[api.js] getAnalysis - payload to Node.js /analyze: file=${filename}, type=${analysisType}, llm_pref=${llm_preference}`);
    // Ensure your Node.js /api/analyze endpoint can receive and use llm_preference
    return apiClient.post('/analyze', { 
        filename, 
        analysis_type: analysisType, 
        llm_preference: llm_preference // Pass llm_preference here
    });
};

// Knowledge Graph Retrieval
export const getKgDataForVisualization = (filename) => {
    console.log(`[api.js] getKgDataForVisualization for file: ${filename}`);
    // This expects a Node.js route like /api/kg/data/:filename that in turn calls Python app.py's /get_kg_data
    // Ensure you have this Node.js route: e.g., app.use('/api/kg', kgRoutes);
    return apiClient.get(`/kg/data/${encodeURIComponent(filename)}`);
};

// File Upload
export const uploadFile = (formData) => apiClient.post('/upload', formData); // Node.js /api/upload

// File Management
export const getUserFiles = () => apiClient.get('/files'); // Node.js /api/files
export const renameUserFile = (serverFilename, newOriginalName) => 
    apiClient.patch(`/files/${encodeURIComponent(serverFilename)}`, { newOriginalName }); // Node.js /api/files/:filename
export const deleteUserFile = (serverFilename) => 
    apiClient.delete(`/files/${encodeURIComponent(serverFilename)}`); // Node.js /api/files/:filename


// --- Default Export ---
// Export the configured Axios instance for direct use or if preferred by some project patterns.
export default apiClient;