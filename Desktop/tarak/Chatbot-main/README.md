🤖 FusedChatbot: AI-Powered Chat & Document Analysis Platform

Welcome to FusedChatbot, a cutting-edge platform that seamlessly blends advanced AI chat capabilities with powerful document analysis tools. By combining the strengths of two distinct projects—a Python-based chatbot with sophisticated Retrieval Augmented Generation (RAG) and a Node.js/React-based application with Gemini integration—FusedChatbot delivers a robust, user-friendly experience for both conversational AI and document interaction.
___________________________________________________________________________________________________________________________________________________________________
🌟 Key Features

Flexible AI ConversationsInteract with multiple Large Language Models (LLMs) to suit your needs:  

Gemini API (via Google AI SDK)  
Groq API (fast inference with LLaMA 3 and more)  
Ollama (local models like qwen2.5:14b-instruct, llama3.2:latest)Choose your preferred LLM provider and model dynamically for chat and analysis.


Advanced Retrieval Augmented Generation (RAG)  

Upload documents (PDF, TXT, DOCX, PPTX) for intelligent processing.  
Multi-Query RAG: Breaks down queries into sub-queries for deeper context retrieval.  
Grounded responses with clear document references for transparency.


Transparent Chain-of-Thought (CoT)See the AI’s reasoning process displayed in a user-friendly format, enhancing trust and understanding.

Powerful Document Analysis  

Generate FAQs from document content.  
Extract key topics for quick insights.  
Create mindmap outlines to visualize document structure.


Personalized User Experience  

Secure user authentication and session management.  
Per-user document indexing and chat history storage.  
Tailored settings for LLM and RAG preferences.


Modern, Interactive UIBuilt with React for a responsive and intuitive experience, featuring:  

Chat interface with LLM and RAG toggles.  
File management and upload widgets.  
Analysis result displays (FAQs, topics, mindmaps).

___________________________________________________________________________________________________________________________________________________________________

🏛 Architecture Overview
FusedChatbot is designed with a modular, two-service architecture for scalability and flexibility:

Node.js/Express Backend (server/)  

Main application server (default port: 5003).  
Handles user authentication, session management, and chat history storage (MongoDB).  
Serves the React frontend and acts as an API gateway to the Python AI Core Service.  
Key routes: /auth, /chat, /files, /analysis, /upload.


Python AI Core Service (server/ai_core_service/)  

Flask-based microservice for AI-intensive tasks (default port: 5001).  
LLM Handler: Manages Gemini, Groq, and Ollama interactions with CoT parsing.  
RAG Pipeline: Processes documents with text extraction (file_parser.py), semantic search via FAISS (faiss_handler.py), and embeddings (e.g., mixedbread-ai/mxbai-embed-large-v1).  
API endpoints: /add_document, /generate_chat_response, /analyze_document, /health.


React Frontend (client/)  

Runs on http://localhost:3000 by default.  
Communicates with the Node.js backend via REST APIs.  
Components: ChatPage.js, FileManagerWidget.js, FileUploadWidget.js, AnalysisResultModal.js.


MongoDB Database  

Stores user accounts, chat history, and session data.

___________________________________________________________________________________________________________________________________________________________________

🚀 Getting Started
Prerequisites

Node.js (v18+ recommended) & npm (or yarn).  
Python (v3.9+ recommended) & pip.  
MongoDB instance (running and accessible).  
(Optional) Ollama for local model support.  
API keys for Gemini and/or Groq.

Setup Instructions

Clone the RepositoryEnsure you have the project files or clone the repository if available.

Python AI Core Service Setup (server/ai_core_service/)  

Navigate to FusedChatbot/server/ai_core_service/.  
Create and activate a virtual environment:  python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1


Install dependencies:  pip install -r requirements.txt


Create a .env file:  GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
GROQ_API_KEY="YOUR_GROQ_API_KEY"
OLLAMA_BASE_URL="http://localhost:11434"
OLLAMA_MODEL="llama3"




Node.js Backend Setup (server/)  

Navigate to FusedChatbot/server/.  
Install dependencies:  npm install


Create a .env file:  PORT=5003
MONGO_URI="mongodb://localhost:27017/fusedChatbotDB"
JWT_SECRET="yourSuperSecretKeyForJWT"
PYTHON_AI_CORE_SERVICE_URL="http://localhost:5001"




React Frontend Setup (client/)  

Navigate to FusedChatbot/client/.  
Install dependencies:  npm install


(Optional) Create a .env file if the backend port differs:  REACT_APP_BACKEND_PORT=5003





Running the Application

Start MongoDBEnsure your MongoDB instance is running.

Launch Python AI Core Service  

Navigate to FusedChatbot/server/.  
Activate the virtual environment.  
Run:  python -m ai_core_service.app


Confirm it’s running on http://localhost:5001.


Launch Node.js Backend  

In a new terminal, navigate to FusedChatbot/server/.  
Run:  node server.js


Confirm it’s running on http://localhost:5003.


Launch React Frontend  

In a new terminal, navigate to FusedChatbot/client/.  
Run:  npm start


Access the app at http://localhost:3000.

___________________________________________________________________________________________________________________________________________________________________

🎯 How to Use FusedChatbot

Open the app in your browser (http://localhost:3000).  
Sign up or log in to your account.  
Upload documents using the file manager.  
Start chatting:  
Select your preferred LLM provider and model.  
Enable RAG or Multi-Query RAG for document-grounded responses.  
View AI reasoning (CoT) and document references.


Analyze documents for FAQs, topics, or mindmap outlines.  
Review and manage your chat history.

__________________________________________________________________________________________________________________________________________________________________

💡 Technical Highlights

Microservice Design: Decouples AI processing (Python) from web serving (Node.js) for scalability.  
Extensible LLM Integration: llm_handler.py abstracts interactions with Gemini, Groq, and Ollama.  
Advanced RAG: Multi-query generation and FAISS-based semantic search for precise context retrieval.  
Transparent AI: CoT reasoning displayed within <thinking> tags for user insight.  
User-Centric: Per-user document indexing and chat history for a personalized experience.  
Configurable: .env files and config.py for easy setup and customization.

__________________________________________________________________________________________________________________________________________________________________

🔮 Future Enhancements

Save user-preferred LLM settings to profiles.  
Add interactive mindmap visualizations.  
Improve RAG with result re-ranking.  
Enable streaming responses for faster interactions.  
Support advanced document parsing (e.g., OCR for images).  
Containerize with Docker for easier deployment.  
Expand testing with unit and integration tests.

___________________________________________________________________________________________________________________________________________________________________

📚 About FusedChatbot
FusedChatbot is the result of merging two innovative projects: a Python-based chatbot with advanced RAG and document analysis, and a Node.js/React-based application with Gemini integration. This fusion creates a powerful, scalable, and user-friendly platform for AI-driven conversations and document insights.
Get started today and experience the future of AI interaction! 🚀
