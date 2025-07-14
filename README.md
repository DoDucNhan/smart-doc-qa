# Smart Document Q&A System

A modern web application that allows users to upload documents and ask intelligent questions about their content using AI-powered question-answering capabilities.

## 🚀 Features

- **Document Upload**: Support for PDF and TXT files
- **AI-Powered Q&A**: Ask natural language questions about your documents
- **Vector Search**: Semantic search using embeddings for accurate answers
- **User Authentication**: Secure login and registration system
- **Real-time Processing**: Background document processing with status updates
- **Modern UI**: Clean, responsive interface built with Next.js and Tailwind CSS
- **RESTful API**: Django REST Framework backend with comprehensive endpoints

## 🛠 Tech Stack

### Backend
- **Django 4.x** - Web framework
- **Django REST Framework** - API development
- **LangChain** - LLM integration and document processing
- **HuggingFace** - Framework for language model for question answering
- **FAISS** - Vector similarity search (local development)
- **SQLite** - Database (development) / PostgreSQL (production)

### Frontend
- **Next.js 14** - React framework with App Router
- **TypeScript** - Type-safe JavaScript
- **Tailwind CSS** - Utility-first CSS framework
- **Axios** - HTTP client for API calls

### AI/ML
- **OpenAI Embeddings** - Text vectorization
- **LangChain Document Loaders** - PDF and text processing
- **Vector Search** - Semantic similarity matching

## 📋 Prerequisites

- Python 3.9+
- Node.js 18+
- npm or yarn
- HuggingFace key
- Git

## 🔧 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/smart-doc-qa.git
cd smart-doc-qa
```

### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Navigate to Django project
cd docqa_backend

# Set up environment variables
cp .env.example .env
# Edit .env file with your OpenAI API key

# Run migrations
python manage.py makemigrations
python manage.py migrate

# Create superuser (optional)
python manage.py createsuperuser

# Start development server
python manage.py runserver
```

### 3. Frontend Setup

```bash
# Navigate to frontend directory (from project root)
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

## ⚙️ Environment Variables

Create a `.env` file in the `backend/` directory:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Django Settings
SECRET_KEY=your_django_secret_key
DEBUG=True

# Database (for production)
DATABASE_URL=postgresql://user:password@localhost:5432/dbname

# Databricks (optional, for production)
DATABRICKS_TOKEN=your_databricks_token
DATABRICKS_HOST=your_databricks_host
```

## 🚦 Usage

### 1. Start the Application

**Backend:**
```bash
cd backend/docqa_backend
python manage.py runserver
```

**Frontend:**
```bash
cd frontend
npm run dev
```

### 2. Access the Application

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- Django Admin: http://localhost:8000/admin

### 3. Using the System

1. **Register/Login**: Create an account or sign in
2. **Upload Documents**: Upload PDF or TXT files
3. **Wait for Processing**: Documents are processed in the background
4. **Ask Questions**: Click on a processed document and ask questions
5. **Get AI Answers**: Receive intelligent responses based on document content

## 📁 Project Structure

```
smart-doc-qa/
├── backend/
│   ├── documents/          # Document management app
│   ├── authentication/     # User authentication app
│   ├── docqa_backend/      # Main Django project
│   ├── manage.py
│   ├── test_auth.py
│   ├── test_huggingface.py
│   ├── requirements.txt
│   └── .env
├── frontend/
│   ├── src/
│   │   ├── app/               # Next.js app directory
│   │   ├── components/        # React components
│   │   └── contexts/          # React contexts
│   ├── package.json
│   └── next.config.js
└── README.md
```

## 🔌 API Endpoints

### Authentication
- `POST /auth/register/` - User registration
- `POST /auth/login/` - User login

### Documents
- `GET /api/documents/` - List user documents
- `POST /api/documents/` - Upload new document
- `GET /api/documents/{id}/` - Get document details
- `DELETE /api/documents/{id}/` - Delete document
- `POST /api/documents/{id}/ask_question/` - Ask question about document

## 🧪 Testing

### Backend Tests
```bash
cd backend/docqa_backend
python manage.py test
```

### Frontend Tests
```bash
cd frontend
npm test
```

### Manual API Testing
```bash
# Register user
curl -X POST http://localhost:8000/auth/register/ \
  -H "Content-Type: application/json" \
  -d '{"username": "testuser", "password": "testpass123", "email": "test@example.com"}'

# Upload document (after login)
curl -X POST http://localhost:8000/api/documents/ \
  -H "Authorization: Token your_token_here" \
  -F "title=Test Document" \
  -F "file=@path/to/document.pdf"
```

## 🐳 Docker Deployment

### Using Docker Compose
```bash
# Build and start all services
docker-compose up --build

# Stop services
docker-compose down
```

### Individual Docker Commands
```bash
# Build backend
docker build -t smart-doc-qa-backend ./backend

# Build frontend  
docker build -t smart-doc-qa-frontend ./frontend

# Run with environment variables
docker run -p 8000:8000 --env-file .env smart-doc-qa-backend
```

## 🌐 Production Deployment

### Backend (Django)
1. Set `DEBUG=False` in settings
2. Configure PostgreSQL database
3. Set up static file serving
4. Use Gunicorn or uWSGI
5. Configure reverse proxy (Nginx)

### Frontend (Next.js)
1. Build production bundle: `npm run build`
2. Start production server: `npm start`
3. Or deploy to Vercel, Netlify, etc.

### Environment Setup
```bash
# Production environment variables
export DEBUG=False
export DATABASE_URL=postgresql://...
export OPENAI_API_KEY=...
export ALLOWED_HOSTS=yourdomain.com
```

## 🔒 Security Considerations

- ✅ Token-based authentication
- ✅ CORS configuration
- ✅ File upload validation
- ✅ User isolation (documents per user)
- ⚠️ Add rate limiting for production
- ⚠️ Implement file size limits
- ⚠️ Add input sanitization
- ⚠️ Use HTTPS in production

## 🚨 Troubleshooting

### Common Issues

**Backend Issues:**
- `ImportError`: Run `pip install -r requirements.txt`
- Database errors: Run `python manage.py migrate`
- CORS errors: Check `CORS_ALLOWED_ORIGINS` in settings
- File upload errors: Verify `MEDIA_ROOT` permissions

**Frontend Issues:**
- API connection errors: Check backend URL in axios calls
- Authentication errors: Verify token storage
- Build errors: Run `npm install` and check dependencies

**AI/ML Issues:**
- OpenAI API errors: Verify API key and billing
- Memory issues: Reduce chunk sizes or add pagination
- Slow processing: Consider background task queues

## 📈 Performance Optimization

### Backend
- Use Celery for background document processing
- Implement caching with Redis
- Add database indexing
- Use connection pooling

### Frontend
- Implement pagination for document lists
- Add loading states and error boundaries
- Use React.memo for expensive components
- Optimize bundle size

### Vector Search
- Use Databricks Vector Search for production
- Implement batch processing for embeddings
- Add result caching
- Consider approximate search algorithms

## 🛣 Roadmap

### Phase 1 (Current)
- [x] Basic document upload and Q&A
- [x] User authentication
- [x] Vector search with FAISS
- [x] Web interface

### Phase 2
- [ ] Support for more file types (DOCX, PPT)
- [ ] Multi-document question answering
- [ ] Question history and favorites
- [ ] Advanced search and filtering

### Phase 3
- [ ] Real-time collaboration
- [ ] Document sharing and permissions
- [ ] Analytics and insights
- [ ] Mobile app

### Phase 4
- [ ] Enterprise features (SSO, RBAC)
- [ ] API rate limiting and quotas
- [ ] Advanced AI features
- [ ] Integration marketplace


### Development Guidelines
- Follow PEP 8 for Python code
- Use TypeScript for all new frontend code
- Add tests for new features
- Update documentation
- Use conventional commit messages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the excellent LLM framework
- [Django](https://djangoproject.com/) and [Next.js](https://nextjs.org/) communities

---

**Built with ❤️ using Django, Next.js, and LangChain**