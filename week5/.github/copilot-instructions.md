# AI Assistant Instructions for LLM Engineering Week 5

This project implements a Retrieval Augmented Generation (RAG) system for an insurance tech company (Insurellm) knowledge base. Here's what you need to know to work effectively with this codebase:

## Project Architecture

### Key Components
- **Knowledge Base**: Stored in `/knowledge-base/` with categorized markdown files:
  - `company/` - Company information
  - `contracts/` - Contract details  
  - `employees/` - Employee records
  - `products/` - Product information

- **Vector Store**: Uses Chroma DB to store document embeddings
  - Location: `/vector_db/`
  - Each document is chunked and embedded using OpenAI's embedding model
  - Alternative: HuggingFace sentence-transformers for local embedding

### Data Flow
1. Markdown documents are loaded using LangChain's DirectoryLoader
2. Documents are split into chunks (default: 1000 chars with 200 char overlap)
3. Text chunks are embedded and stored in Chroma vector store
4. Query embeddings are compared against stored vectors for retrieval

## Development Guidelines

### Environment Setup
- Create a `.env` file with:
  ```
  OPENAI_API_KEY=your-key
  ```
- Required dependencies in core notebooks:
  - langchain
  - openai
  - chromadb
  - plotly
  - sklearn (for visualization)

### Performance Considerations
- If document chunking crashes, increase chunk size to 2000 and overlap to 400
- Vector store operations can be intensive - consider batch processing for large document sets
- Use community-contributed optimizations from `/community-contributions/` for specific use cases

### Code Patterns
```python
# Standard document loading pattern
loader = DirectoryLoader("path/", glob="**/*.md", 
                       loader_cls=TextLoader,
                       loader_kwargs={'encoding': 'utf-8'})

# Vector store initialization
vectorstore = Chroma.from_documents(documents=chunks,
                                  embedding=embeddings,
                                  persist_directory=db_name)
```

## Critical Paths

### Data Ingestion Pipeline
1. Document loading from markdown files
2. Text chunking and metadata attachment 
3. Embedding generation
4. Vector store persistence

### Query Flow
1. Question/query embedding
2. Vector similarity search
3. Context retrieval
4. LLM response generation

## Community Contributions
- Check `/community-contributions/` for specialized implementations:
  - Gmail integration
  - Obsidian file handling
  - UI implementations
  - Alternative embedding models

## Cost Optimization
- Default to using `gpt-4-mini` for lower costs
- Consider HuggingFace embeddings instead of OpenAI for embedding generation
- Implement proper chunking to minimize API calls

## Note on Updates
This project evolves with student contributions. Review `/community-contributions/` for the latest patterns and optimizations.