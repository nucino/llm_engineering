````instructions
# LLM Engineering Course - AI Agent Instructions

## Project Overview
This is an 8-week progressive LLM engineering course that teaches students to build sophisticated AI applications, culminating in a multi-agent autonomous system. The course emphasizes hands-on learning through Jupyter notebooks and practical projects.

## Architecture & Structure

### Week Progression
- **Weeks 1-2**: Foundation (web scraping, APIs, basic LLM integration)
- **Weeks 3-4**: Advanced techniques (tokenizers, models, tool usage)
- **Weeks 5-6**: RAG systems (retrieval-augmented generation)  
- **Weeks 7-8**: Fine-tuning and autonomous agents

### Key Technologies
- **LLM Providers**: OpenAI (primary), Anthropic, Google Gemini, Ollama (local)
- **RAG Stack**: LangChain + ChromaDB + sentence-transformers/OpenAI embeddings
- **Development**: Jupyter notebooks, uv package manager, Modal for cloud deployment
- **UI Frameworks**: Gradio for quick interfaces, Streamlit for web apps

## Essential Patterns

### Environment Setup
All notebooks follow this standard pattern:
```python
# Standard imports for LLM projects
import os
from dotenv import load_dotenv
from openai import OpenAI
from IPython.display import Markdown, display

# Load environment variables with override
load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

# Standard API key validation
if not api_key:
    print("No API key found - check troubleshooting notebook")
elif not api_key.startswith("sk-proj-"):
    print("API key format looks incorrect")
else:
    print("API key found and looks good!")
```

### LLM Client Initialization
**OpenAI**: `client = OpenAI()`
**Ollama Local**: `client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")`
**Anthropic**: `client = anthropic.Anthropic()`

### RAG Implementation Pattern (Weeks 5-6)
Standard RAG setup uses this stack:
```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain

# Typical vector store setup
embeddings = OpenAIEmbeddings()  # or HuggingFaceEmbeddings for free
vectorstore = Chroma(persist_directory="vector_db", embedding_function=embeddings)
```

**ChromaDB Collection Management**: Always check and delete existing collections before recreation:
```python
# Check if collection exists and delete
collection_name = "products"
existing_collection_names = client.list_collections()
if collection_name in existing_collection_names:
    client.delete_collection(collection_name)
collection = client.create_collection(collection_name)
```

**SentenceTransformer for Free Embeddings** (Week 8 pattern):
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
vector = model.encode(["text to embed"])[0]  # Returns numpy array
```

## Developer Workflows

### Project Setup
1. Use `uv` for dependency management (replacing conda/pip)
2. Environment file `.env` contains API keys (never commit)
3. Run `uv sync` to install dependencies from `pyproject.toml`

### Debugging Process
1. **Environment Issues**: Use `setup/troubleshooting.ipynb` and `setup/diagnostics.py`
2. **API Problems**: Check key validation patterns in any notebook
3. **Import Errors**: Restart kernel and run cells from top sequentially
4. **Ollama Issues**: Ensure `ollama serve` is running locally

### Community Contributions
- Organized in `week*/community-contributions/` directories
- Follow naming: `day1_project_description.ipynb` or `project_name/`
- Include standalone README.md with setup instructions
- Use modular architecture (separate utils, main app files)

## Code Quality Standards

### Notebook Structure
1. **Header**: Markdown with project description and features
2. **Imports**: Standard import block with environment checks  
3. **Configuration**: Model names, API endpoints as constants
4. **Core Logic**: Functions for data processing, LLM calls
5. **Execution**: Main workflow with clear section headers

### Error Handling
- Always validate API keys before making calls
- Include fallback options (Ollama for OpenAI, free embeddings)
- Provide clear error messages pointing to troubleshooting resources

### File Organization
- **Scripts**: Standalone `.py` files for reusable utilities
- **Notebooks**: `.ipynb` for educational/experimental content
- **Apps**: Gradio/Streamlit apps in dedicated directories with requirements
- **Knowledge Bases**: `knowledge-base/` or `data/` for RAG documents

## Data Persistence & Custom Classes

### The Item Class Pattern (Weeks 6-8)
The `Item` class is central to fine-tuning projects - it represents curated training data:
```python
from items import Item  # Import AFTER HuggingFace login!

# Item structure
class Item:
    title: str
    price: float
    category: str
    prompt: str  # Training prompt with question + description
    token_count: int
```

**Critical workflow**: Items are serialized to `train.pkl` and `test.pkl` using pickle. These files persist across weeks:
- Week 6: Create pkl files via data curation (`pickle.dump(train, file)`)
- Week 7: Load for fine-tuning (`pickle.load(file)`)
- Week 8: Reuse for RAG/ensemble systems

**Always import Item AFTER logging into HuggingFace** because Item initializes a tokenizer:
```python
from huggingface_hub import login
login(hf_token, add_to_git_credential=True)
from items import Item  # Order matters!
```

## Integration Points

### Modal for Cloud Deployment (Week 8)
Modal deploys GPU-intensive workloads with infrastructure-as-code:
```python
import modal
app = modal.App("pricer-service")
image = Image.debian_slim().pip_install("torch", "transformers", "bitsandbytes")
secrets = [modal.Secret.from_name("hf-secret")]  # Or "huggingface-secret"

@app.cls(image=image, secrets=secrets, gpu="T4", timeout=1800)
class Pricer:
    @modal.build()
    def download_model_to_folder(self):
        # Download models during build phase
        pass
    
    @modal.method()
    def price(self, description: str) -> float:
        # Inference method
        pass
```

**Deploy from terminal**: `modal deploy pricer_service` or `modal deploy -m pricer_service`
**Call remotely**: `Pricer = modal.Cls.from_name("pricer-service", "Pricer"); result = Pricer().price.remote("description")`

**Important**: Set Modal secrets at https://modal.com/secrets/ - name it `hf-secret` or `huggingface-secret` consistently.

### Gradio UI Deployment
Standard pattern for interactive interfaces:
```python
import gradio as gr

def process_function(input_text):
    # Your logic here
    return result

# Basic interface
gr.Interface(fn=process_function, inputs="textbox", outputs="textbox").launch()

# Chat interface
gr.ChatInterface(fn=chat_function, type="messages").launch(share=True)
```

**Sharing options**:
- `share=True`: Creates public gradio.live link (requires HTTP tunneling - may fail on corporate networks)
- `inbrowser=True`: Auto-opens browser window
- `server_port=7860`: Specify port (default 7860)

### Local vs Cloud Models
- **Development**: Use Ollama with llama3.2 locally (avoid llama3.3 - 70B too large)
- **Production**: OpenAI gpt-4o-mini for cost efficiency  
- **Fine-tuning**: Modal + HuggingFace for training jobs (Week 7 uses Google Colab)

### Vector Database Setup
- **ChromaDB**: Default choice, `persist_directory` for local storage
- **Embeddings**: OpenAI (paid, 1536 dims) or sentence-transformers/all-MiniLM-L6-v2 (free, 384 dims)
- **Chunking**: RecursiveCharacterTextSplitter with 1000 char chunks, 200 overlap

## Project-Specific Conventions

### Model Selection
- **Default OpenAI**: `gpt-4o-mini` (cost-effective replacement for gpt-4o-mini)
- **Default Ollama**: `llama3.2` (avoid llama3.3 - too large at 70B)
- **Embeddings**: `text-embedding-3-small` or `sentence-transformers/all-MiniLM-L6-v2`
- **Fine-tuning base**: `meta-llama/Meta-Llama-3.1-8B`

### File Naming
- Notebooks: `dayX_descriptive_name.ipynb`
- Solutions: `solutions/` directory with same naming
- Community: `community-contributions/username_project/`
- Data files: `train.pkl`, `test.pkl`, `validation.pkl` (pickle format)

### Documentation Requirements
- Include setup instructions in every project README
- Document API key requirements and cost warnings
- Provide alternative free options (Ollama, HuggingFace)
- Reference troubleshooting resources for common issues

## Week-Specific Patterns

### Week 5-6 RAG: Brute-Force Context Injection
Simple pattern: search context dict by substring matching:
```python
context = {}  # Load employee/product docs into dict
def get_relevant_context(message):
    relevant = []
    for title, details in context.items():
        if title.lower() in message.lower():
            relevant.append(details)
    return relevant
```

### Week 7: Google Colab for Fine-Tuning
Week 7 notebooks run in Google Colab (GPU required). Links in week7/day*.ipynb redirect to Colab notebooks.
Key imports for Colab environment:
```python
from google.colab import userdata
hf_token = userdata.get('HF_TOKEN')
```

### Week 8: Multi-Modal RAG + Ensemble Pricing
Complex architecture combining:
1. **ChromaDB vectorstore** with 400K product embeddings
2. **Fine-tuned LLM** deployed on Modal for specialist pricing
3. **Classical ML** (Random Forest) for ensemble predictions
4. **Autonomous agents** coordinating all pricers

## Common Issues to Avoid
- Never hardcode API keys
- Check Ollama server is running before local LLM calls  
- Use `load_dotenv(override=True)` to reload environment changes
- Restart Jupyter kernel when changing dependencies
- Include error handling for malformed LLM responses (especially JSON)
- **ChromaDB version issues**: If crashes on Windows, try `pip install chromadb==0.5.0`
- **Modal Windows Unicode errors**: Run `modal deploy` from Anaconda Prompt/PowerShell instead of Jupyter if emoji rendering fails
- **HuggingFace login before Item import**: Always `login()` before importing custom classes that use tokenizers
````