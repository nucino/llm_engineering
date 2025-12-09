# LLM Engineering Week 6: Product Pricer AI Agent Instructions

## Project Overview

This is an **LLM fine-tuning project** that builds a model to predict product prices from text descriptions. The architecture centers around three core components:

1. **Data Curation Pipeline** (`items.py`, `loaders.py`) - Processes Amazon product data into training-ready format
2. **Model Training Framework** - Fine-tunes LLaMA models using curated datasets
3. **Evaluation System** (`testing.py`) - Tests model performance with color-coded accuracy metrics

## Key Architecture Patterns

### Data Flow: Raw → Curated → Training → Evaluation
```
Amazon Dataset → ItemLoader → Item objects → Training prompts → Fine-tuned model → Price predictions
```

### Core Components

**`Item` Class** (`items.py`):
- Tokenizes product text to exactly 160-180 tokens using LLaMA tokenizer
- Creates training prompts: `"How much does this cost to the nearest dollar?\n\n{product_text}\n\nPrice is ${price}.00"`
- Test prompts omit the price for inference: `test_prompt()` method
- Filters products by `MIN_TOKENS=150`, `MAX_TOKENS=160`, price range $0.5-$999.49

**`ItemLoader` Class** (`loaders.py`):
- Processes HuggingFace datasets in parallel chunks (`CHUNK_SIZE=1000`)
- Uses `ProcessPoolExecutor` for concurrent data processing
- Loads from `"McAuley-Lab/Amazon-Reviews-2023"` dataset with category filters

**`Tester` Class** (`testing.py`):
- Color-coded evaluation: Green (<20% error), Orange (20-40% error), Red (>40% error)
- Uses RMSLE (Root Mean Square Log Error) as primary metric
- Scatter plots compare predictions vs ground truth

## Environment & Dependencies

### Standard Setup Pattern
```python
# Every notebook starts with this pattern:
load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN', 'your-key-if-not-using-env')

# HuggingFace login required for model access
hf_token = os.environ['HF_TOKEN'] 
login(hf_token, add_to_git_credential=True)
```

### Required Dependencies
- `transformers`, `datasets`, `huggingface_hub` for model/data handling
- `matplotlib` for visualization (use named colors from matplotlib docs)
- `pickle` for dataset serialization/caching
- `tqdm` for progress tracking in data processing
- `openai`, `anthropic` for API-based model comparisons

## Development Workflows

### Data Curation Workflow
1. Load raw Amazon datasets by category (Appliances, Electronics, etc.)
2. Create `Item` objects that filter by content length and price range
3. Tokenize and truncate text to fit model context window
4. Generate training/test prompts with consistent format
5. Save as pickle files for reuse: `pickle.dump(items, open('filename.pkl', 'wb'))`

### Model Training Patterns
- **Base Model**: Always `"meta-llama/Meta-Llama-3.1-8B"` 
- **Token Management**: 180 total tokens (160 product text + 20 prompt overhead)
- **Fine-tuning**: OpenAI API for frontier models, LoRA/QLoRA for open-source
- **Prompt Engineering**: Consistent format across training/inference

### Testing & Evaluation
```python
# Standard testing pattern
from testing import Tester
Tester.test(prediction_function, test_data)

# Creates colored output and scatter plots automatically
```

## Project-Specific Conventions

### Tokenization Strategy
- Use LLaMA tokenizer for consistency: `AutoTokenizer.from_pretrained(BASE_MODEL)`
- Numbers 1-999 are single tokens in LLaMA (unlike Qwen2/Gemma)
- Text cleaning removes product codes (7+ chars with numbers)

### Price Format Standardization
- Always format as `"Price is $123.00"` (with .00 suffix)
- Price range: $0.50 to $999.49 for training data
- Round predictions to nearest dollar for evaluation

### Data Persistence
- Use pickle files for intermediate dataset caching
- HuggingFace Hub for sharing processed datasets
- Format: `Dataset.from_dict({"text": prompts, "price": prices})`

## Critical Integration Points

### HuggingFace Ecosystem
- Dataset loading: `load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{category}")`
- Model access requires Meta approval for LLaMA models
- Push datasets to Hub: `dataset.push_to_hub("username/dataset-name")`

### Concurrent Processing
- Use `ProcessPoolExecutor` for data loading (8 workers default)
- Chunk datasets for memory efficiency
- Progress tracking with `tqdm` for long-running operations

## Debugging & Common Issues

### Memory Management
- Use `torch.cuda.empty_cache()` between model loads
- Consider quantization for large models: `BitsAndBytesConfig(load_in_4bit=True)`

### Data Quality
- Check token counts: Items with <150 or >160 tokens are rejected
- Verify price extraction: Some products have invalid price strings
- Monitor training/test split balance across price ranges

This project follows Ed Donner's LLM Engineering course structure with emphasis on practical, production-ready data curation and model evaluation techniques.