# Lab 2.6: Embeddings, Context Windows, and Tokenisation

## Objective
Understand tokenization, embeddings, and context window limitations in LLMs through hands-on exploration.

## Duration
~15 minutes

## Prerequisites
- Python 3.8+
- Azure OpenAI API key and endpoint

---

## Exercise: Exploring Tokens, Embeddings, and Context Limits

### Background
- **Tokens**: LLMs don't read "words" - they read "tokens" (subword units)
- **Embeddings**: Dense vector representations of tokens/text
- **Context Window**: Maximum number of tokens the model can process at once
- **Tokenizers**: BPE (Byte Pair Encoding), SentencePiece

---

## Step-by-Step Instructions

### Step 1: Create Your Lab File

```cmd
cd C:\genai-labs
notepad tokens_embeddings.py
```

### Step 2: Write the Exploration Code

```python
from openai import OpenAI
import tiktoken  # OpenAI's tokenizer library

# Azure OpenAI Configuration
endpoint = "YOUR_ENDPOINT"
deployment_name = "YOUR_DEPLOYMENT_NAME"
api_key = "YOUR_API_KEY"

client = OpenAI(base_url=endpoint, api_key=api_key)

def count_tokens(text, model="gpt-4"):
    """
    Count tokens using tiktoken library
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(text)
        return len(tokens), tokens
    except:
        # Fallback: approximate (1 token ≈ 4 characters)
        return len(text) // 4, []

def demo_tokenization():
    """
    Demonstrate how text is tokenized
    """
    print("=" * 70)
    print("1. TOKENIZATION: How LLMs Read Text")
    print("=" * 70)
    
    print("\nLLMs don't read 'words' - they read 'tokens' (subword units)")
    print("Tokenization splits text into tokens based on patterns learned from data\n")
    
    test_texts = [
        "Hello world",
        "Hello, world!",
        "Extraordinary",
        "SupercalifragilisticExpialidocious",
        "GPT-4 is amazing",
        "人工智能",  # Chinese
        "print('Hello')"  # Code
    ]
    
    for text in test_texts:
        token_count, tokens = count_tokens(text)
        print(f"Text: '{text}'")
        print(f"  → Token count: {token_count}")
        if tokens:
            # Show first few token IDs
            print(f"  → First 5 token IDs: {tokens[:5]}")
        print()
    
    print("💡 Key Observations:")
    print("   - Common words: 1 token (e.g., 'Hello', 'world')")
    print("   - Punctuation: Often separate tokens")
    print("   - Rare/long words: Multiple tokens")
    print("   - Non-English: Can be many tokens per character")
    print("   - Code: Special handling for syntax")

def demo_token_vs_word():
    """
    Compare token count vs word count
    """
    print("\n\n" + "=" * 70)
    print("2. TOKENS vs WORDS vs CHARACTERS")
    print("=" * 70)
    
    texts = [
        "The quick brown fox jumps.",
        "Antidisestablishmentarianism is long.",
        "I love AI and ML!",
        "你好世界"  # Chinese: Hello World
    ]
    
    print("\n| Text | Words | Characters | Tokens |")
    print("|------|-------|------------|--------|")
    
    for text in texts:
        word_count = len(text.split())
        char_count = len(text)
        token_count, _ = count_tokens(text)
        
        print(f"| {text[:30]:<30} | {word_count:>5} | {char_count:>10} | {token_count:>6} |")
    
    print("\n💡 Key Points:")
    print("   - English: ~1 token per word (sometimes less)")
    print("   - Long words: More than 1 token")
    print("   - Non-English (Chinese): Many tokens per character")
    print("   - Rule of thumb: 1 token ≈ 4 characters (English)")

def demo_cost_implications():
    """
    Show cost implications of tokenization
    """
    print("\n\n" + "=" * 70)
    print("3. TOKEN PRICING: Why Token Count Matters")
    print("=" * 70)
    
    print("\nLLM APIs charge by TOKEN, not by word or character!\n")
    
    # Example pricing (GPT-4)
    input_price_per_1k = 0.03  # $0.03 per 1K input tokens
    output_price_per_1k = 0.06  # $0.06 per 1K output tokens
    
    print(f"Example: GPT-4 Pricing")
    print(f"  Input:  ${input_price_per_1k} per 1,000 tokens")
    print(f"  Output: ${output_price_per_1k} per 1,000 tokens")
    
    # Calculate cost for different scenarios
    scenarios = [
        ("Short query (50 tokens in, 100 out)", 50, 100),
        ("Medium doc analysis (2K in, 500 out)", 2000, 500),
        ("Long document (8K in, 1K out)", 8000, 1000),
        ("Max context (128K in, 2K out)", 128000, 2000)
    ]
    
    print("\n| Scenario | Input | Output | Cost |")
    print("|----------|-------|--------|------|")
    
    for desc, inp, out in scenarios:
        cost = (inp / 1000 * input_price_per_1k) + (out / 1000 * output_price_per_1k)
        print(f"| {desc:<35} | {inp:>6} | {out:>6} | ${cost:>6.4f} |")
    
    print("\n💡 Practical Tips:")
    print("   - Count tokens BEFORE sending to API")
    print("   - Truncate long documents to fit context window")
    print("   - Use cheaper models (GPT-3.5) when possible")
    print("   - Monitor token usage to control costs")

def demo_context_window():
    """
    Demonstrate context window limitations
    """
    print("\n\n" + "=" * 70)
    print("4. CONTEXT WINDOW: The Token Limit")
    print("=" * 70)
    
    print("\nContext Window = Maximum tokens model can process (input + output)\n")
    
    print("📊 Context Windows by Model:")
    models = [
        ("GPT-3.5-turbo", 4096, "4K"),
        ("GPT-3.5-turbo-16k", 16384, "16K"),
        ("GPT-4", 8192, "8K"),
        ("GPT-4-32k", 32768, "32K"),
        ("GPT-4-turbo", 128000, "128K"),
        ("Claude 3 Opus", 200000, "200K"),
        ("Gemini 1.5 Pro", 1000000, "1M")
    ]
    
    print("\n| Model | Context Window | Tokens |")
    print("|-------|----------------|--------|")
    for name, tokens, display in models:
        print(f"| {name:<20} | {display:>14} | {tokens:>8,} |")
    
    print("\n⚠️  What Happens at the Boundary?")
    print("   - Exceeding limit → API returns error")
    print("   - Must truncate input or use summarization")
    print("   - Output tokens count towards limit too!")
    
    print("\n💡 Context Window = System Prompt + User Input + History + Output")
    
    # Example calculation
    print("\n📝 Example Breakdown:")
    print("   System prompt:    200 tokens")
    print("   User query:       100 tokens")
    print("   Conversation:   1,000 tokens")
    print("   Response:         500 tokens")
    print("   ─────────────────────────────")
    print("   Total:          1,800 tokens")
    print("   ")
    print("   ✓ Fits in 4K context window")
    print("   ✓ Leaves 2,296 tokens for more conversation")

def demo_embeddings():
    """
    Demonstrate embeddings and their use
    """
    print("\n\n" + "=" * 70)
    print("5. EMBEDDINGS: Vector Representations of Text")
    print("=" * 70)
    
    print("\nWhat are Embeddings?")
    print("  - Dense vector representations of text")
    print("  - Capture semantic meaning")
    print("  - Similar meanings → similar vectors")
    
    print("\n📐 Embedding Dimensions:")
    print("   - text-embedding-ada-002 (OpenAI): 1536 dimensions")
    print("   - text-embedding-3-small: 1536 dimensions")
    print("   - text-embedding-3-large: 3072 dimensions")
    print("   - BGE, E5 (open-source): 768-1024 dimensions")
    
    print("\n🎯 Use Cases:")
    print("   1. Semantic Search")
    print("      - Convert query to embedding")
    print("      - Find documents with similar embeddings")
    print("      - Used in: RAG, document search")
    
    print("\n   2. Classification")
    print("      - Embed text → use vector as features")
    print("      - Train classifier on embeddings")
    
    print("\n   3. Clustering")
    print("      - Group similar documents together")
    print("      - Topic modeling, deduplication")
    
    print("\n   4. Recommendation")
    print("      - 'Users who liked X also liked Y'")
    print("      - Based on embedding similarity")
    
    print("\n💡 Embedding Models vs Generative Models:")
    print("\n┌──────────────────────┬─────────────────┬──────────────────┐")
    print("│ Aspect               │ Embedding Model │ Generative Model │")
    print("├──────────────────────┼─────────────────┼──────────────────┤")
    print("│ Output               │ Vector          │ Text             │")
    print("│ Purpose              │ Similarity      │ Generation       │")
    print("│ Examples             │ text-embed-3    │ GPT-4            │")
    print("│ Use Case             │ Search, RAG     │ Chat, Q&A        │")
    print("│ Cost                 │ Cheap           │ Expensive        │")
    print("└──────────────────────┴─────────────────┴──────────────────┘")

def call_embedding_api(text):
    """
    Call Azure OpenAI Embeddings API
    Note: Requires separate deployment for embeddings model
    """
    # This is illustrative - requires embeddings deployment
    print("\n--- Embedding API Call (Conceptual) ---")
    print(f"Input: '{text}'")
    print("Output: [1536-dimensional vector]")
    print("  [0.0234, -0.0123, 0.0456, ..., -0.0089]")
    print("\n💡 Use embeddings to find similar text via cosine similarity")

def demo_tokenizer_types():
    """
    Explain different tokenizer types
    """
    print("\n\n" + "=" * 70)
    print("6. TOKENIZER TYPES: BPE vs SentencePiece")
    print("=" * 70)
    
    print("\n🔧 Byte Pair Encoding (BPE):")
    print("   - Used by: GPT series, many English models")
    print("   - Learns merges from character pairs")
    print("   - Example: 'playing' → 'play' + 'ing'")
    print("   - Good for: English and similar languages")
    
    print("\n🔧 SentencePiece:")
    print("   - Used by: T5, multilingual models")
    print("   - Language-agnostic (works on raw bytes)")
    print("   - Better for: Multilingual, non-space languages")
    print("   - Example: Handles Chinese without spaces")
    
    print("\n🔧 WordPiece:")
    print("   - Used by: BERT")
    print("   - Similar to BPE but different merge algorithm")
    
    print("\n💡 Why Different Tokenizers?")
    print("   - Each model has its own tokenizer!")
    print("   - GPT-4 tokenizer ≠ GPT-3.5 tokenizer ≠ Claude tokenizer")
    print("   - Always use matching tokenizer for token counting")

def demo_practical_tips():
    """
    Provide practical tips
    """
    print("\n\n" + "=" * 70)
    print("7. PRACTICAL TIPS & BEST PRACTICES")
    print("=" * 70)
    
    print("\n✅ DO:")
    print("   1. Count tokens before API calls (use tiktoken library)")
    print("   2. Leave buffer for output (if 4K limit, use max 3K input)")
    print("   3. Truncate long documents intelligently (keep important parts)")
    print("   4. Use cheaper models for simple tasks")
    print("   5. Monitor token usage in production")
    print("   6. Use embeddings for search (cheaper than generation)")
    
    print("\n❌ DON'T:")
    print("   1. Assume 1 word = 1 token (varies by language)")
    print("   2. Send entire documents blindly (check token count first)")
    print("   3. Forget output tokens count towards limit")
    print("   4. Use wrong tokenizer for token counting")
    print("   5. Ignore non-English token inflation (Chinese, Arabic, etc.)")
    
    print("\n🛠️  Token Optimization Strategies:")
    print("   - Summarize long context before sending")
    print("   - Use bullet points instead of prose (fewer tokens)")
    print("   - Remove unnecessary whitespace and formatting")
    print("   - Split large documents into chunks (process separately)")
    print("   - Cache embeddings (don't recompute for same text)")

def main():
    """
    Run all demonstrations
    """
    print("\n🚀 EMBEDDINGS, CONTEXT WINDOWS & TOKENIZATION - HANDS-ON LAB\n")
    
    try:
        demo_tokenization()
        input("\n➡️  Press Enter to continue...")
        
        demo_token_vs_word()
        input("\n➡️  Press Enter to continue...")
        
        demo_cost_implications()
        input("\n➡️  Press Enter to continue...")
        
        demo_context_window()
        input("\n➡️  Press Enter to continue...")
        
        demo_embeddings()
        input("\n➡️  Press Enter to continue...")
        
        demo_tokenizer_types()
        input("\n➡️  Press Enter to continue...")
        
        demo_practical_tips()
        
    except Exception as e:
        print(f"\n⚠️  Note: Some features require tiktoken library")
        print(f"   Install with: pip install tiktoken")
        print(f"   Error: {e}")
    
    # Summary
    print("\n\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("\n✓ Tokens: Subword units, not words (1 token ≈ 4 chars English)")
    print("✓ Tokenization: BPE, SentencePiece, WordPiece")
    print("✓ Context Window: Max tokens (input + output) model can handle")
    print("✓ Cost: Based on tokens, not words or characters")
    print("✓ Embeddings: Vector representations for similarity/search")
    print("✓ Different models have different tokenizers and context limits")
    print("✓ Always count tokens before API calls!")
    print("✓ Use embeddings for search, generative models for text")
    print("\n" + "=" * 70)
    
    print("\n📚 Quick Reference:")
    print("   • 1 token ≈ 4 characters (English)")
    print("   • 1 token ≈ ¾ word (English)")
    print("   • 100 tokens ≈ 75 words")
    print("   • 1,000 tokens ≈ 750 words")
    print("   • Always leave ~20% buffer for output")

if __name__ == "__main__":
    main()
```

### Step 3: Install Required Library

```cmd
pip install tiktoken
```

### Step 4: Configure and Run

Replace API credentials and run:

```cmd
python tokens_embeddings.py
```

---

## Expected Output

Interactive demonstrations of:
1. **How tokenization works** with various text types
2. **Token vs word count** comparisons
3. **Cost implications** of token-based pricing
4. **Context window limits** by model
5. **Embeddings** and their use cases
6. **Tokenizer types** (BPE, SentencePiece)
7. **Practical tips** for token optimization

---

## Key Concepts Deep Dive

### 1. How BPE Tokenization Works

```
Step 1: Start with characters
  "playing" → ['p', 'l', 'a', 'y', 'i', 'n', 'g']

Step 2: Learn frequent pairs
  Most common: 'p'+'l' = 'pl'
  Result: ['pl', 'a', 'y', 'i', 'n', 'g']

Step 3: Continue merging
  'pl'+'ay' = 'play'
  Result: ['play', 'i', 'n', 'g']

Step 4: Final merge
  'i'+'ng' = 'ing'
  Result: ['play', 'ing']  ← 2 tokens!
```

### 2. Token Count Formula (Approximation)

```
English:
  tokens ≈ characters / 4
  tokens ≈ words × 1.3

Non-English (Chinese):
  tokens ≈ characters × 2-3

Code:
  tokens ≈ characters / 3.5
```

### 3. Context Window Calculation

```
Total Context = System + Input + History + Output

Example with 4K context:
  System prompt:     200 tokens
  User message:      300 tokens
  Conversation:    1,000 tokens
  Model response:    500 tokens
  ───────────────────────────────
  Total used:      2,000 tokens
  Remaining:       2,096 tokens ✓
```

### 4. Embedding Similarity

```python
# Conceptual example
text1 = "dog"  → embedding1 = [0.2, 0.8, 0.1, ...]
text2 = "puppy" → embedding2 = [0.3, 0.7, 0.15, ...]
text3 = "car"  → embedding3 = [-0.5, 0.1, 0.9, ...]

# Cosine similarity
similarity(text1, text2) = 0.92  # High (similar meaning)
similarity(text1, text3) = 0.15  # Low (different meaning)
```

---

## Practical Examples

### Example 1: Count Tokens Before API Call

```python
import tiktoken

def safe_api_call(text, max_tokens=4000):
    encoding = tiktoken.encoding_for_model("gpt-4")
    token_count = len(encoding.encode(text))
    
    if token_count > max_tokens - 500:  # Leave buffer
        print(f"⚠️  Warning: {token_count} tokens (too long)")
        # Truncate or split
        return None
    else:
        print(f"✓ {token_count} tokens (safe)")
        # Make API call
        return call_api(text)
```

### Example 2: Optimize Token Usage

```python
# ❌ Verbose (more tokens)
prompt = """
Please analyze the following document and provide 
a comprehensive summary of all the key points that 
are mentioned within the text below:
"""

# ✅ Concise (fewer tokens)
prompt = "Summarize key points:"
```

---

## Common Issues & Solutions

| Problem | Cause | Solution |
|---------|-------|----------|
| API error: "max tokens exceeded" | Input + output > context limit | Truncate input, reduce max_tokens |
| High costs | Too many tokens | Use cheaper model, optimize prompts |
| Wrong token count | Wrong tokenizer | Use model-specific tokenizer |
| Slow search | Generating for each query | Use embeddings instead |
| Poor non-English support | English-optimized tokenizer | Use multilingual model |

---

## Reflection Questions

1. Why do LLMs use tokens instead of words?
2. How does token count affect cost?
3. What happens when you exceed the context window?
4. When would you use embeddings vs generative models?
5. Why do different models have different tokenizers?

---

## Bonus: Token Counting Script

```cmd
# Save this as count_tokens.py
python -c "import tiktoken; text=input('Enter text: '); enc=tiktoken.get_encoding('cl100k_base'); print(f'Tokens: {len(enc.encode(text))}')"
```

---

## Next Steps
- Proceed to Lab 2.7 to learn about hallucination, grounding, and bias
- Understand LLM limitations and mitigation strategies
