# Lab 2.2: Transformer Architecture - Conceptual Recap

## Objective
Understand the core components of transformer architecture through hands-on exploration of attention mechanisms.

## Duration
~15 minutes

## Prerequisites
- Python 3.8+
- Azure OpenAI API key and endpoint

---

## Exercise: Visualizing Self-Attention and Token Flow

### Background
The transformer architecture has several key components:
- **Self-Attention**: Allows each token to "look at" all other tokens
- **Query, Key, Value (Q, K, V)**: Three representations used to compute attention
- **Multi-Head Attention**: Multiple attention mechanisms running in parallel
- **Positional Encoding**: Adds position information to tokens

---

## Step-by-Step Instructions

### Step 1: Create Your Lab File

1. Open Command Prompt:
```cmd
cd C:\genai-labs
notepad transformer_architecture.py
```

### Step 2: Write the Attention Exploration Code

```python
from openai import OpenAI
import numpy as np

# Azure OpenAI Configuration
endpoint = "YOUR_ENDPOINT"
deployment_name = "YOUR_DEPLOYMENT_NAME"
api_key = "YOUR_API_KEY"

client = OpenAI(base_url=endpoint, api_key=api_key)

def simple_attention_demo():
    """
    Demonstrate attention mechanism conceptually
    """
    print("=" * 70)
    print("UNDERSTANDING SELF-ATTENTION")
    print("=" * 70)
    
    sentence = "The cat sat on the mat"
    tokens = sentence.split()
    
    print(f"\nInput sentence: '{sentence}'")
    print(f"Tokens: {tokens}")
    
    print("\n--- How Self-Attention Works ---")
    print("\nFor each token, we compute attention scores with ALL other tokens:")
    print("This tells us 'how much should this token pay attention to each other token?'\n")
    
    # Simulated attention weights (in real transformers, these are learned)
    # Each row = one token's attention to all tokens
    attention_matrix = np.array([
        [0.7, 0.1, 0.1, 0.05, 0.03, 0.02],  # The
        [0.1, 0.6, 0.2, 0.05, 0.03, 0.02],  # cat
        [0.1, 0.2, 0.5, 0.1, 0.05, 0.05],   # sat
        [0.05, 0.05, 0.3, 0.4, 0.1, 0.1],   # on
        [0.05, 0.05, 0.1, 0.2, 0.5, 0.1],   # the
        [0.05, 0.3, 0.2, 0.1, 0.1, 0.25],   # mat
    ])
    
    print("Attention Matrix (rows = query token, columns = what it attends to):")
    print("\n        ", "  ".join([f"{t:>6}" for t in tokens]))
    for i, token in enumerate(tokens):
        print(f"{token:>6}: ", "  ".join([f"{attention_matrix[i][j]:.2f}" for j in range(len(tokens))]))
    
    print("\n💡 Key Observations:")
    print("   - Each token attends most to itself (diagonal values are high)")
    print("   - 'cat' attends to 'The' and 'sat' (subject-verb relationship)")
    print("   - 'sat' attends to 'cat' and 'on' (verb connecting subject and preposition)")
    print("   - 'mat' attends to 'cat' (what's sitting on it)")

def call_azure_openai_with_attention(prompt, max_tokens=100):
    """
    Call Azure OpenAI and demonstrate token-by-token generation
    """
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=100,
            temperature=0.7
        )
        
        answer = response.choices[0].message.content
        return answer, response.usage.model_dump() if hasattr(response, 'usage') else {}
    except Exception as e:
        return f"Error: {e}", {}

def multi_head_attention_demo():
    """
    Demonstrate multi-head attention concept
    """
    print("\n\n" + "=" * 70)
    print("MULTI-HEAD ATTENTION")
    print("=" * 70)
    
    print("\nInstead of one attention mechanism, transformers use MULTIPLE heads.")
    print("Each head can learn different types of relationships:\n")
    
    sentence = "The quick brown fox jumps over the lazy dog"
    
    print(f"Example: '{sentence}'\n")
    print("Head 1 (Syntactic): Focuses on grammar relationships")
    print("  - 'fox' → 'The', 'quick', 'brown' (adjectives modifying noun)")
    print("  - 'jumps' → 'fox' (subject-verb agreement)")
    
    print("\nHead 2 (Semantic): Focuses on meaning relationships")
    print("  - 'quick' → 'fox', 'jumps' (fast movement)")
    print("  - 'lazy' → 'dog' (attribute of dog)")
    
    print("\nHead 3 (Positional): Focuses on nearby words")
    print("  - Each word attends most to its immediate neighbors")
    
    print("\n💡 Key Point: Multiple heads = Multiple perspectives on the same text!")

def demonstrate_encoder_decoder_types():
    """
    Demonstrate different transformer architectures
    """
    print("\n\n" + "=" * 70)
    print("ENCODER vs DECODER vs ENCODER-DECODER")
    print("=" * 70)
    
    print("\n1. ENCODER-ONLY (BERT)")
    print("   - Reads ENTIRE input at once (bidirectional)")
    print("   - Best for: Classification, understanding tasks")
    print("   - Example task: Is this sentence positive or negative?")
    
    # Quick sentiment analysis demo
    prompt1 = "Classify sentiment as positive or negative: 'I love this product!'"
    print(f"\n   Task: {prompt1}")
    response1, _ = call_azure_openai_with_attention(prompt1, max_tokens=10)
    print(f"   Answer: {response1}")
    
    print("\n2. DECODER-ONLY (GPT)")
    print("   - Reads ONLY previous tokens (autoregressive)")
    print("   - Best for: Text generation, completion")
    print("   - Example task: Complete this sentence")
    
    prompt2 = "Complete this sentence with 5 words: 'The best way to learn programming is'"
    print(f"\n   Task: {prompt2}")
    response2, _ = call_azure_openai_with_attention(prompt2, max_tokens=10)
    print(f"   Answer: {response2}")
    
    print("\n3. ENCODER-DECODER (T5)")
    print("   - Encoder reads input, decoder generates output")
    print("   - Best for: Translation, summarization")
    print("   - Example task: Translate English to French")

def positional_encoding_demo():
    """
    Explain positional encoding
    """
    print("\n\n" + "=" * 70)
    print("POSITIONAL ENCODING")
    print("=" * 70)
    
    print("\nProblem: Self-attention has no sense of word ORDER!")
    print("'dog bites man' vs 'man bites dog' would look the same without positions.\n")
    
    print("Solution: Add positional information to each token")
    
    sentence1 = "The cat chased the mouse"
    sentence2 = "The mouse chased the cat"
    
    print(f"\nSentence 1: '{sentence1}'")
    print("Positions:   0    1     2      3    4")
    
    print(f"\nSentence 2: '{sentence2}'")
    print("Positions:   0    1      2     3   4")
    
    print("\n💡 Key Point: Position encoding lets the model know:")
    print("   - 'cat' at position 1 vs position 4 are different")
    print("   - Word order matters for meaning!")
    
    print("\nTwo types of positional encoding:")
    print("   - Sinusoidal: Fixed mathematical formula (original transformer)")
    print("   - Learned: Model learns positions during training (modern LLMs)")

def main():
    """
    Run all demonstrations
    """
    print("\n🚀 TRANSFORMER ARCHITECTURE - HANDS-ON LAB\n")
    
    # Demo 1: Self-Attention
    simple_attention_demo()
    
    # Demo 2: Multi-Head Attention
    multi_head_attention_demo()
    
    # Demo 3: Architecture Types
    demonstrate_encoder_decoder_types()
    
    # Demo 4: Positional Encoding
    positional_encoding_demo()
    
    # Summary
    print("\n\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("\n✓ Self-Attention: Each token attends to all others (Q, K, V mechanism)")
    print("✓ Multi-Head: Multiple attention patterns in parallel")
    print("✓ Encoder-Only (BERT): Bidirectional, best for understanding")
    print("✓ Decoder-Only (GPT): Autoregressive, best for generation")
    print("✓ Encoder-Decoder (T5): Both, best for transformation tasks")
    print("✓ Positional Encoding: Adds word order information")
    print("✓ Layer Norm + Residuals: Helps training stability")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
```

### Step 3: Configure API Credentials

1. Replace `YOUR_AZURE_ENDPOINT`, `YOUR_API_KEY`, and `YOUR_DEPLOYMENT_NAME`
2. Save the file

### Step 4: Install Dependencies

```cmd
pip install requests numpy
```

### Step 5: Run the Lab

```cmd
python transformer_architecture.py
```

---

## Expected Output

You should see:
1. **Attention matrix** showing how tokens attend to each other
2. **Multi-head attention** explanation with examples
3. **Live API calls** demonstrating encoder-only vs decoder-only architectures
4. **Positional encoding** importance demonstration

---

## Key Concepts Deep Dive

### 1. Query, Key, Value (Q, K, V)

**Intuition**: Like a database lookup
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I have to offer?"
- **Value (V)**: "What actual information do I contain?"

**How it works**:
1. Each token creates Q, K, V vectors
2. Attention score = Q · K^T (dot product)
3. Apply softmax to get attention weights
4. Weighted sum of Values = output

### 2. Scaled Dot-Product Attention Formula

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

- **Q·K^T**: Similarity between query and all keys
- **√d_k**: Scaling factor (prevents large values)
- **softmax**: Converts to probabilities (sum to 1)
- **·V**: Weight the values by attention scores

### 3. Multi-Head Attention

Why multiple heads?
- Different heads learn different patterns
- Head 1: Syntax (grammar relationships)
- Head 2: Semantics (meaning relationships)  
- Head 3: Positional (nearby words)
- etc.

Then **concatenate** all heads and project back.

### 4. Self-Attention vs Cross-Attention

**Self-Attention**: Token attends to tokens in same sequence
- Example: In "The cat sat", 'sat' attends to 'cat'
- Used in: Encoder and Decoder

**Cross-Attention**: Token attends to different sequence
- Example: Translation - French output attends to English input
- Used in: Encoder-Decoder models (connecting encoder to decoder)

### 5. Architecture Comparison

| Type | Direction | Use Case | Examples |
|------|-----------|----------|----------|
| **Encoder-Only** | Bidirectional | Understanding, Classification | BERT, RoBERTa |
| **Decoder-Only** | Left-to-right | Generation, Completion | GPT, LLaMA, Claude |
| **Encoder-Decoder** | Both | Translation, Summarization | T5, BART |

---

## Reflection Questions

1. How does attention help the model understand context?
2. Why do we need multiple attention heads?
3. When would you use BERT vs GPT?
4. Why is positional encoding necessary?

---

## Bonus: Token Flow Through Transformer Stack

```
Input Text: "Hello world"
      ↓
[1] Tokenization: ["Hello", "world"]
      ↓
[2] Embedding: Convert to vectors [768-dim]
      ↓
[3] Positional Encoding: Add position info
      ↓
[4] Layer 1: Multi-Head Attention → Add & Norm → FFN → Add & Norm
      ↓
[5] Layer 2: Same as Layer 1
      ↓
    ... (repeat for N layers: GPT-4 has ~100 layers)
      ↓
[N] Output: Probability distribution over vocabulary
      ↓
[7] Decoding: Select next token
```

---

## Next Steps
- Proceed to Lab 2.3 to understand pretraining objectives
- Explore how transformers are trained on massive datasets
