# Lab 2.3: LLM Pretraining and Language Modelling Objectives

## Objective
Understand how LLMs are pretrained and what they learn through autoregressive and masked language modeling.

## Duration
~15 minutes

## Prerequisites
- Python 3.8+
- Azure OpenAI API key and endpoint

---

## Exercise: Exploring Pretraining Objectives

### Background
LLMs learn language patterns through **pretraining** on massive text corpora:
- **Autoregressive LM (GPT)**: Predict the next token
- **Masked LM (BERT)**: Predict hidden tokens
- **Scale Laws**: More data + more parameters = better performance

---

## Step-by-Step Instructions

### Step 1: Create Your Lab File

```cmd
cd C:\genai-labs
notepad pretraining_objectives.py
```

### Step 2: Write the Pretraining Exploration Code

```python
from openai import OpenAI
import time

# Azure OpenAI Configuration
endpoint = "YOUR_ENDPOINT"
deployment_name = "YOUR_DEPLOYMENT_NAME"
api_key = "YOUR_API_KEY"

client = OpenAI(base_url=endpoint, api_key=api_key)

def call_azure_openai(prompt, max_completion_tokens=50, temperature=0.7):
    """Call Azure OpenAI API"""
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=max_completion_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def demo_autoregressive_lm():
    """
    Demonstrate Autoregressive Language Modeling (GPT-style)
    """
    print("=" * 70)
    print("1. AUTOREGRESSIVE LANGUAGE MODELING (GPT-Style)")
    print("=" * 70)
    
    print("\nObjective: Predict the NEXT token given previous tokens")
    print("Training: Model sees: 'The cat sat on' → must predict 'the'")
    print("          Then: 'The cat sat on the' → must predict 'mat'")
    print("          (Left-to-right, one token at a time)\n")
    
    print("--- Example 1: Next Token Prediction ---")
    prefixes = [
        "The capital of France is",
        "To make a cake, you need flour, sugar, and",
        "The fastest way to learn programming is"
    ]
    
    for prefix in prefixes:
        print(f"\nInput: '{prefix}'")
        response = call_azure_openai(prefix, max_completion_tokens=5, temperature=0)
        print(f"Model predicts: '{response}'")
    
    print("\n💡 Key Points:")
    print("   - Model ONLY sees previous tokens (causal/autoregressive)")
    print("   - Cannot look ahead to future tokens")
    print("   - This is how GPT generates text: one token at a time")
    print("   - Training objective: Maximize P(token_n | token_1...token_n-1)")

def demo_masked_lm():
    """
    Demonstrate Masked Language Modeling (BERT-style)
    """
    print("\n\n" + "=" * 70)
    print("2. MASKED LANGUAGE MODELING (BERT-Style)")
    print("=" * 70)
    
    print("\nObjective: Predict MASKED tokens using context from BOTH sides")
    print("Training: 'The [MASK] sat on the mat' → must predict 'cat'")
    print("          (Bidirectional: sees words before AND after the mask)\n")
    
    print("--- Example: Fill-in-the-Blank Tasks ---")
    
    masked_sentences = [
        ("The ___ sat on the mat", "cat"),
        ("Python is a programming ___", "language"),
        ("The Earth revolves around the ___", "Sun")
    ]
    
    for masked, answer in masked_sentences:
        prompt = f"Fill in the blank with ONE word: '{masked}'\nAnswer with just the word."
        print(f"\nInput: '{masked}'")
        response = call_azure_openai(prompt, max_completion_tokens=5, temperature=0)
        print(f"Model predicts: '{response.strip()}'")
        print(f"Expected: '{answer}'")
    
    print("\n💡 Key Points:")
    print("   - Model sees FULL context (before and after mask)")
    print("   - Better for understanding tasks (classification, QA)")
    print("   - NOT suitable for generation (needs all tokens at once)")
    print("   - Training objective: Maximize P(masked_token | all_other_tokens)")

def demo_what_llms_learn():
    """
    Demonstrate what knowledge LLMs acquire during pretraining
    """
    print("\n\n" + "=" * 70)
    print("3. WHAT LLMs LEARN DURING PRETRAINING")
    print("=" * 70)
    
    print("\nLLMs learn multiple types of knowledge from pretraining data:\n")
    
    # Syntax
    print("--- A. Syntax (Grammar Rules) ---")
    prompt = "Is this grammatically correct? 'She go to store yesterday'\nAnswer: Yes or No, then explain briefly."
    response = call_azure_openai(prompt, max_tokens=50)
    print(f"Question: Is 'She go to store yesterday' grammatically correct?")
    print(f"Model: {response}\n")
    
    # Facts
    print("--- B. World Knowledge (Facts) ---")
    prompt = "Who wrote Romeo and Juliet? Answer with just the name."
    response = call_azure_openai(prompt, max_completion_tokens=10)
    print(f"Question: Who wrote Romeo and Juliet?")
    print(f"Model: {response}\n")
    
    # Reasoning Patterns
    print("--- C. Reasoning Patterns ---")
    prompt = "If all cats are animals, and Tom is a cat, is Tom an animal? Answer: Yes or No with brief reasoning."
    response = call_azure_openai(prompt, max_completion_tokens=50)
    print(f"Question: Logical reasoning test")
    print(f"Model: {response}\n")
    
    # Common Sense
    print("--- D. Common Sense ---")
    prompt = "If I drop a glass on a hard floor, what typically happens? Answer briefly."
    response = call_azure_openai(prompt, max_completion_tokens=30)
    print(f"Question: What happens if you drop glass on hard floor?")
    print(f"Model: {response}\n")
    
    print("💡 Key Points:")
    print("   ✓ Syntax: Grammar rules, sentence structure")
    print("   ✓ Facts: World knowledge from training data")
    print("   ✓ Reasoning: Patterns of logical thinking")
    print("   ✓ Common Sense: Everyday knowledge and expectations")

def demo_pretraining_data():
    """
    Explain pretraining data sources
    """
    print("\n\n" + "=" * 70)
    print("4. PRETRAINING DATA SOURCES")
    print("=" * 70)
    
    print("\nLarge Language Models are trained on diverse text from:")
    print("\n📚 Data Sources:")
    print("   - Web Crawls: Common Crawl, web pages (largest source)")
    print("   - Books: Fiction, non-fiction, academic texts")
    print("   - Code Repositories: GitHub, StackOverflow")
    print("   - Academic Papers: ArXiv, research publications")
    print("   - Wikipedia: Curated, factual content")
    print("   - News Articles: Current events, journalism")
    print("   - Conversations: Reddit, forums (filtered)")
    
    print("\n🎯 Quality Filtering is Critical:")
    print("   - Remove toxic/harmful content")
    print("   - Deduplicate (remove exact copies)")
    print("   - Filter low-quality text")
    print("   - Balance data sources")
    print("   - Respect copyright and privacy")
    
    print("\n📊 Scale (Example: GPT-3):")
    print("   - Training tokens: ~300 billion")
    print("   - Training data size: ~570 GB of text")
    print("   - Training time: Weeks/months on thousands of GPUs")
    print("   - Training cost: Millions of dollars")

def demo_scale_laws():
    """
    Explain scale laws
    """
    print("\n\n" + "=" * 70)
    print("5. SCALE LAWS: Size Matters")
    print("=" * 70)
    
    print("\nScale Laws (Kaplan et al., 2020): Performance improves predictably with:")
    print("\n1. Model Size (Parameters)")
    print("   - GPT-2: 1.5 billion parameters")
    print("   - GPT-3: 175 billion parameters")
    print("   - GPT-4: Estimated 1+ trillion parameters")
    
    print("\n2. Data Size (Tokens)")
    print("   - More diverse training data → better performance")
    print("   - Diminishing returns eventually")
    
    print("\n3. Compute (GPU hours)")
    print("   - Longer training → better optimization")
    print("   - Compute = Model_Size × Data_Size × Training_Time")
    
    print("\n⚖️ Trade-offs:")
    print("   - Bigger models = Better performance but slower & more expensive")
    print("   - More data = Better coverage but diminishing returns")
    print("   - More compute = Better training but higher cost")
    
    print("\n💡 Optimal Scaling (Chinchilla findings):")
    print("   - Models were often 'undertrained' (too little data for size)")
    print("   - Optimal: Scale model AND data proportionally")
    print("   - Example: 70B parameter model needs ~1.4 trillion tokens")

def demo_knowledge_cutoff():
    """
    Demonstrate knowledge cutoff implications
    """
    print("\n\n" + "=" * 70)
    print("6. KNOWLEDGE CUTOFF: The Staleness Problem")
    print("=" * 70)
    
    print("\nProblem: LLMs only know information from their training data!")
    
    print("\n--- Testing Knowledge Boundaries ---")
    
    # Historical event (should know)
    prompt1 = "Who was the president of the United States in 2020? Just the name."
    response1 = call_azure_openai(prompt1, max_completion_tokens=10)
    print(f"\nHistorical question: {prompt1}")
    print(f"Model: {response1}")
    print("✓ Model knows this (within training data)")
    
    # Check model's knowledge cutoff
    prompt2 = "What is your knowledge cutoff date? Answer briefly."
    response2 = call_azure_openai(prompt2, max_completion_tokens=30)
    print(f"\nMeta question: What is your knowledge cutoff?")
    print(f"Model: {response2}")
    
    print("\n💡 Practical Implications:")
    print("   ❌ Cannot answer questions about events after cutoff")
    print("   ❌ Cannot know about new products, companies, or people")
    print("   ❌ May have outdated information (APIs, best practices)")
    print("   ✅ Solution: RAG (Retrieval-Augmented Generation)")
    print("   ✅ Solution: Fine-tuning on recent data")
    print("   ✅ Solution: Function calling to fetch live data")

def main():
    """
    Run all demonstrations
    """
    print("\n🚀 LLM PRETRAINING & LANGUAGE MODELING - HANDS-ON LAB\n")
    
    demo_autoregressive_lm()
    time.sleep(1)
    
    demo_masked_lm()
    time.sleep(1)
    
    demo_what_llms_learn()
    time.sleep(1)
    
    demo_pretraining_data()
    
    demo_scale_laws()
    
    demo_knowledge_cutoff()
    
    # Summary
    print("\n\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("\n✓ Autoregressive LM (GPT): Predict next token (good for generation)")
    print("✓ Masked LM (BERT): Predict masked tokens (good for understanding)")
    print("✓ LLMs learn: syntax, facts, reasoning, and common sense")
    print("✓ Pretraining data: Web, books, code, academic papers")
    print("✓ Scale laws: Bigger models + more data + more compute = better")
    print("✓ Knowledge cutoff: Models only know data up to training date")
    print("✓ Quality > Quantity: Filtering and curation matter!")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
```

### Step 3: Configure Your API

Replace placeholders with your Azure OpenAI credentials and save.

### Step 4: Run the Lab

```cmd
python pretraining_objectives.py
```

---

## Expected Output

You'll see demonstrations of:
1. **Autoregressive prediction** - next token completion
2. **Masked language modeling** - fill-in-the-blank
3. **Types of knowledge** learned during pretraining
4. **Data sources** and quality filtering
5. **Scale laws** explanation
6. **Knowledge cutoff** demonstration

---

## Key Concepts Deep Dive

### 1. Pretraining vs Fine-tuning

| Stage | Objective | Data | Duration |
|-------|-----------|------|----------|
| **Pretraining** | Learn language patterns | Massive unlabeled text (TB) | Weeks/Months |
| **Fine-tuning** | Adapt to specific tasks | Labeled task data (GB) | Hours/Days |

### 2. Autoregressive vs Masked LM

| Aspect | Autoregressive (GPT) | Masked (BERT) |
|--------|---------------------|---------------|
| **Direction** | Left-to-right only | Bidirectional |
| **Training** | Predict next token | Predict masked tokens |
| **Best for** | Generation, completion | Classification, understanding |
| **Example** | "The cat" → predict "sat" | "The [MASK] sat" → predict "cat" |

### 3. What LLMs Learn

**Syntax**: 
- Grammatical rules
- Sentence structure
- Language conventions

**Facts** (stored as statistical patterns, not database):
- Historical events
- Scientific knowledge
- Cultural information

**Reasoning Patterns**:
- Logical inference
- Mathematical patterns
- Problem-solving strategies

**Common Sense**:
- Physical world understanding
- Social norms
- Cause-effect relationships

### 4. Scale Laws Formula (Simplified)

```
Performance ∝ N^α × D^β × C^γ

Where:
N = Number of parameters (model size)
D = Dataset size (number of tokens)
C = Compute (GPU hours)
α, β, γ = Scaling exponents (~0.1-0.3)
```

**Key Insight**: Returns are predictable but diminishing!

---

## Reflection Questions

1. Why is autoregressive modeling good for text generation?
2. What's the advantage of bidirectional context in BERT?
3. How does model size affect performance and cost?
4. Why is knowledge cutoff a problem for real-world applications?
5. How would you address outdated information in an LLM?

---

## Common Misconceptions

❌ **"LLMs memorize the entire internet"**
- ✅ Actually: Learn statistical patterns, not exact copies

❌ **"Bigger is always better"**
- ✅ Actually: Need to balance size, data, and compute optimally

❌ **"LLMs understand language like humans"**
- ✅ Actually: Pattern matching, no true understanding

❌ **"Pretraining is enough for production use"**
- ✅ Actually: Need alignment, fine-tuning, or RAG for specific tasks

---

## Next Steps
- Proceed to Lab 2.4 to explore decoding strategies
- Learn how to control LLM output generation
