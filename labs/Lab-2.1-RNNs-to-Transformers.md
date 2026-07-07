# Lab 2.1: From RNNs to Transformers

## Objective
Understand why transformers replaced RNNs by observing the vanishing gradient problem and comparing processing times.

## Duration
~15 minutes

## Prerequisites
- Python 3.8+
- Azure OpenAI API key and endpoint

---

## Exercise: Comparing Sequential vs Parallel Processing

### Background
RNNs process text **sequentially** (word by word), which causes:
- **Vanishing gradients**: Information from early words fades away
- **Slow processing**: Cannot process words in parallel
- **Long-range dependency issues**: Hard to remember context from 100+ tokens ago

Transformers use **self-attention** to process all words simultaneously, solving these problems.

---

## Step-by-Step Instructions

### Step 1: Setup Your Environment

1. Open **Command Prompt** or **PowerShell**
2. Create a new folder for your lab:
```cmd
mkdir C:\genai-labs
cd C:\genai-labs
```

3. Create a Python file:
```cmd
notepad rnn_vs_transformer.py
```

### Step 2: Write the Comparison Code

Copy this code into `rnn_vs_transformer.py`:

```python
import time
import requests
import json

# Azure OpenAI Configuration
AZURE_ENDPOINT = "YOUR_AZURE_ENDPOINT"  # e.g., https://your-resource.openai.azure.com/
API_KEY = "YOUR_API_KEY"
DEPLOYMENT_NAME = "YOUR_DEPLOYMENT_NAME"  # e.g., gpt-4o

# Test sentences demonstrating long-range dependencies
sentences = [
    "The cat sat on the mat.",
    "The cat that the dog that the boy owned chased sat on the mat.",
    "In the beginning of the story, there was a cat. Many adventures happened. In the end, who sat on the mat?"
]

def call_azure_openai(prompt):
    """Call Azure OpenAI API"""
    url = f"{AZURE_ENDPOINT}/openai/deployments/{DEPLOYMENT_NAME}/chat/completions?api-version=2024-02-15-preview"
    
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY
    }
    
    data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Answer briefly."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 50
    }
    
    start_time = time.time()
    response = requests.post(url, headers=headers, json=data)
    end_time = time.time()
    
    if response.status_code == 200:
        result = response.json()
        answer = result['choices'][0]['message']['content']
        return answer, end_time - start_time
    else:
        return f"Error: {response.status_code}", 0

# Main demonstration
print("=" * 60)
print("RNN vs TRANSFORMER DEMONSTRATION")
print("=" * 60)
print("\nModern transformers (like GPT) handle long-range dependencies better than RNNs.\n")

for i, sentence in enumerate(sentences, 1):
    print(f"\n--- Test {i} ---")
    print(f"Input: {sentence}")
    print(f"Question: What sat on the mat?")
    
    prompt = f"{sentence}\n\nQuestion: What sat on the mat?\nAnswer:"
    answer, duration = call_azure_openai(prompt)
    
    print(f"Answer: {answer}")
    print(f"Time: {duration:.2f} seconds")
    
    # Explanation
    if i == 1:
        print("\n💡 Explanation: Simple sentence - both RNNs and Transformers handle this easily.")
    elif i == 2:
        print("\n💡 Explanation: Nested structure - RNNs struggle with nested dependencies,")
        print("   but Transformers use attention to connect 'cat' with 'sat' directly.")
    elif i == 3:
        print("\n💡 Explanation: Long-range dependency - RNNs suffer from vanishing gradients.")
        print("   Transformers maintain context across the entire sequence using attention.")

print("\n" + "=" * 60)
print("KEY TAKEAWAYS:")
print("=" * 60)
print("✓ RNNs process sequentially → slow and lose information over time")
print("✓ Transformers process in parallel → fast and maintain context")
print("✓ Attention mechanism allows direct connections between any words")
print("✓ This is why transformers replaced RNNs for NLP tasks!")
print("=" * 60)
```

### Step 3: Configure Your API Credentials

1. Replace `YOUR_AZURE_ENDPOINT` with your Azure OpenAI endpoint
2. Replace `YOUR_API_KEY` with your API key
3. Replace `YOUR_DEPLOYMENT_NAME` with your deployment name
4. Save the file (Ctrl+S)

### Step 4: Install Required Libraries

```cmd
pip install requests
```

### Step 5: Run the Exercise

```cmd
python rnn_vs_transformer.py
```

---

## Expected Output

You should see:
- Three test cases with increasing complexity
- How the transformer handles each case
- Processing times
- Explanations for each test

---

## Key Concepts Learned

### 1. **Sequential Processing (RNNs)**
- Process one word at a time: word₁ → word₂ → word₃
- Information flows through a chain
- Early information can "fade" (vanishing gradient problem)

### 2. **Parallel Processing (Transformers)**
- Process all words simultaneously
- Each word can "attend" to any other word
- No information loss due to distance

### 3. **Vanishing Gradient Problem**
- In RNNs, gradients get smaller as they backpropagate through time
- Makes it hard to learn long-range dependencies
- Why RNNs struggle with sentences longer than 50-100 words

### 4. **Why Transformers Won**
- ✅ Parallelization → faster training and inference
- ✅ Attention mechanism → better long-range dependencies
- ✅ No vanishing gradients → can handle much longer contexts
- ✅ Scale better → enabled GPT-3, GPT-4, and beyond

---

## Historical Timeline (Reference)

| Year | Milestone | Significance |
|------|-----------|--------------|
| 2013 | Word2Vec | Dense word embeddings |
| 2014 | Seq2Seq | RNN encoder-decoder architecture |
| 2017 | **Transformer** | "Attention is All You Need" paper |
| 2018 | ELMo | Contextualized embeddings (still LSTM-based) |
| 2018 | GPT-1 | First transformer decoder-only model |
| 2018 | BERT | Bidirectional transformer encoder |
| 2019 | GPT-2 | Scaled up to 1.5B parameters |
| 2020 | GPT-3 | 175B parameters, few-shot learning |
| 2023 | GPT-4 | Multimodal, advanced reasoning |
| 2024 | GPT-4o | Optimized, faster, multimodal |

---

## Reflection Questions

1. What happens to the answer quality as sentence complexity increases?
2. Why can't RNNs process words in parallel?
3. How does attention help with long-range dependencies?
4. Why was the transformer architecture revolutionary?

---

## Next Steps
- Proceed to Lab 2.2 to understand the transformer architecture in detail
- Explore how self-attention actually works mathematically
