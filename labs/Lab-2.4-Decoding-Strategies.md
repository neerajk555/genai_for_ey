# Lab 2.4: Decoding Strategies

## Objective
Understand and experiment with different decoding strategies to control LLM text generation.

## Duration
~15 minutes

## Prerequisites
- Python 3.8+
- Azure OpenAI API key and endpoint

---

## Exercise: Controlling Text Generation with Decoding Parameters

### Background
When LLMs generate text, they predict a **probability distribution** over all possible next tokens. Decoding strategies determine **how we select** the next token from this distribution.

**Key Parameters**:
- **Temperature**: Controls randomness (0 = deterministic, 2 = very random)
- **Top-k**: Consider only top k most likely tokens
- **Top-p (nucleus)**: Consider tokens with cumulative probability p
- **Penalties**: Discourage repetition

---

## Step-by-Step Instructions

### Step 1: Create Your Lab File

```cmd
cd C:\genai-labs
notepad decoding_strategies.py
```

### Step 2: Write the Decoding Exploration Code

```python
from openai import OpenAI

# Azure OpenAI Configuration
endpoint = "YOUR_ENDPOINT"
deployment_name = "YOUR_DEPLOYMENT_NAME"
api_key = "YOUR_API_KEY"

client = OpenAI(base_url=endpoint, api_key=api_key)

def call_azure_openai(prompt, temperature=0.7, top_p=1.0, max_completion_tokens=50, 
                      frequency_penalty=0, presence_penalty=0):
    """
    Call Azure OpenAI with specific decoding parameters
    """
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            top_p=top_p,
            max_completion_tokens=max_completion_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def demo_temperature():
    """
    Demonstrate temperature parameter effects
    """
    print("=" * 70)
    print("1. TEMPERATURE: Controlling Randomness")
    print("=" * 70)
    
    print("\nTemperature controls how 'creative' or 'deterministic' the model is:")
    print("  - temperature = 0   → Always picks most likely token (deterministic)")
    print("  - temperature = 0.7 → Balanced creativity (default)")
    print("  - temperature = 1.5 → Very creative/random")
    
    prompt = "Complete this story in 2 sentences: Once upon a time, there was a brave knight who"
    
    temperatures = [0, 0.7, 1.5]
    
    for temp in temperatures:
        print(f"\n--- Temperature = {temp} ---")
        print(f"Prompt: {prompt}")
        
        # Run twice to show consistency
        for i in range(2):
            response = call_azure_openai(prompt, temperature=temp, max_completion_tokens=50)
            print(f"  Run {i+1}: {response[:100]}...")
    
    print("\n💡 Key Observations:")
    print("   - temp=0: Both runs produce IDENTICAL output (deterministic)")
    print("   - temp=0.7: Both runs are different but reasonable")
    print("   - temp=1.5: Both runs are very different, more creative/random")
    
    print("\n📋 When to use each:")
    print("   • Temperature 0-0.3: Factual tasks (Q&A, classification, extraction)")
    print("   • Temperature 0.7-1.0: Creative tasks (writing, brainstorming)")
    print("   • Temperature 1.0-2.0: Very creative (poetry, unconventional ideas)")

def demo_top_k_top_p():
    """
    Demonstrate top-k and top-p sampling
    """
    print("\n\n" + "=" * 70)
    print("2. TOP-K and TOP-P SAMPLING")
    print("=" * 70)
    
    print("\nProblem: Even with temperature, model might pick very unlikely words")
    print("Solution: Limit the token choices\n")
    
    print("TOP-K Sampling:")
    print("  - Consider only the top K most likely tokens")
    print("  - Example: top_k=50 → choose from 50 most likely tokens")
    print("  - Fixed number of candidates")
    
    print("\nTOP-P (Nucleus) Sampling:")
    print("  - Consider tokens until cumulative probability reaches P")
    print("  - Example: top_p=0.9 → choose from smallest set covering 90% probability")
    print("  - Dynamic number of candidates (adapts to certainty)")
    
    prompt = "The capital of France is"
    
    print("\n--- Example 1: High Certainty Question ---")
    print(f"Prompt: {prompt}")
    
    # Top-p = 0.1 (very restrictive)
    response1 = call_azure_openai(prompt, temperature=0.7, top_p=0.1, max_completion_tokens=5)
    print(f"  top_p=0.1 (restrictive): {response1}")
    
    # Top-p = 1.0 (no restriction)
    response2 = call_azure_openai(prompt, temperature=0.7, top_p=1.0, max_completion_tokens=5)
    print(f"  top_p=1.0 (unrestricted): {response2}")
    
    print("\n💡 Key Point:")
    print("   - For certain questions: top_p doesn't matter much")
    print("   - For creative tasks: top_p controls diversity")
    
    print("\n📋 Practical Guidelines:")
    print("   • top_p = 0.1-0.5: Conservative, focused (Q&A, code)")
    print("   • top_p = 0.9-1.0: Diverse, creative (writing, brainstorming)")
    print("   • Common: Use top_p OR temperature, not both at extreme values")

def demo_greedy_vs_sampling():
    """
    Compare greedy decoding with sampling
    """
    print("\n\n" + "=" * 70)
    print("3. GREEDY DECODING vs SAMPLING")
    print("=" * 70)
    
    print("\nGreedy Decoding: Always pick the most probable token")
    print("  - Deterministic (same output every time)")
    print("  - Implemented as: temperature = 0")
    
    print("\nSampling: Probabilistically choose based on distribution")
    print("  - Non-deterministic (different outputs)")
    print("  - Implemented as: temperature > 0")
    
    prompt = "Write a creative product name for a smartwatch that tracks fitness and health:"
    
    print(f"\n--- Task: {prompt} ---\n")
    
    print("Greedy (temperature=0) - 3 runs:")
    for i in range(3):
        response = call_azure_openai(prompt, temperature=0, max_completion_tokens=10)
        print(f"  Run {i+1}: {response}")
    
    print("\nSampling (temperature=1.0) - 3 runs:")
    for i in range(3):
        response = call_azure_openai(prompt, temperature=1.0, max_completion_tokens=10)
        print(f"  Run {i+1}: {response}")
    
    print("\n💡 Key Observations:")
    print("   - Greedy: All 3 runs produce the SAME name")
    print("   - Sampling: All 3 runs produce DIFFERENT names")

def demo_repetition_penalties():
    """
    Demonstrate repetition, frequency, and presence penalties
    """
    print("\n\n" + "=" * 70)
    print("4. REPETITION PENALTIES")
    print("=" * 70)
    
    print("\nProblem: LLMs sometimes repeat words or phrases")
    print("Solution: Apply penalties to discourage repetition\n")
    
    print("FREQUENCY PENALTY (0.0 to 2.0):")
    print("  - Penalizes tokens based on how OFTEN they appeared")
    print("  - Higher frequency = stronger penalty")
    
    print("\nPRESENCE PENALTY (0.0 to 2.0):")
    print("  - Penalizes tokens if they appeared AT ALL (binary)")
    print("  - Encourages diverse vocabulary")
    
    prompt = "List 5 benefits of exercise:"
    
    print(f"\n--- Task: {prompt} ---\n")
    
    print("Without penalty (frequency_penalty=0, presence_penalty=0):")
    response1 = call_azure_openai(prompt, temperature=0.7, 
                                  frequency_penalty=0, presence_penalty=0,
                                  max_completion_tokens=100)
    print(f"{response1}\n")
    
    print("With penalties (frequency_penalty=0.5, presence_penalty=0.5):")
    response2 = call_azure_openai(prompt, temperature=0.7,
                                  frequency_penalty=0.5, presence_penalty=0.5,
                                  max_completion_tokens=100)
    print(f"{response2}\n")
    
    print("💡 Key Observations:")
    print("   - Without penalties: May reuse similar phrases")
    print("   - With penalties: More varied vocabulary and structure")
    
    print("\n📋 When to use:")
    print("   • frequency_penalty: Reduce word repetition (creative writing)")
    print("   • presence_penalty: Encourage topic diversity (lists, brainstorming)")
    print("   • Typical values: 0.1 to 0.8 (higher can make text incoherent)")

def demo_use_case_guide():
    """
    Provide practical guidance for different use cases
    """
    print("\n\n" + "=" * 70)
    print("5. CHOOSING THE RIGHT STRATEGY: Practical Guide")
    print("=" * 70)
    
    use_cases = [
        {
            "task": "Q&A / Factual Extraction",
            "temp": 0,
            "top_p": 0.1,
            "freq_pen": 0,
            "pres_pen": 0,
            "reason": "Need deterministic, accurate answers"
        },
        {
            "task": "Code Generation",
            "temp": 0,
            "top_p": 0.1,
            "freq_pen": 0,
            "pres_pen": 0,
            "reason": "Code must be correct and consistent"
        },
        {
            "task": "Classification / Labeling",
            "temp": 0,
            "top_p": 0.1,
            "freq_pen": 0,
            "pres_pen": 0,
            "reason": "Need consistent labels"
        },
        {
            "task": "Creative Writing / Stories",
            "temp": 0.9,
            "top_p": 0.95,
            "freq_pen": 0.3,
            "pres_pen": 0.3,
            "reason": "Want creativity and variety"
        },
        {
            "task": "Brainstorming / Idea Generation",
            "temp": 1.0,
            "top_p": 1.0,
            "freq_pen": 0.5,
            "pres_pen": 0.5,
            "reason": "Need diverse, novel ideas"
        },
        {
            "task": "Summarization",
            "temp": 0.3,
            "top_p": 0.5,
            "freq_pen": 0.2,
            "pres_pen": 0.2,
            "reason": "Balance accuracy and readability"
        },
        {
            "task": "Chatbot / Conversation",
            "temp": 0.7,
            "top_p": 0.9,
            "freq_pen": 0.2,
            "pres_pen": 0.2,
            "reason": "Natural, engaging, not repetitive"
        }
    ]
    
    print("\n| Use Case | Temperature | Top-P | Freq Penalty | Reason |")
    print("|----------|-------------|-------|--------------|--------|")
    for uc in use_cases:
        print(f"| {uc['task']:<28} | {uc['temp']:<11} | {uc['top_p']:<5} | {uc['freq_pen']:<12} | {uc['reason']:<30} |")
    
    print("\n💡 General Rules of Thumb:")
    print("   1. Factual/Deterministic tasks → Low temperature (0-0.3)")
    print("   2. Creative tasks → Higher temperature (0.7-1.2)")
    print("   3. Use top_p for dynamic control (usually 0.9-1.0)")
    print("   4. Add penalties (0.2-0.5) when you see repetition")
    print("   5. Don't use high temperature AND high top_p together")
    print("   6. Start conservative, increase if output too boring")

def demo_beam_search_note():
    """
    Explain beam search (not available in most APIs)
    """
    print("\n\n" + "=" * 70)
    print("6. BEAM SEARCH (Advanced - Not in standard API)")
    print("=" * 70)
    
    print("\nBeam Search: Keep multiple candidate sequences")
    print("  - Instead of picking 1 token, keep top K sequences")
    print("  - At each step, expand all K sequences")
    print("  - Keep best K complete sequences")
    
    print("\n📊 Example (beam_width=3):")
    print("  Step 1: 'The' → Keep 3 best: ['cat', 'dog', 'bird']")
    print("  Step 2: Expand each → Keep 3 best overall sequences")
    print("  Step 3: Continue until all beams complete")
    
    print("\n✅ Advantages:")
    print("   - More thorough exploration than greedy")
    print("   - Better for tasks with 'optimal' output (translation)")
    
    print("\n❌ Disadvantages:")
    print("   - Much slower (beam_width × compute)")
    print("   - Can produce generic/boring text")
    print("   - Not available in most LLM APIs (GPT, Claude)")
    
    print("\n💡 When to use (if available):")
    print("   - Machine translation")
    print("   - Text summarization with specific constraints")
    print("   - Tasks where quality > speed")

def main():
    """
    Run all demonstrations
    """
    print("\n🚀 DECODING STRATEGIES - HANDS-ON LAB\n")
    
    demo_temperature()
    input("\n➡️  Press Enter to continue to next section...")
    
    demo_top_k_top_p()
    input("\n➡️  Press Enter to continue to next section...")
    
    demo_greedy_vs_sampling()
    input("\n➡️  Press Enter to continue to next section...")
    
    demo_repetition_penalties()
    input("\n➡️  Press Enter to continue to next section...")
    
    demo_use_case_guide()
    
    demo_beam_search_note()
    
    # Summary
    print("\n\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("\n✓ Temperature: Controls randomness (0=deterministic, 1+=creative)")
    print("✓ Top-p: Dynamic token selection based on cumulative probability")
    print("✓ Top-k: Fixed number of candidate tokens (less common)")
    print("✓ Frequency penalty: Reduce repeated words")
    print("✓ Presence penalty: Encourage diverse vocabulary")
    print("✓ Greedy decoding: temperature=0 (deterministic)")
    print("✓ Sampling: temperature>0 (probabilistic)")
    print("✓ Choose strategy based on task requirements!")
    print("\n" + "=" * 70)
    
    print("\n📚 Quick Reference Card:")
    print("  Factual tasks → temp=0, top_p=0.1")
    print("  Creative tasks → temp=0.9, top_p=0.95, penalties=0.3")
    print("  Balanced → temp=0.7, top_p=0.9, penalties=0.2")

if __name__ == "__main__":
    main()
```

### Step 3: Configure Your API

Replace placeholders with your Azure OpenAI credentials and save.

### Step 4: Run the Lab

```cmd
python decoding_strategies.py
```

---

## Expected Output

Interactive demonstration of:
1. **Temperature** effects on creativity vs consistency
2. **Top-p sampling** for dynamic token selection
3. **Greedy vs sampling** comparison
4. **Repetition penalties** to reduce redundancy
5. **Practical guide** for choosing parameters by use case

---

## Key Concepts Deep Dive

### 1. How Decoding Works

```
Step 1: Model outputs probability distribution
  Token: "cat"   "dog"   "bird"  "the"   (50k+ tokens)
  Prob:  0.45    0.30    0.15    0.05    ...

Step 2: Apply temperature (if temp ≠ 1)
  P'(token) = exp(log(P(token)) / temperature)
  - temp < 1: Sharpens distribution (more confident)
  - temp > 1: Flattens distribution (more random)

Step 3: Apply top-p filtering
  - Sort tokens by probability (descending)
  - Keep tokens until cumulative prob ≥ p
  - Renormalize

Step 4: Sample from filtered distribution
  - If temp = 0: Pick highest probability (greedy)
  - If temp > 0: Sample probabilistically

Step 5: Apply penalties
  - Reduce probability of recently used tokens
  - frequency_penalty: proportional to count
  - presence_penalty: binary (yes/no)
```

### 2. Temperature Formula

```
P'_i = exp(logit_i / T) / Σ_j exp(logit_j / T)

Where:
  T = temperature
  logit_i = model's raw output for token i
  P'_i = adjusted probability
```

**Effect**:
- **T → 0**: Distribution becomes peaked (argmax)
- **T = 1**: No change (original distribution)
- **T → ∞**: Uniform distribution (completely random)

### 3. Top-p (Nucleus) Algorithm

```
1. Sort tokens by probability: P₁ ≥ P₂ ≥ P₃ ≥ ...
2. Find smallest k where: Σ(i=1 to k) P_i ≥ p
3. Keep only top k tokens
4. Renormalize: P'_i = P_i / Σ(kept tokens)
5. Sample from P'
```

### 4. Parameter Interactions

| Temperature | Top-p | Result |
|-------------|-------|--------|
| 0 | any | Deterministic (greedy) |
| 0.3 | 0.1 | Very focused, consistent |
| 0.7 | 0.9 | Balanced (recommended default) |
| 1.0 | 1.0 | Maximum randomness |
| 1.5 | 0.95 | Creative but not totally random |

**Warning**: Avoid `temp=2.0, top_p=1.0` → Too random, likely incoherent!

---

## Common Patterns

### Pattern 1: Deterministic Output
```python
temperature = 0
top_p = 0.1
# Use for: Q&A, classification, extraction
```

### Pattern 2: Creative Writing
```python
temperature = 0.9
top_p = 0.95
frequency_penalty = 0.3
presence_penalty = 0.3
# Use for: Stories, creative content
```

### Pattern 3: Conversational
```python
temperature = 0.7
top_p = 0.9
frequency_penalty = 0.2
presence_penalty = 0.2
# Use for: Chatbots, natural dialogue
```

---

## Reflection Questions

1. Why does temperature=0 always produce the same output?
2. When would you use top-p instead of top-k?
3. How do frequency_penalty and presence_penalty differ?
4. What happens if you set temperature=2.0 and top_p=1.0?
5. Which settings would you use for generating Python code?

---

## Troubleshooting Guide

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| Output too boring/repetitive | Temperature too low | Increase to 0.7-1.0 |
| Output incoherent/random | Temperature too high | Decrease to 0.5-0.8 |
| Same output every time | temp=0 (greedy) | Increase temp > 0 for variety |
| Too much word repetition | No penalties | Add frequency_penalty=0.3 |
| Output too long/rambling | No constraints | Add max_tokens, presence_penalty |

---

## Next Steps
- Proceed to Lab 2.5 to learn about instruction tuning and fine-tuning
- Understand how to align models for specific tasks
