# Lab 2.5: Instruction Tuning, Fine-Tuning, and Alignment

## Objective
Understand the difference between base models and aligned models, and learn when to use prompt engineering vs fine-tuning vs RAG.

## Duration
~15 minutes

## Prerequisites
- Python 3.8+
- Azure OpenAI API key and endpoint

---

## Exercise: Exploring Model Alignment and Decision Framework

### Background
**Base models** (after pretraining) are poor at following instructions. They complete text but don't "help."

**Alignment** makes models:
- Follow instructions reliably
- Refuse harmful requests
- Provide helpful, honest, harmless responses

**Techniques**:
- **Supervised Fine-Tuning (SFT)**: Train on instruction-response pairs
- **RLHF**: Reward models for good behavior
- **DPO**: Simpler alternative to RLHF
- **LoRA/QLoRA**: Parameter-efficient fine-tuning

---

## Step-by-Step Instructions

### Step 1: Create Your Lab File

```cmd
cd C:\genai-labs
notepad alignment_finetuning.py
```

### Step 2: Write the Alignment Exploration Code

```python
import requests
import json

# Azure OpenAI Configuration
AZURE_ENDPOINT = "YOUR_AZURE_ENDPOINT"
API_KEY = "YOUR_API_KEY"
DEPLOYMENT_NAME = "YOUR_DEPLOYMENT_NAME"

def call_azure_openai(messages, temperature=0.7, max_tokens=150):
    """Call Azure OpenAI API"""
    url = f"{AZURE_ENDPOINT}/openai/deployments/{DEPLOYMENT_NAME}/chat/completions?api-version=2024-02-15-preview"
    
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY
    }
    
    data = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['message']['content']
    else:
        return f"Error: {response.status_code}"

def demo_base_vs_aligned():
    """
    Demonstrate difference between base and instruction-tuned models
    """
    print("=" * 70)
    print("1. BASE MODEL vs INSTRUCTION-TUNED MODEL")
    print("=" * 70)
    
    print("\nBASE MODEL (after pretraining only):")
    print("  - Trained to complete text")
    print("  - Doesn't follow instructions well")
    print("  - Continues the prompt rather than answering")
    
    print("\nINSTRUCTION-TUNED MODEL (after SFT/RLHF):")
    print("  - Trained to follow instructions")
    print("  - Provides helpful responses")
    print("  - Refuses harmful requests")
    
    # Simulate base model behavior vs instruction-tuned
    print("\n--- Example 1: Simple Question ---")
    question = "What is the capital of France?"
    
    print(f"\nUser: {question}")
    print("\nBase model (text completion mode):")
    print("  'What is the capital of France? The capital of Germany is Berlin.'")
    print("  'What is the capital of France? What is the capital of Spain?'")
    print("  ❌ Just continues the pattern, doesn't answer!")
    
    print("\nInstruction-tuned model:")
    response = call_azure_openai([
        {"role": "user", "content": question}
    ], temperature=0, max_tokens=20)
    print(f"  {response}")
    print("  ✅ Actually answers the question!")
    
    # Example 2: Harmful request
    print("\n--- Example 2: Safety Alignment ---")
    harmful_request = "How do I hack into someone's email account?"
    
    print(f"\nUser: {harmful_request}")
    print("\nBase model (no safety training):")
    print("  'Here are the steps to hack an email: First, try phishing...'")
    print("  ❌ Would provide harmful information!")
    
    print("\nInstruction-tuned model with RLHF:")
    response = call_azure_openai([
        {"role": "user", "content": harmful_request}
    ], temperature=0.7, max_tokens=100)
    print(f"  {response}")
    print("  ✅ Refuses and explains why!")

def demo_supervised_finetuning():
    """
    Explain Supervised Fine-Tuning (SFT)
    """
    print("\n\n" + "=" * 70)
    print("2. SUPERVISED FINE-TUNING (SFT)")
    print("=" * 70)
    
    print("\nObjective: Train model to follow instructions")
    
    print("\n📚 Training Data Format (Instruction-Response Pairs):")
    print("""
    Example 1:
      Instruction: "Translate to French: Hello, how are you?"
      Response: "Bonjour, comment allez-vous ?"
    
    Example 2:
      Instruction: "Summarize this article: [article text]"
      Response: "[concise summary]"
    
    Example 3:
      Instruction: "Write a Python function to calculate factorial"
      Response: "[Python code with explanation]"
    """)
    
    print("📊 SFT Process:")
    print("  1. Collect high-quality instruction-response pairs (thousands)")
    print("  2. Fine-tune pretrained model on this data")
    print("  3. Model learns to: read instruction → generate appropriate response")
    print("  4. Typical dataset size: 10K - 100K examples")
    
    print("\n✅ Result: Model becomes an instruction-following assistant!")

def demo_rlhf_intuition():
    """
    Explain RLHF at intuitive level
    """
    print("\n\n" + "=" * 70)
    print("3. RLHF (Reinforcement Learning from Human Feedback)")
    print("=" * 70)
    
    print("\nProblem: SFT alone isn't enough for nuanced behavior")
    print("Solution: Use human preferences to guide model behavior\n")
    
    print("📋 RLHF Process (3 Steps):")
    
    print("\n  STEP 1: Supervised Fine-Tuning")
    print("    - Start with SFT model (instruction-following)")
    
    print("\n  STEP 2: Reward Model Training")
    print("    - Show humans multiple model outputs for same prompt")
    print("    - Humans rank outputs: A > B > C > D")
    print("    - Train a 'reward model' to predict human preferences")
    
    print("\n  STEP 3: RL Optimization (PPO - Proximal Policy Optimization)")
    print("    - Generate outputs with model")
    print("    - Reward model scores each output")
    print("    - Adjust model to maximize reward")
    print("    - Repeat thousands of times")
    
    print("\n💡 Intuition:")
    print("  - Like training a dog with treats!")
    print("  - Model tries different responses")
    print("  - Gets 'reward' for good responses")
    print("  - Learns to produce high-reward (human-preferred) outputs")
    
    print("\n✅ Result: Model aligns with human values and preferences")
    
    print("\n⚠️  Challenges:")
    print("  - Complex to implement (reward model + RL training)")
    print("  - Expensive (needs lots of human feedback)")
    print("  - Can be unstable (RL training is tricky)")

def demo_dpo():
    """
    Explain Direct Preference Optimization (DPO)
    """
    print("\n\n" + "=" * 70)
    print("4. DPO (Direct Preference Optimization)")
    print("=" * 70)
    
    print("\nDPO is a simpler alternative to RLHF")
    
    print("\n📊 Comparison:")
    print("\n  RLHF:")
    print("    1. Train reward model (separate model)")
    print("    2. Use RL (PPO) to optimize (complex)")
    print("    3. Two-stage training")
    
    print("\n  DPO:")
    print("    1. Skip reward model!")
    print("    2. Directly optimize on preference data (simpler)")
    print("    3. One-stage training")
    
    print("\n💡 How DPO Works:")
    print("  - Given prompt and two responses: preferred vs rejected")
    print("  - Train model to increase probability of preferred response")
    print("  - Decrease probability of rejected response")
    print("  - No separate reward model needed!")
    
    print("\n✅ Advantages of DPO:")
    print("  - Simpler: No reward model, no RL")
    print("  - Faster: One training stage instead of two")
    print("  - More stable: No RL instabilities")
    print("  - Competitive results with RLHF")
    
    print("\n📈 Trend: Many recent models use DPO (Llama 3, Mistral)")

def demo_lora_qlora():
    """
    Explain Parameter-Efficient Fine-Tuning
    """
    print("\n\n" + "=" * 70)
    print("5. PARAMETER-EFFICIENT FINE-TUNING: LoRA & QLoRA")
    print("=" * 70)
    
    print("\nProblem: Fine-tuning large models is expensive")
    print("  - GPT-3 (175B parameters): Needs ~700GB GPU memory!")
    print("  - Full fine-tuning updates ALL parameters")
    print("  - Slow, expensive, requires massive hardware")
    
    print("\nSolution: LoRA (Low-Rank Adaptation)")
    
    print("\n💡 Key Idea:")
    print("  - Don't update original weights W")
    print("  - Add small 'adapter' matrices: W + A×B")
    print("  - Only train A and B (much smaller!)")
    print("  - Example: Train 0.1% of parameters instead of 100%")
    
    print("\n📊 LoRA Benefits:")
    print("  ✓ 100x fewer parameters to train")
    print("  ✓ 10x less memory needed")
    print("  ✓ Faster training")
    print("  ✓ Multiple adapters for different tasks")
    print("  ✓ Can switch adapters at inference")
    
    print("\n🔧 QLoRA (Quantized LoRA):")
    print("  - LoRA + quantization (4-bit weights)")
    print("  - Fine-tune 70B model on single consumer GPU!")
    print("  - Example: 70B model normally needs 140GB, QLoRA needs ~24GB")
    
    print("\n📈 Practical Impact:")
    print("  - LoRA democratized fine-tuning")
    print("  - Can fine-tune large models on consumer hardware")
    print("  - Very popular for domain-specific customization")

def demo_decision_framework():
    """
    Provide decision framework: prompt vs fine-tune vs RAG
    """
    print("\n\n" + "=" * 70)
    print("6. DECISION FRAMEWORK: Prompt vs Fine-Tune vs RAG")
    print("=" * 70)
    
    print("\nThree main approaches to customize LLM behavior:\n")
    
    # Scenario 1
    print("--- Scenario 1: Need model to know recent product documentation ---")
    print("  ❌ Fine-tuning: Expensive, needs retraining for updates")
    print("  ✅ RAG: Retrieve relevant docs, include in prompt")
    print("  Reason: Information changes frequently, need real-time access\n")
    
    # Scenario 2
    print("--- Scenario 2: Need model to follow specific output format ---")
    print("  ✅ Prompt Engineering: Add format instructions to prompt")
    print("  ❌ Fine-tuning: Overkill for simple formatting")
    print("  Reason: Can be achieved with clear instructions\n")
    
    # Scenario 3
    print("--- Scenario 3: Need model to adopt specific writing style ---")
    print("  🤔 Prompt Engineering: Try first with few-shot examples")
    print("  ✅ Fine-tuning: If prompt doesn't work consistently")
    print("  Reason: Style is complex, may need many examples\n")
    
    # Scenario 4
    print("--- Scenario 4: Need model to classify industry-specific texts ---")
    print("  ✅ Fine-tuning (with LoRA): Best accuracy for domain-specific task")
    print("  🤔 Few-shot prompting: Can work if classes are simple")
    print("  Reason: Domain-specific knowledge, benefits from training\n")
    
    print("\n📋 Decision Matrix:")
    print("\n┌─────────────────────┬──────────────────┬───────────────┬─────────┐")
    print("│ Need                │ Prompt Engineer  │ Fine-Tune     │ RAG     │")
    print("├─────────────────────┼──────────────────┼───────────────┼─────────┤")
    print("│ New facts/docs      │ ✗ (context limit)│ ✗ (expensive) │ ✓✓✓     │")
    print("│ Format control      │ ✓✓✓              │ ✗ (overkill)  │ ✗       │")
    print("│ Style/tone          │ ✓✓ (few-shot)    │ ✓✓✓           │ ✗       │")
    print("│ Domain expertise    │ ✓ (if simple)    │ ✓✓✓           │ ✓✓      │")
    print("│ Task-specific       │ ✓✓ (few-shot)    │ ✓✓✓           │ ✗       │")
    print("│ Low latency         │ ✓✓✓              │ ✓✓            │ ✓       │")
    print("│ Low cost            │ ✓✓✓              │ ✗             │ ✓✓      │")
    print("│ Easy to update      │ ✓✓✓              │ ✗             │ ✓✓✓     │")
    print("└─────────────────────┴──────────────────┴───────────────┴─────────┘")
    
    print("\n💡 General Principle:")
    print("  1. START: Prompt engineering (cheapest, fastest)")
    print("  2. IF insufficient: Try RAG (if knowledge-intensive)")
    print("  3. IF still insufficient: Fine-tuning (most expensive)")
    print("  4. Can COMBINE: RAG + Fine-tuned model = best of both!")

def demo_practical_example():
    """
    Show practical example of prompt engineering
    """
    print("\n\n" + "=" * 70)
    print("7. HANDS-ON: Prompt Engineering vs Fine-Tuning Example")
    print("=" * 70)
    
    print("\nTask: Make model respond in a specific format (JSON)")
    
    # Bad prompt
    print("\n--- Approach 1: Unclear Prompt ---")
    bad_prompt = "Tell me about Python"
    print(f"Prompt: '{bad_prompt}'")
    response1 = call_azure_openai([
        {"role": "user", "content": bad_prompt}
    ], temperature=0, max_tokens=100)
    print(f"Response: {response1[:100]}...")
    print("❌ Free-form text, not JSON")
    
    # Good prompt
    print("\n--- Approach 2: Clear Instruction (Prompt Engineering) ---")
    good_prompt = """Provide information about Python programming language.

Output format (JSON):
{
  "name": "language name",
  "type": "programming language type",
  "features": ["feature1", "feature2", "feature3"],
  "use_cases": ["use_case1", "use_case2"]
}

Respond ONLY with valid JSON, no additional text."""
    
    print(f"Prompt: [Detailed format instruction with JSON schema]")
    response2 = call_azure_openai([
        {"role": "user", "content": good_prompt}
    ], temperature=0, max_tokens=200)
    print(f"Response:\n{response2}")
    print("✅ Properly formatted JSON!")
    
    print("\n💡 Key Takeaway:")
    print("  - Prompt engineering can solve MANY tasks")
    print("  - Fine-tuning only needed when prompting fails consistently")
    print("  - Always try prompt engineering first (cheaper & faster)")

def main():
    """
    Run all demonstrations
    """
    print("\n🚀 INSTRUCTION TUNING, FINE-TUNING & ALIGNMENT - HANDS-ON LAB\n")
    
    demo_base_vs_aligned()
    input("\n➡️  Press Enter to continue...")
    
    demo_supervised_finetuning()
    input("\n➡️  Press Enter to continue...")
    
    demo_rlhf_intuition()
    input("\n➡️  Press Enter to continue...")
    
    demo_dpo()
    input("\n➡️  Press Enter to continue...")
    
    demo_lora_qlora()
    input("\n➡️  Press Enter to continue...")
    
    demo_decision_framework()
    input("\n➡️  Press Enter to continue...")
    
    demo_practical_example()
    
    # Summary
    print("\n\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("\n✓ Base models need alignment to follow instructions")
    print("✓ SFT: Train on instruction-response pairs")
    print("✓ RLHF: Use human feedback + RL (complex but powerful)")
    print("✓ DPO: Simpler alternative to RLHF (trending)")
    print("✓ LoRA/QLoRA: Parameter-efficient fine-tuning")
    print("✓ Decision Framework:")
    print("    1. Prompt Engineering → Try first (cheapest)")
    print("    2. RAG → For knowledge-intensive tasks")
    print("    3. Fine-Tuning → Last resort (most expensive)")
    print("✓ Can combine approaches: RAG + Fine-tuned model")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
```

### Step 3: Configure Your API

Replace placeholders with your credentials and save.

### Step 4: Run the Lab

```cmd
python alignment_finetuning.py
```

---

## Expected Output

You'll see:
1. **Comparison** between base and aligned models
2. **SFT process** explanation
3. **RLHF intuition** (reward model + RL)
4. **DPO** as simpler alternative
5. **LoRA/QLoRA** for efficient fine-tuning
6. **Decision framework** for choosing the right approach
7. **Practical example** of prompt engineering vs fine-tuning

---

## Key Concepts Deep Dive

### Alignment Stages

```
[Pretraining]
    ↓
  Base Model (completes text, doesn't follow instructions)
    ↓
[Supervised Fine-Tuning (SFT)]
    ↓
  Instruction-Following Model
    ↓
[RLHF / DPO]
    ↓
  Aligned Model (helpful, harmless, honest)
```

### When to Use Each Approach

| Approach | Use When | Cost | Speed | Flexibility |
|----------|----------|------|-------|-------------|
| **Prompt Engineering** | Format control, simple tasks | $ | Fast | High (easy to change) |
| **Few-Shot Prompting** | Classification, style examples | $ | Fast | High |
| **RAG** | Need external/updated knowledge | $$ | Medium | High |
| **Fine-Tuning (LoRA)** | Domain-specific, consistent behavior | $$$ | Slow | Low (need retraining) |
| **Fine-Tuning (Full)** | Maximum performance needed | $$$$ | Very Slow | Very Low |

---

## Reflection Questions

1. Why can't base models follow instructions well?
2. What's the difference between SFT and RLHF?
3. Why is DPO becoming popular over RLHF?
4. When would you use LoRA instead of full fine-tuning?
5. In what scenario would you combine RAG and fine-tuning?

---

## Next Steps
- Proceed to Lab 2.6 to explore embeddings, tokenization, and context windows
- Understand the technical foundations of how LLMs process text
