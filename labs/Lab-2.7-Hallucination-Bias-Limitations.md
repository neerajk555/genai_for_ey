# Lab 2.7: Hallucination, Grounding, Bias, and Model Limitations

## Objective
Understand LLM limitations including hallucination, bias, and knowledge cutoff, and learn practical mitigation strategies.

## Duration
~15 minutes

## Prerequisites
- Python 3.8+
- Azure OpenAI API key and endpoint

---

## Exercise: Exploring LLM Limitations and Mitigation Strategies

### Background
LLMs have significant limitations:
- **Hallucination**: Generate plausible-sounding but false information
- **Bias**: Reflect biases from training data
- **Knowledge Cutoff**: Only know data up to training date
- **Lack of Grounding**: No connection to verified facts

Understanding these limitations is crucial for responsible AI deployment.

---

## Step-by-Step Instructions

### Step 1: Create Your Lab File

```cmd
cd C:\genai-labs
notepad llm_limitations.py
```

### Step 2: Write the Exploration Code

```python
from openai import OpenAI

# Azure OpenAI Configuration
endpoint = "YOUR_ENDPOINT"
deployment_name = "YOUR_DEPLOYMENT_NAME"
api_key = "YOUR_API_KEY"

client = OpenAI(base_url=endpoint, api_key=api_key)

def call_azure_openai(messages, temperature=0.7, max_completion_tokens=200):
    """Call Azure OpenAI API"""
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def demo_hallucination():
    """
    Demonstrate hallucination with various examples
    """
    print("=" * 70)
    print("1. HALLUCINATION: When LLMs Make Things Up")
    print("=" * 70)
    
    print("\nWhat is Hallucination?")
    print("  - LLM generates confident-sounding but FALSE information")
    print("  - Looks plausible but is factually incorrect")
    print("  - Model doesn't 'know' it's wrong!\n")
    
    print("--- Example 1: Non-existent Person ---")
    question1 = "Tell me about Dr. Sarah Martinez's groundbreaking work on quantum linguistics at MIT."
    print(f"Question: {question1}")
    print("\n⚠️  WARNING: This person doesn't exist!")
    
    response1 = call_azure_openai([
        {"role": "user", "content": question1}
    ], temperature=0.7)
    print(f"\nModel Response:\n{response1}")
    print("\n💡 Observation: Model may generate plausible-sounding details about")
    print("   a non-existent person and field. This is HALLUCINATION!")
    
    input("\n➡️  Press Enter to continue...")
    
    print("\n--- Example 2: Non-existent Book ---")
    question2 = "What are the main themes in the book 'Digital Shadows' by Robert Thompson published in 2021?"
    print(f"\nQuestion: {question2}")
    print("⚠️  WARNING: This book doesn't exist!")
    
    response2 = call_azure_openai([
        {"role": "user", "content": question2}
    ], temperature=0.7)
    print(f"\nModel Response:\n{response2}")
    print("\n💡 Observation: Model may fabricate plot, themes, and analysis!")
    
    print("\n--- Example 3: False Statistics ---")
    question3 = "What percentage of Fortune 500 companies use blockchain for supply chain in 2024?"
    print(f"\nQuestion: {question3}")
    
    response3 = call_azure_openai([
        {"role": "user", "content": question3}
    ], temperature=0.7)
    print(f"\nModel Response:\n{response3}")
    print("\n💡 Observation: Model may cite specific percentages without source!")

def demo_hallucination_types():
    """
    Categorize types of hallucinations
    """
    print("\n\n" + "=" * 70)
    print("2. TYPES OF HALLUCINATIONS")
    print("=" * 70)
    
    print("\n📋 Category 1: FACTUAL HALLUCINATIONS")
    print("   - False facts, dates, numbers, statistics")
    print("   - Example: 'Paris is the capital of Germany'")
    print("   - Most dangerous in enterprise settings")
    
    print("\n📋 Category 2: SOURCE HALLUCINATIONS")
    print("   - Cites non-existent papers, articles, URLs")
    print("   - Example: 'According to Smith et al. (2022) in Nature...'")
    print("   - Common in academic/research queries")
    
    print("\n📋 Category 3: REASONING HALLUCINATIONS")
    print("   - Logical errors presented as correct reasoning")
    print("   - Example: Incorrect mathematical proofs")
    print("   - Hard to detect without domain expertise")
    
    print("\n📋 Category 4: CONTEXTUAL HALLUCINATIONS")
    print("   - Makes up details from provided context")
    print("   - Example: 'As mentioned in the document...' (when it wasn't)")
    print("   - Happens with RAG systems")
    
    print("\n💡 Why Do Hallucinations Happen?")
    print("   1. Training objective: Predict plausible next token (not truth)")
    print("   2. No access to external knowledge/databases")
    print("   3. Fills gaps with 'reasonable' guesses")
    print("   4. No uncertainty representation ('I don't know' is rare)")

def demo_grounding():
    """
    Demonstrate grounding and its importance
    """
    print("\n\n" + "=" * 70)
    print("3. GROUNDING: Connecting to Verified Sources")
    print("=" * 70)
    
    print("\nGrounding = Connecting model output to verified, authoritative sources\n")
    
    print("--- Example: Ungrounded Response ---")
    question = "What are the side effects of medication XYZ-123?"
    print(f"Question: {question}")
    
    response1 = call_azure_openai([
        {"role": "user", "content": question}
    ], temperature=0.3)
    print(f"\nUngrounded Response:\n{response1}")
    print("\n⚠️  DANGER: No sources cited, could be hallucinated!")
    
    input("\n➡️  Press Enter to continue...")
    
    print("\n--- Example: Grounded Response (with RAG) ---")
    print("Approach: Retrieve authoritative medical information first")
    
    # Simulate RAG with explicit source
    grounded_prompt = f"""Based on this verified medical information:

SOURCE: FDA Drug Information Database
MEDICATION: XYZ-123
COMMON SIDE EFFECTS: Nausea (15%), headache (10%), dizziness (8%)
SERIOUS SIDE EFFECTS: Allergic reactions (rare, <1%)

Question: {question}

Provide answer based ONLY on the source above. Cite the source."""
    
    response2 = call_azure_openai([
        {"role": "user", "content": grounded_prompt}
    ], temperature=0)
    print(f"\nGrounded Response:\n{response2}")
    print("\n✅ BETTER: Response tied to specific source!")
    
    print("\n💡 Grounding Strategies:")
    print("   1. RAG (Retrieval-Augmented Generation)")
    print("      → Retrieve relevant docs → Include in prompt → Generate")
    print("   2. Citations")
    print("      → Instruct model to cite sources")
    print("   3. Tool Use / Function Calling")
    print("      → Connect to databases, APIs for real data")
    print("   4. Hybrid Approach")
    print("      → Combine multiple strategies")

def demo_bias():
    """
    Demonstrate bias in LLMs
    """
    print("\n\n" + "=" * 70)
    print("4. BIAS IN LLMs")
    print("=" * 70)
    
    print("\nBias Sources:")
    print("  1. Training Data Bias - Reflects biases in internet text")
    print("  2. Instruction Bias - How we prompt affects output")
    print("  3. Evaluation Bias - How we measure success")
    
    print("\n--- Example 1: Occupational Stereotypes ---")
    
    # Note: Modern models have safety training, but biases can still emerge
    test_prompts = [
        "Complete the sentence: The nurse prepared her",
        "Complete the sentence: The engineer prepared his",
        "Complete the sentence: The CEO prepared their"
    ]
    
    print("\nTesting occupational associations:")
    for prompt in test_prompts:
        print(f"\n  Prompt: '{prompt}'")
        print("  → Testing for gender bias in pronoun associations")
    
    print("\n💡 Modern LLMs have mitigation, but bias can emerge subtly")
    
    print("\n--- Example 2: Cultural Bias ---")
    print("\nBias Example: 'Describe a family dinner'")
    print("  → May default to Western cultural norms")
    print("  → Overlooks diverse cultural practices")
    
    print("\n--- Example 3: Language Bias ---")
    print("\nEnglish-centric training:")
    print("  → Better performance on English")
    print("  → Lower quality for other languages")
    print("  → More tokens needed for non-English text")
    
    print("\n📋 Types of Bias:")
    print("   • Gender Bias: Stereotypical gender associations")
    print("   • Racial Bias: Stereotypes about ethnicities")
    print("   • Cultural Bias: Western-centric perspectives")
    print("   • Age Bias: Assumptions about age groups")
    print("   • Socioeconomic Bias: Assumptions about wealth/education")

def demo_mitigation_strategies():
    """
    Demonstrate practical mitigation strategies
    """
    print("\n\n" + "=" * 70)
    print("5. MITIGATION STRATEGIES: Practical Solutions")
    print("=" * 70)
    
    print("\n🛠️  Strategy 1: SYSTEM PROMPTS WITH CONSTRAINTS")
    
    print("\n--- Without Constraints ---")
    question = "Is coffee good for health?"
    response1 = call_azure_openai([
        {"role": "user", "content": question}
    ], temperature=0.3, max_tokens=100)
    print(f"Question: {question}")
    print(f"Response: {response1[:150]}...")
    
    input("\n➡️  Press Enter to continue...")
    
    print("\n--- With Constraints (Better) ---")
    response2 = call_azure_openai([
        {"role": "system", "content": """You are a helpful assistant. Follow these rules:
1. If you don't know, say 'I don't have verified information'
2. Cite uncertainty when appropriate
3. Never make up sources or statistics
4. Request clarification for ambiguous questions"""},
        {"role": "user", "content": question}
    ], temperature=0.3, max_tokens=100)
    print(f"Response: {response2[:150]}...")
    print("\n✅ System prompt sets clear behavioral guidelines!")
    
    print("\n\n🛠️  Strategy 2: RAG (Retrieval-Augmented Generation)")
    print("   How it works:")
    print("     1. User asks question")
    print("     2. Retrieve relevant documents from knowledge base")
    print("     3. Include documents in prompt")
    print("     4. Model answers based on provided context")
    print("   Benefits:")
    print("     ✓ Grounds responses in real documents")
    print("     ✓ Reduces hallucination")
    print("     ✓ Enables citations")
    print("     ✓ Keeps info up-to-date (update docs, not model)")
    
    print("\n\n🛠️  Strategy 3: OUTPUT VERIFICATION")
    print("   Techniques:")
    print("     • Chain-of-Verification: Ask model to verify its own output")
    print("     • Multi-Model Consensus: Compare outputs from multiple models")
    print("     • Human-in-the-Loop: Human reviews critical outputs")
    print("     • Automated Fact-Checking: Check against knowledge bases")
    
    print("\n\n🛠️  Strategy 4: TEMPERATURE & SAMPLING")
    print("   • Lower temperature (0-0.3) → More deterministic, less creative")
    print("   • Reduces hallucination for factual queries")
    print("   • But doesn't eliminate it!")
    
    print("\n\n🛠️  Strategy 5: PROMPT ENGINEERING")
    print("   ✓ 'List sources for your answer'")
    print("   ✓ 'If unsure, say I don't know'")
    print("   ✓ 'Base answer only on this context: [context]'")
    print("   ✓ 'Think step-by-step and verify each claim'")

def demo_knowledge_cutoff():
    """
    Demonstrate knowledge cutoff limitations
    """
    print("\n\n" + "=" * 70)
    print("6. KNOWLEDGE CUTOFF: The Staleness Problem")
    print("=" * 70)
    
    print("\nProblem: Models only know information from training data")
    
    # Check model's knowledge cutoff
    question1 = "What is your knowledge cutoff date?"
    print(f"\nQuestion: {question1}")
    response1 = call_azure_openai([
        {"role": "user", "content": question1}
    ], temperature=0, max_tokens=50)
    print(f"Model: {response1}")
    
    print("\n--- Testing Current Events ---")
    question2 = "What major tech event happened last week?"
    print(f"\nQuestion: {question2}")
    response2 = call_azure_openai([
        {"role": "user", "content": question2}
    ], temperature=0.5, max_tokens=100)
    print(f"Model: {response2}")
    print("\n💡 Model cannot know events after its training cutoff!")
    
    print("\n⚠️  Practical Implications:")
    print("   ✗ Cannot answer about recent events")
    print("   ✗ Outdated information (APIs, best practices)")
    print("   ✗ New products, companies, people unknown")
    print("   ✗ Changed facts (company acquisitions, new research)")
    
    print("\n✅ Solutions:")
    print("   1. RAG with updated documents")
    print("   2. Function calling to fetch live data")
    print("   3. Fine-tuning on recent data (expensive)")
    print("   4. Web search integration")
    print("   5. Clear documentation of cutoff date to users")

def demo_decision_matrix():
    """
    Provide decision matrix for risk mitigation
    """
    print("\n\n" + "=" * 70)
    print("7. RISK MITIGATION DECISION MATRIX")
    print("=" * 70)
    
    print("\n┌─────────────────────┬──────────────┬─────────────────────────┐")
    print("│ Use Case            │ Risk Level   │ Recommended Mitigations │")
    print("├─────────────────────┼──────────────┼─────────────────────────┤")
    print("│ Medical advice      │ 🔴 CRITICAL  │ RAG + Human review +    │")
    print("│                     │              │ Liability disclaimers   │")
    print("├─────────────────────┼──────────────┼─────────────────────────┤")
    print("│ Legal advice        │ 🔴 CRITICAL  │ RAG + Lawyer review +   │")
    print("│                     │              │ Source citations        │")
    print("├─────────────────────┼──────────────┼─────────────────────────┤")
    print("│ Financial advice    │ 🔴 CRITICAL  │ RAG + Fact-check +      │")
    print("│                     │              │ Disclaimers             │")
    print("├─────────────────────┼──────────────┼─────────────────────────┤")
    print("│ Customer support    │ 🟠 HIGH      │ RAG on KB + Escalation  │")
    print("│                     │              │ to human for complex    │")
    print("├─────────────────────┼──────────────┼─────────────────────────┤")
    print("│ Code generation     │ 🟡 MEDIUM    │ Testing + Code review + │")
    print("│                     │              │ Security scan           │")
    print("├─────────────────────┼──────────────┼─────────────────────────┤")
    print("│ Content generation  │ 🟡 MEDIUM    │ Fact-check + Human edit │")
    print("│                     │              │ for publishing          │")
    print("├─────────────────────┼──────────────┼─────────────────────────┤")
    print("│ Brainstorming       │ 🟢 LOW       │ Minimal (creativity OK) │")
    print("├─────────────────────┼──────────────┼─────────────────────────┤")
    print("│ Classification      │ 🟢 LOW       │ Validate on test set    │")
    print("└─────────────────────┴──────────────┴─────────────────────────┘")
    
    print("\n💡 Golden Rule:")
    print("   Higher stakes → More mitigation layers needed!")

def main():
    """
    Run all demonstrations
    """
    print("\n🚀 HALLUCINATION, GROUNDING, BIAS & LIMITATIONS - HANDS-ON LAB\n")
    
    demo_hallucination()
    input("\n➡️  Press Enter to continue to next section...")
    
    demo_hallucination_types()
    input("\n➡️  Press Enter to continue...")
    
    demo_grounding()
    input("\n➡️  Press Enter to continue...")
    
    demo_bias()
    input("\n➡️  Press Enter to continue...")
    
    demo_mitigation_strategies()
    input("\n➡️  Press Enter to continue...")
    
    demo_knowledge_cutoff()
    input("\n➡️  Press Enter to continue...")
    
    demo_decision_matrix()
    
    # Summary
    print("\n\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("\n✓ Hallucination: LLMs confidently generate false information")
    print("✓ Types: Factual, source, reasoning, contextual hallucinations")
    print("✓ Grounding: Connect outputs to verified sources (RAG)")
    print("✓ Bias: LLMs reflect training data biases (gender, culture, etc.)")
    print("✓ Knowledge Cutoff: Only know data up to training date")
    print("✓ Mitigation Strategies:")
    print("    • System prompts with constraints")
    print("    • RAG (Retrieval-Augmented Generation)")
    print("    • Output verification (human/automated)")
    print("    • Lower temperature for factual tasks")
    print("    • Prompt engineering (request sources, uncertainty)")
    print("✓ Risk-based approach: More critical = more mitigations")
    print("\n" + "=" * 70)
    
    print("\n⚠️  REMEMBER:")
    print("   Never fully trust LLM output without verification!")
    print("   Always implement appropriate safeguards for your use case!")

if __name__ == "__main__":
    main()
```

### Step 3: Configure Your API

Replace placeholders with your credentials and save.

### Step 4: Run the Lab

```cmd
python llm_limitations.py
```

---

## Expected Output

Interactive exploration of:
1. **Hallucination examples** - non-existent people, books, facts
2. **Types of hallucinations** - factual, source, reasoning, contextual
3. **Grounding strategies** - RAG and source citation
4. **Bias** in LLMs - gender, cultural, language bias
5. **Mitigation strategies** - practical solutions
6. **Knowledge cutoff** implications
7. **Risk decision matrix** for different use cases

---

## Key Concepts Deep Dive

### 1. Why Hallucination Happens

```
LLM Training Objective:
  Maximize P(next_token | previous_tokens)

NOT trained to:
  ✗ Verify facts
  ✗ Say "I don't know"
  ✗ Distinguish truth from plausible fiction

Result:
  → Model generates "plausible" tokens
  → No internal fact-checking mechanism
  → Confidence ≠ Correctness
```

### 2. RAG Architecture (Mitigation)

```
Traditional LLM:
  User Question → LLM → Answer (may hallucinate)

RAG Pipeline:
  1. User Question
       ↓
  2. Retrieve relevant docs from knowledge base
       ↓
  3. Combine question + docs in prompt
       ↓
  4. LLM generates answer based on docs
       ↓
  5. Answer with citations

Benefits: Grounded, verifiable, updatable
```

### 3. Bias Mitigation Hierarchy

```
Level 1: Model-level (Training)
  - Diverse training data
  - Debiasing techniques
  - Red-teaming during development

Level 2: Prompt-level (Deployment)
  - Explicit fairness instructions
  - Diverse examples in few-shot
  - Neutral language

Level 3: Output-level (Post-processing)
  - Bias detection tools
  - Human review
  - Filtering/rewriting

Level 4: System-level (Application)
  - Diverse team building system
  - Regular audits
  - User feedback loops
```

---

## Practical Mitigation Checklist

### ✅ Before Deployment

- [ ] Test with adversarial prompts (hallucination triggers)
- [ ] Measure hallucination rate on validation set
- [ ] Test for bias across demographic groups
- [ ] Document knowledge cutoff date
- [ ] Implement RAG if knowledge-intensive
- [ ] Set up output verification process
- [ ] Create escalation path to humans
- [ ] Add appropriate disclaimers

### ✅ During Operation

- [ ] Monitor for hallucinations (user reports)
- [ ] Log and review problematic outputs
- [ ] Update knowledge base regularly (RAG)
- [ ] A/B test mitigation strategies
- [ ] Collect user feedback
- [ ] Regular bias audits
- [ ] Track confidence scores vs accuracy

### ✅ Prompt Engineering

- [ ] Use system prompts with clear constraints
- [ ] Request citations: "List sources for your answer"
- [ ] Request uncertainty: "If unsure, say 'I don't know'"
- [ ] Provide context: "Base answer only on this: [context]"
- [ ] Use chain-of-thought: "Think step-by-step"
- [ ] Set temperature appropriately (low for facts)

---

## Common Scenarios & Solutions

| Scenario | Problem | Solution |
|----------|---------|----------|
| Medical Q&A | Critical hallucinations | RAG + Human review + Disclaimers |
| Customer Support | Inconsistent answers | RAG on knowledge base + Escalation |
| Code Generation | Insecure/buggy code | Testing + Security scan + Review |
| Content Creation | False claims | Fact-checking + Citations + Human edit |
| Research Assistant | Fake citations | Grounding + Citation verification |

---

## Reflection Questions

1. Why can't LLMs reliably say "I don't know"?
2. How does RAG reduce hallucination?
3. What types of bias have you observed in LLM outputs?
4. How would you verify LLM output for a medical application?
5. What's the relationship between temperature and hallucination?

---

## Additional Resources

**Testing for Hallucinations**:
- TruthfulQA dataset
- HaluEval benchmark
- Adversarial prompts collection

**Bias Detection Tools**:
- WinoBias, WinoGender
- BOLD (Bias in Open-ended Language Generation Dataset)
- Holistic Evaluation of Language Models (HELM)

---

## Next Steps
- Proceed to Lab 2.8 for model selection and enterprise considerations
- Learn how to choose the right model for your use case
