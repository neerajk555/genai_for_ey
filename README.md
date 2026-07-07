# LLM Foundations - Hands-On Lab Guides

## 📚 Complete Lab Series for Gen AI Training

Welcome to the hands-on lab series for understanding Large Language Models and Transformers! This collection contains 8 beginner-friendly exercises designed to be completed in under 15 minutes each.

---

## 🎯 Lab Overview

| Lab | Topic | Duration | Difficulty |
|-----|-------|----------|------------|
| [Lab 2.1](labs/Lab-2.1-RNNs-to-Transformers.md) | From RNNs to Transformers | 15 min | ⭐ Beginner |
| [Lab 2.2](labs/Lab-2.2-Transformer-Architecture.md) | Transformer Architecture - Conceptual Recap | 15 min | ⭐⭐ Beginner |
| [Lab 2.3](labs/Lab-2.3-Pretraining-Objectives.md) | LLM Pretraining and Language Modelling | 15 min | ⭐ Beginner |
| [Lab 2.4](labs/Lab-2.4-Decoding-Strategies.md) | Decoding Strategies | 15 min | ⭐⭐ Beginner |
| [Lab 2.5](labs/Lab-2.5-Alignment-FineTuning.md) | Instruction Tuning, Fine-Tuning, and Alignment | 15 min | ⭐⭐ Beginner |
| [Lab 2.6](labs/Lab-2.6-Embeddings-Tokenization.md) | Embeddings, Context Windows, and Tokenisation | 15 min | ⭐⭐ Beginner |
| [Lab 2.7](labs/Lab-2.7-Hallucination-Bias-Limitations.md) | Hallucination, Grounding, Bias, and Limitations | 15 min | ⭐⭐ Beginner |
| [Lab 2.8](labs/Lab-2.8-Model-Selection-Enterprise.md) | Model Selection and Enterprise Considerations | 15 min | ⭐⭐ Beginner |

---

## 🚀 Getting Started

### Prerequisites

1. **Windows Machine** (as specified)
2. **Python 3.8 or higher**
3. **Azure OpenAI API Access** with:
   - Endpoint URL
   - API Key
   - Deployment Name

### Initial Setup

```cmd
# 1. Create lab directory
mkdir C:\genai-labs
cd C:\genai-labs

# 2. Install Python (if not already installed)
# Download from: https://www.python.org/downloads/

# 3. Install required libraries
pip install requests tiktoken numpy
```

### Azure OpenAI Configuration

Before starting the labs, you'll need:

1. **Azure OpenAI Endpoint**
   - Format: `https://your-resource.openai.azure.com/`
   - Find in Azure Portal → Your OpenAI Resource → Keys and Endpoint

2. **API Key**
   - Found in: Azure Portal → Keys and Endpoint → KEY 1 or KEY 2

3. **Deployment Name**
   - The name you gave when deploying a model (e.g., `gpt-4o`, `gpt-35-turbo`)
   - Find in: Azure Portal → Model deployments

---

## 📖 Lab Descriptions

### Lab 2.1: From RNNs to Transformers
**What you'll learn:**
- Why RNNs have vanishing gradient problems
- How transformers process text in parallel
- Historical evolution: ELMo → GPT → BERT → GPT-4
- Practical comparison of sequential vs parallel processing

**Key Exercise:** Test how modern transformers handle long-range dependencies better than RNNs.

---

### Lab 2.2: Transformer Architecture
**What you'll learn:**
- Self-attention mechanism (Query, Key, Value)
- Multi-head attention and its purpose
- Encoder-only (BERT) vs Decoder-only (GPT) vs Encoder-Decoder (T5)
- Positional encoding and token flow

**Key Exercise:** Visualize attention patterns and understand how tokens "attend" to each other.

---

### Lab 2.3: LLM Pretraining and Language Modelling
**What you'll learn:**
- Autoregressive language modeling (GPT-style)
- Masked language modeling (BERT-style)
- What LLMs learn: syntax, facts, reasoning
- Scale laws and knowledge cutoff

**Key Exercise:** Compare autoregressive vs masked language modeling with live examples.

---

### Lab 2.4: Decoding Strategies
**What you'll learn:**
- Temperature control (creativity vs determinism)
- Top-p (nucleus) sampling
- Greedy decoding vs sampling
- Repetition penalties

**Key Exercise:** Experiment with different decoding parameters and see their effects on output.

---

### Lab 2.5: Instruction Tuning, Fine-Tuning, and Alignment
**What you'll learn:**
- Base models vs instruction-tuned models
- Supervised Fine-Tuning (SFT)
- RLHF and DPO explained
- Parameter-efficient fine-tuning (LoRA/QLoRA)
- When to use prompt engineering vs fine-tuning vs RAG

**Key Exercise:** Compare aligned vs unaligned model behavior and explore the decision framework.

---

### Lab 2.6: Embeddings, Context Windows, and Tokenisation
**What you'll learn:**
- How tokenization works (BPE, SentencePiece)
- Token vs word vs character
- Context window limits
- Embedding models vs generative models
- Token-based pricing implications

**Key Exercise:** Count tokens in different texts and understand cost/length implications.

---

### Lab 2.7: Hallucination, Grounding, Bias, and Limitations
**What you'll learn:**
- What hallucination is and why it happens
- Types of hallucinations (factual, source, reasoning)
- Grounding with RAG
- Bias in LLMs
- Practical mitigation strategies

**Key Exercise:** Observe hallucination examples and test mitigation techniques.

---

### Lab 2.8: Model Selection and Enterprise Considerations
**What you'll learn:**
- Closed-source (GPT, Claude) vs open-source (Llama, Mistral)
- Understanding benchmarks (MMLU, HumanEval, MT-Bench)
- Large vs small model trade-offs
- Cost analysis and TCO
- Enterprise requirements: compliance, SLA, data residency

**Key Exercise:** Use decision framework to select the right model for different scenarios.

---

## 🛠️ How to Use These Labs

### Step-by-Step Process:

1. **Read the background** section to understand concepts
2. **Follow the setup** instructions to create your Python script
3. **Configure your Azure OpenAI** credentials
4. **Run the lab** and observe the output
5. **Review key concepts** in the deep dive section
6. **Answer reflection questions** to test understanding
7. **Proceed to next lab** when ready

### Tips for Success:

✅ **DO:**
- Work through labs in order (they build on each other)
- Take time to read explanations
- Experiment with different parameters
- Answer reflection questions
- Take notes on key insights

❌ **DON'T:**
- Skip reading the background sections
- Rush through without understanding
- Ignore errors (they're learning opportunities)
- Forget to save your code (you can reuse it!)

---

## 📊 Learning Path

```
Start Here
    ↓
Lab 2.1: Why Transformers? (Foundation)
    ↓
Lab 2.2: How Transformers Work (Architecture)
    ↓
Lab 2.3: How LLMs are Trained (Pretraining)
    ↓
Lab 2.4: Controlling Generation (Decoding)
    ↓
Lab 2.5: Making Models Helpful (Alignment)
    ↓
Lab 2.6: Technical Foundations (Tokens & Embeddings)
    ↓
Lab 2.7: Understanding Limitations (Safety)
    ↓
Lab 2.8: Choosing the Right Model (Enterprise)
    ↓
Complete! 🎉
```

---

## 🔧 Troubleshooting

### Common Issues:

**Issue 1: API Authentication Error**
```
Error: 401 Unauthorized
```
**Solution:** Check that your API key is correct and has not expired.

**Issue 2: Module Not Found**
```
ModuleNotFoundError: No module named 'requests'
```
**Solution:** Install required libraries:
```cmd
pip install requests tiktoken numpy
```

**Issue 3: Context Window Exceeded**
```
Error: maximum context length exceeded
```
**Solution:** Reduce `max_tokens` parameter or shorten input.

**Issue 4: Rate Limit Error**
```
Error: 429 Too Many Requests
```
**Solution:** Wait a moment and retry. Your API tier may have rate limits.

---

## 💡 Key Concepts Summary

After completing all labs, you'll understand:

### Technical Concepts:
- ✅ Transformer architecture (attention, multi-head, encoder/decoder)
- ✅ Tokenization and embeddings
- ✅ Context windows and their limits
- ✅ Pretraining objectives (autoregressive, masked LM)
- ✅ Decoding strategies (temperature, top-p, sampling)
- ✅ Fine-tuning methods (SFT, RLHF, DPO, LoRA)

### Practical Skills:
- ✅ Making API calls to Azure OpenAI
- ✅ Controlling output with parameters
- ✅ Understanding token counts and costs
- ✅ Identifying and mitigating hallucinations
- ✅ Choosing the right model for a use case
- ✅ Enterprise deployment considerations

### Best Practices:
- ✅ Always count tokens before API calls
- ✅ Use RAG for knowledge-intensive tasks
- ✅ Start with prompt engineering before fine-tuning
- ✅ Implement safeguards for critical applications
- ✅ Monitor costs and performance
- ✅ Test on your own data, not just benchmarks

---

## 📚 Additional Resources

### Documentation:
- [Azure OpenAI Service Docs](https://learn.microsoft.com/azure/ai-services/openai/)
- [OpenAI API Reference](https://platform.openai.com/docs)
- [Transformer Paper (Attention is All You Need)](https://arxiv.org/abs/1706.03762)

### Tools:
- [tiktoken](https://github.com/openai/tiktoken) - Token counting
- [Hugging Face](https://huggingface.co/) - Model hub and datasets
- [LangChain](https://www.langchain.com/) - LLM application framework
- [LlamaIndex](https://www.llamaindex.ai/) - RAG framework

### Benchmarks:
- [MMLU](https://github.com/hendrycks/test) - Multitask Language Understanding
- [HumanEval](https://github.com/openai/human-eval) - Code generation
- [TruthfulQA](https://github.com/sylinrl/TruthfulQA) - Factuality
- [HELM](https://crfm.stanford.edu/helm/) - Holistic evaluation

---

## 🎓 Learning Outcomes

By completing these labs, you will be able to:

1. **Explain** the evolution from RNNs to transformers
2. **Describe** how self-attention and transformers work
3. **Understand** LLM pretraining and alignment
4. **Use** Azure OpenAI API effectively
5. **Control** text generation with decoding strategies
6. **Identify** and mitigate hallucinations and bias
7. **Calculate** token counts and costs
8. **Select** appropriate models for different use cases
9. **Design** enterprise LLM deployments
10. **Apply** best practices for safe and effective LLM use

---

## 🤝 Support

If you encounter issues:

1. Check the **Troubleshooting** section in each lab
2. Review **Key Concepts** for clarification
3. Verify your Azure OpenAI credentials
4. Ensure all dependencies are installed
5. Check API rate limits and quotas

---

## ✅ Completion Checklist

Track your progress:

- [ ] Lab 2.1: RNNs to Transformers
- [ ] Lab 2.2: Transformer Architecture
- [ ] Lab 2.3: Pretraining Objectives
- [ ] Lab 2.4: Decoding Strategies
- [ ] Lab 2.5: Alignment and Fine-Tuning
- [ ] Lab 2.6: Embeddings and Tokenization
- [ ] Lab 2.7: Hallucination and Bias
- [ ] Lab 2.8: Model Selection

---

## 🎉 What's Next?

After completing these labs, consider:

1. **Building a RAG System**
   - Use LangChain or LlamaIndex
   - Integrate with your own documents

2. **Fine-Tuning a Model**
   - Try LoRA fine-tuning on Hugging Face
   - Use a smaller open-source model

3. **Deploying an Application**
   - Build a chatbot or Q&A system
   - Implement proper safeguards

4. **Exploring Advanced Topics**
   - Multi-modal models (vision + language)
   - Agent frameworks (AutoGPT, LangGraph)
   - Advanced RAG techniques

---

## 📝 Feedback

These labs are designed to be beginner-friendly and practical. If you have suggestions for improvement:
- Note any confusing sections
- Suggest additional examples
- Report any errors or issues
- Share what worked well

---

## 📄 License

These lab materials are provided for educational purposes.

---

**Happy Learning! 🚀**

Start with [Lab 2.1: From RNNs to Transformers](Lab-2.1-RNNs-to-Transformers.md)
