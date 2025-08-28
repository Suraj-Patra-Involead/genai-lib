* What the repo is
* How to install dependencies
* Folder structure
* How to use each module
* Links to the **example notebooks (`.ipynb`)** for learning

Here’s a clean **`README.md`** you can put at the root of your repo:

```markdown
# GENAI_LIB 🚀
A lightweight, extensible library for working with **Large Language Models (LLMs)** and **Prompting Techniques**.  

This project contains two main modules:  
1. **llm_tools** → A robust, multi-provider LLM client (Ollama, OpenAI, Hugging Face).  
2. **prompting_tools** → A collection of classical prompting techniques with Ollama.  

It is designed for **learners, developers, and researchers** who want a clean abstraction layer for experimenting with LLMs.

---

## 📂 Folder Structure

```

GENAI\_LIB/
│
├── llm\_tools/
│   ├── llm\_tools.py         # Multi-provider LLM client
│   ├── llm\_tools.ipynb      # Tutorial notebook
│   └── **init**.py
│
├── prompting\_tools/
│   ├── prompter.py          # Classical prompting techniques
│   ├── prompting\_tools.ipynb# Tutorial notebook
│   └── **init**.py
│
└── requirements.txt         # Python dependencies

````

---

## 🔧 Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/yourusername/GENAI_LIB.git
cd GENAI_LIB
pip install -r requirements.txt
````

Requirements (minimal):

```
requests>=2.20.0
ollama>=0.1.0
```

---

## ⚡ Usage

### 1. LLM Client (multi-provider)

Supports **Ollama**, **OpenAI**, and **Hugging Face Inference API**.

```python
from llm_tools.llm_tools import LLMConfig, LLMClient, Provider

cfg = LLMConfig(
    provider=Provider.OLLAMA,
    model="mistral:latest",
    temperature=0.7,
    base_url="http://localhost:11434"
)

client = LLMClient(cfg)
res = client.generate(user_prompt="Explain diffusion models simply.")
print(res.text)
```

➡️ Learn more in the tutorial:
[📘 llm\_tools.ipynb](llm_tools/llm_tools.ipynb)

---

### 2. Prompting Techniques

Experiment with **zero-shot, few-shot, chain-of-thought, self-refine, etc.**

```python
from prompting_tools.prompter import PromptingTechniques

prompter = PromptingTechniques(model="mistral:latest")

answer = prompter.chain_of_thought("What is the square root of 144?")
print(answer)
```

➡️ Learn more in the tutorial:
[📘 prompting\_tools.ipynb](prompting_tools/prompting_tools.ipynb)

---

## 🧑‍💻 Features

* ✅ Multi-provider support (Ollama / OpenAI / Hugging Face)
* ✅ Classical chat structure (system + user + history)
* ✅ Robust error handling with retries
* ✅ Minimal dependencies (`requests` + `ollama`)
* ✅ Ready-to-use prompting strategies

---

## 📖 Tutorials

* [LLM Client Walkthrough](llm_tools/llm_tools.ipynb)
* [Prompting Techniques Guide](prompting_tools/prompting_tools.ipynb)

---

## 👨‍💻 Author

**Suraj (Data Science Intern)**
*Built for learning and practical experimentation with Generative AI.*

---

## 📜 License

MIT License – free to use, modify, and share.