


Training SLMs for Function Calling on Unsloth This repository provides an overview of how we trained Small Language Models (SLMs) on Unsloth AI for function-calling capabilities using Salesforce's function-calling dataset. The resulting models are designed for use in creating responsive React agents, set up simply with a for-loop in our agent notebook. 

## 📂 Repository Structure ----------------------- * 

**`training/`**: Contains scripts and configurations for training SLMs on Unsloth. * 

**`agent_notebook.ipynb`**: A Jupyter notebook demonstrating how to set up a React agent using the trained SLMs. * 

**`README.md`**: This document. 📋 How It Works --------------- 1. **Training with Unsloth on Salesforce Dataset** * We fine-tuned a set of Small Language Models (SLMs) using the Salesforce function-calling dataset. * This dataset includes various function-call structures, allowing the SLMs to learn efficient, reliable function-calling patterns. * Training was conducted on **Unsloth AI** for optimized GPU performance, allowing us to handle function calls with minimal latency and high accuracy. 

**Agent Notebook Setup** * In the `agent_notebook.ipynb`, we demonstrate how to create a basic React agent using the trained SLMs. * The agent operates within a simple **for-loop** that cycles through: * **Thought** * **Action** * **Pause** * **Observation** This structured loop enables the agent to function dynamically and respond effectively to inputs without needing complex frameworks. 





---
base_model: unsloth/tiny-lama/phi-instruct
library_name: peft
---