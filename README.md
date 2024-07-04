# README

This README file provides step-by-step instructions for setting up, converting, and using the "Indic-gemma-2b-finetuned-sft-Navarasa" model with `llama.cpp` and Hugging Face Hub.

## 1. Install the Required Packages

To get started, install the necessary packages by running the following commands:

```bash
!pip install huggingface_hub
!pip install torch
!pip install transformers
!pip install sentencepiece
!pip install protobuf==3.20.3
!pip install ctransformers
!pip install llama-cpp-python
```

## 2. Download a Model Snapshot from Hugging Face Hub

Download the model snapshot using the `huggingface_hub` package:

```python
from huggingface_hub import snapshot_download

model_id = "Telugu-LLM-Labs/Indic-gemma-2b-finetuned-sft-Navarasa-2.0"
snapshot_download(repo_id=model_id, local_dir="Indic-gemma-2b-finetuned-sft-Navarasa", local_dir_use_symlinks=False, revision="main")
```

List the downloaded files to confirm the download:

```bash
!ls -lash Indic-gemma-2b-finetuned-sft-Navarasa
```

## 3. Clone the llama.cpp Repository

Clone the `llama.cpp` repository:

```bash
!git clone https://github.com/ggerganov/llama.cpp.git
```

Install the required dependencies:

```bash
!pip install -r llama.cpp/requirements.txt
```

## 4. Convert the Hugging Face Model to GGUF Format

Convert the downloaded Hugging Face model to GGUF format:

```bash
!python llama.cpp/convert-hf-to-gguf.py Indic-gemma-2b-finetuned-sft-Navarasa \
  --outfile Indic-gemma-2b-finetuned-sft-Navarasa-2.0.gguf \
  --outtype q8_0
```

Verify the converted model:

```bash
!ls -lash Indic-gemma-2b-finetuned-sft-Navarasa-2.0.gguf
```

## 5. Push the Model to a Repository on Hugging Face Hub

First, log in to Hugging Face Hub:

```bash
!huggingface-cli login
```

Create a repository and upload the GGUF model:

```python
from huggingface_hub import HfApi
import os
import getpass

# Set the token as an environment variable (recommended for security)
api = HfApi(token=getpass.getpass())

model_id = "05deepak/Indic-gemma-2b-finetuned-sft-Navarasa-2.1.gguf"
api.create_repo(model_id, exist_ok=True, repo_type="model")

api.upload_file(
    path_or_fileobj="Indic-gemma-2b-finetuned-sft-Navarasa-2.0.gguf",
    path_in_repo="Indic-gemma-2b-finetuned-sft-Navarasa-2.0.gguf",
    repo_id=model_id,
)
```

## 6. Inferencing GGUF Model from Hugging Face and Local

Download the GGUF model and run inference:

```python
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# Download the GGUF model
model_name = "05deepak/Indic-gemma-2b-finetuned-sft-Navarasa-2.1.gguf"
model_file = "Indic-gemma-2b-finetuned-sft-Navarasa-2.0.gguf"
model_path_hf = hf_hub_download(model_name, filename=model_file)

# Local path to the model
model_path_local = "/content/Indic-gemma-2b-finetuned-sft-Navarasa-2.0.gguf"

# Instantiate model from downloaded file
llm = Llama(
    model_path=model_path_local,
    n_ctx=16000,
    n_threads=32,
    n_gpu_layers=0
)

# Generation kwargs
generation_kwargs = {
    "max_tokens": 20000,
    "stop": ["</s>"],
    "echo": False,
    "top_k": 1
}

# Prompt for inference
prompt = "what is machine learning "

# Run inference
res = llm(prompt, **generation_kwargs)

# Print the generated text
print(res["choices"][0]["text"])
```

## Answer:

Machine learning is a subset of artificial intelligence that involves the development of algorithms and statistical models to enable computers to learn from data. It allows machines to improve their performance on a specific task by analyzing and making decisions based on data, rather than being explicitly programmed. Machine learning can be applied to a wide range of applications, including image and speech recognition, predictive analytics, and fraud detection.
