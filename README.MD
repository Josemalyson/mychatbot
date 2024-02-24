# Install ollama in localhost

    curl -fsSL https://ollama.com/install.sh | sh

# Quickstart
https://github.com/ollama/ollama

https://huggingface.co/blog/llama2#how-to-prompt-llama-2

    ollama run llama2

# Required config local to run ollama
    https://ollama.com/library/llama2
    
Note: You should have at least 8 GB of RAM available to run the 7B models, 16 GB to run the 13B models, and 32 GB to run the 33B models.

# Ollama with Huggin Face Hub
https://python.langchain.com/docs/integrations/llms/ollama#setup

### Test run services

```shell
    curl http://localhost:11434/api/generate -d '{
      "model": "llama2",
      "prompt":"Why is the sky blue?"
    }'
```

# Run LLMs locally
https://python.langchain.com/docs/guides/local_llms

# Using local models
https://python.langchain.com/docs/use_cases/question_answering/local_retrieval_qa#document-loading

# Local models
https://python.langchain.com/docs/integrations/llms/titan_takeoff#installation

https://python.langchain.com/docs/integrations/llms/xinference#deploy-xinference-locally-or-in-a-distributed-cluster.