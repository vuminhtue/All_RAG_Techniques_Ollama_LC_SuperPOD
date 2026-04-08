- This work follow the repo from https://github.com/NirDiamant/RAG_Techniques 
- The original repo used OpenAI API and Cohere embedding for multiple RAG techniques
- We modify NirDiamant's original repo to work with our HPC using Ollama LLMs (mostly gemma3:12b) and embedding models (mxbai-embed-large:335m). 
- LLMs and embedding models are arbitrarily selected to work with NVIDIA GPU V100 on M3 and could be adjusted to bigger model on SMU SuperPOD depending on the need.
- The material has been revamped with Step by Step tutorial for students to follow
- This is purely for education purpose and we support our SMU faculty with different RAG workflow
- For every ipynb, with testing/ollama module loaded, you need to fill in the following code at the beginning of each notebook to let the Ollama know its hosts and ports:

```
import os
# get home and job id
home_dir =  os.getenv('HOME')
job_id = os.getenv('SLURM_JOB_ID')

# get ollama directory or default to home
ollama_dir = os.getenv('OLLAMA_BASE_DIR', home_dir)

try:    
    with open(f"{ollama_dir}/ollama/host_{job_id}.txt") as f:
        HOST = f.read().strip()
    with open(f"{ollama_dir}/ollama/port_{job_id}.txt") as f:
        PORT = f.read().strip()
    ollama_url = f"http://{HOST}:{PORT}"
except Exception as e:
    print("[⚠️] Could not read host/port, you manually set the `ollama_url` below.")

print("The port that serve ollama is: ", ollama_url)
```

And insert **ollama_url** to ChatOllama:

```
llm = ChatOllama(base_url=ollama_url,model="gemma3:4b",temperature=0)
```
