Steps to install vllm

## 1. Create a separate environment of vllm
```
conda create -n vllm python=3.9.23 
conda activate vllm
````

## 2. Install pytorch with cuda 12.8
```
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

## 3. Install vllm ([source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#nvidia-cuda))
### a. If you are using pip
```
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128
```

### b. If you are using uv
```
uv pip install vllm --torch-backend=auto
```

## 4. Downloading model weights
Either download the model weights using the following code in local folder (cache [can be of any name]).

```
from transformers import AutoModelForCausalLM, AutoTokenizer

cache_dir = 'cache'
model_name = "numind/NuExtract-v1.5"

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, cache_dir=cache_dir)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
```
After the download is complete, you will see a folder inside cache: `models--numind--NuExtract-v1.5/snapshots/randomstring` - copy this absolute path

Another way to download hf models locally is shown [here](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli)

## 5. Serving model using vllm on GPU server ([source](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html))

```
conda activate vllm
vllm serve model_path
```

## 6. Accessing the hosted model

* Note that the vllm serve command hosts the model on "http://localhost:8000/v1"
* Therefore, any device which requires access to this model should create a ssh tunnel on port 8000 of the gpu server where the model is hosted.

* **Skip this step if you are running the extraction code on same server where model is hosted**


For example:

```
 ssh -N -L 8000:hostname:8000 user_name@gpu_cluster_ip
```

 Assuming the gpu cluster is available at lemat.com and the username is sam. The model is running on gpu007, use the following command on the laptop to create tunnel

```
ssh -N -L 8000:gpu007:8000 sam@lemat.com
```

## 7. Running the local inference

* Change model name in lm = dspy.LM()
* For example line 158 in dspy_local_extract_synthesis_procedure_from_text_strict-2.py
* openai/{absolute path of model weights which was used to launch the vllm server}

* NOTE: 
 - do not end the path with **/**
 - Since you will be using absolute path, there will be **//** after openai which is correct
 - Leave the api key or change it if you used it while running vllm serve
 
