# Text generation web UI Testing
### Here there be dragons. (But V1 and V2 GPTQ support)

- Allow 4bit loras and use of Autograd + AutoGPTQ for inference 
- Use GPT-J 4-bits (GPTQv1/v2)
- GPT-NeoXT 4-bits (GPTQv1/v2)
- 8 bit threshold slider, default 1.5 (pre compute 7.0)
- V1 Models work in --autograd (declare with --v1)
- V2 Models work in both.
- Offloading works in autograd with --gpu-memory but doesn't 100% hodl while generating
- Offloading with llama_inference_offload, fastest multi-gpu besides exllama
- Autograd + quant_attn beating Autogtpq on P6000!
- Only load one 4bit lora at a time and apply with no loras before switching.
- Train 4-bit loras with Autograd and hopefully soon AutoGPTQ
- exLlama support (compute 7 and up for benefits)
- more parameters from UI for remote hosts


#### Depends on:
https://github.com/Ph0rk0z/GPTQ-Merged (dual module branch)

~~https://github.com/sterlind/peft~~ (now auto patches)

6/24/23
```
Merge the new model page. Hope to break out Autograd into it's own loader soon.
```

6/17/23
```
New branch https://github.com/Ph0rk0z/text-generation-webui-testing/tree/model-page
Uses the new loader based model loading. All the kinks aren't worked out yet.
Definitely required in the future as new inference methods are added.
Not sure how I feel about it so I'll try it out for a few days first.
```

6/8/23
```
exllama support merged
insane inference speed and working multi-gpu
```

5/30/23
```
Dirty lora support for AutoGPTQ. You need my fork or merged PR,
also get PEFT current pip install git+https://github.com/huggingface/peft
No training yet.
```

5/17/23
```
Update submodules, supporting a new method of splitting that makes 65b possible over 2, 
even janky cards at higher speed. No more OOM on 65b at full context.
```


5/8/23
```
I think autograd problem is fixed.. equal or faster than GPTQ
Update the submodules git submodule update --recursive --remote
```


4/22/23
```
New --mlp-attn, slightly faster on some contexts but no lora support added yet.
both --xformers and --sdp-attention prevent the 30b from going OOM at full context.
```

4/18/23
```
Using the patch for PEFT and no longer depends on PEFT fork.
Makes it easier to run main branch side by side.
Rewrote the GPTQ loader as well to be more compact.
You may have to update tokenizers agian and install colorama from pip.
```
4/11/23
```
Update to new PEFT version
https://github.com/sterlind/peft
```


4/10/23
```
pip install deepspeed -U
pip install xmformers
Xformers install will upgrade torch to 2.0
YOU WILL HAVE TO RECOMPILE YOUR CUDA KERNELS!!
```


4/8/23 - Update transformers!
```
pip install tokenizers==0.13.1
pip install protobuf==3.20.0
pip install git+https://github.com/huggingface/transformers
```
Repos are linked as submodules.. you may have to update them: https://stackoverflow.com/a/1032653
```
git submodule update --remote
```

#### Why?

* 13b and 30b llama response times for me become usable with a lora or not.
* Changes aren't so clean to be accepted as a p/r

#### How?

* Clone and re-use your oobabooga/text-generation-webui conda environment.
* Build GPTQ kernel with python setup.py install after cloing into repositories/
* Also build and install patched PEFT.

#### Windows?

* I don't know, can't use it. Try WSL

#### Credits

* https://github.com/johnsmith0031/alpaca_lora_4bit
* https://github.com/0cc4m/GPTQ-for-LLaMa

#### Example Commands
```
python server.py --model llama-30b --chat --autograd --wbits 4 
python server.py --model opt-13b --chat --autograd --wbits 4 --lora opt-13b-lora-1.0ep
python server.py --model oasst-sft-1-pythia-12b --chat --autograd --wbits 4 --model_type gptneox
python server.py --model oasst-sft-1-pythia-12b --chat --autograd --wbits 4 --model_type gptneox --v1
python server.py --model llama-7b-4bit-128g --chat --groupsize 128 --wbits 4 --model_type llama
python server.py --model llama-30b-4bit-128g --chat --autograd --groupsize 128  --wbits 4 --model_type llama

```

|![Image1](https://github.com/oobabooga/screenshots/raw/main/qa.png) | ![Image2](https://github.com/oobabooga/screenshots/raw/main/cai3.png) |
|:---:|:---:|
|![Image3](https://github.com/oobabooga/screenshots/raw/main/gpt4chan.png) | ![Image4](https://github.com/oobabooga/screenshots/raw/main/galactica.png) |

## Features

* 3 interface modes: default, notebook, and chat
* Multiple model backends: tranformers, llama.cpp, AutoGPTQ, GPTQ-for-LLaMa, ExLlama, RWKV, FlexGen
* Dropdown menu for quickly switching between different models
* LoRA: load and unload LoRAs on the fly, load multiple LoRAs at the same time, train a new LoRA
* Precise instruction templates for chat mode, including Alpaca, Vicuna, Open Assistant, Dolly, Koala, ChatGLM, MOSS, RWKV-Raven, Galactica, StableLM, WizardLM, Baize, Ziya, Chinese-Vicuna, MPT, INCITE, Wizard Mega, KoAlpaca, Vigogne, Bactrian, h2o, and OpenBuddy
* [Multimodal pipelines, including LLaVA and MiniGPT-4](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/multimodal)
* 8-bit and 4-bit inference through bitsandbytes
* CPU mode for transformers models
* [DeepSpeed ZeRO-3 inference](docs/DeepSpeed.md)
* [Extensions](docs/Extensions.md)
* [Custom chat characters](docs/Chat-mode.md)
* Very efficient text streaming
* Markdown output with LaTeX rendering, to use for instance with [GALACTICA](https://github.com/paperswithcode/galai)
* Nice HTML output for GPT-4chan
* API, including endpoints for websocket streaming ([see the examples](https://github.com/oobabooga/text-generation-webui/blob/main/api-examples))

To learn how to use the various features, check out the Documentation: https://github.com/oobabooga/text-generation-webui/tree/main/docs

## Installation

### One-click installers

| Windows | Linux | macOS | WSL |
|--------|--------|--------|--------|
| [oobabooga-windows.zip](https://github.com/oobabooga/text-generation-webui/releases/download/installers/oobabooga_windows.zip) | [oobabooga-linux.zip](https://github.com/oobabooga/text-generation-webui/releases/download/installers/oobabooga_linux.zip) |[oobabooga-macos.zip](https://github.com/oobabooga/text-generation-webui/releases/download/installers/oobabooga_macos.zip) | [oobabooga-wsl.zip](https://github.com/oobabooga/text-generation-webui/releases/download/installers/oobabooga_wsl.zip) |

Just download the zip above, extract it, and double-click on "start". The web UI and all its dependencies will be installed in the same folder.

* The source codes are here: https://github.com/oobabooga/one-click-installers
* There is no need to run the installers as admin.
* AMD doesn't work on Windows.
* Huge thanks to [@jllllll](https://github.com/jllllll), [@ClayShoaf](https://github.com/ClayShoaf), and [@xNul](https://github.com/xNul) for their contributions to these installers.

### Manual installation using Conda

Recommended if you have some experience with the command line.

#### 0. Install Conda

https://docs.conda.io/en/latest/miniconda.html

On Linux or WSL, it can be automatically installed with these two commands:

```
curl -sL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > "Miniconda3.sh"
bash Miniconda3.sh
```
Source: https://educe-ubc.github.io/conda.html

#### 1. Create a new conda environment

```A gradio web UI for running Large Language Models like LLaMA, llama.cpp, GPT-J, OPT, and GALACTICA.
41
​
42
​
43
[[Try it on Google Colab]](https://colab.research.google.com/github/oobabooga/AI-Notebooks/blob/main/Colab-TextGen-GPU.ipynb)
44

conda create -n textgen python=3.10.9
conda activate textgen
```

#### 2. Install Pytorch

| System | GPU | Command |
|--------|---------|---------|
| Linux/WSL | NVIDIA | `pip3 install torch torchvision torchaudio` |
| Linux | AMD | `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2` |
| MacOS + MPS (untested) | Any | `pip3 install torch torchvision torchaudio` |
| Windows | NVIDIA | `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117` |

The up-to-date commands can be found here: https://pytorch.org/get-started/locally/. 

#### 2.1 Special instructions

* MacOS users: https://github.com/oobabooga/text-generation-webui/pull/393
* AMD users: https://rentry.org/eq3hg

#### 3. Install the web UI

```
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui
pip install -r requirements.txt
```

#### llama.cpp with GPU acceleration

Requires the additional compilation step described here: [GPU acceleration](https://github.com/oobabooga/text-generation-webui/blob/main/docs/llama.cpp-models.md#gpu-acceleration).

#### bitsandbytes

bitsandbytes >= 0.39 may not work on older NVIDIA GPUs. In that case, to use `--load-in-8bit`, you may have to downgrade like this:

* Linux: `pip install bitsandbytes==0.38.1`
* Windows: `pip install https://github.com/jllllll/bitsandbytes-windows-webui/raw/main/bitsandbytes-0.38.1-py3-none-any.whl`

### Alternative: Docker

```
ln -s docker/{Dockerfile,docker-compose.yml,.dockerignore} .
cp docker/.env.example .env
# Edit .env and set TORCH_CUDA_ARCH_LIST based on your GPU model
docker compose up --build
```

* You need to have docker compose v2.17 or higher installed. See [this guide](https://github.com/oobabooga/text-generation-webui/blob/main/docs/Docker.md) for instructions.
* For additional docker files, check out [this repository](https://github.com/Atinoda/text-generation-webui-docker).

### Updating the requirements

From time to time, the `requirements.txt` changes. To update, use this command:

```
conda activate textgen
cd text-generation-webui
pip install -r requirements.txt --upgrade
```
## Downloading models

Models should be placed inside the `models/` folder.

[Hugging Face](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads) is the main place to download models. These are some examples:

* [Pythia](https://huggingface.co/models?sort=downloads&search=eleutherai%2Fpythia+deduped)
* [OPT](https://huggingface.co/models?search=facebook/opt)
* [GALACTICA](https://huggingface.co/models?search=facebook/galactica)
* [GPT-J 6B](https://huggingface.co/EleutherAI/gpt-j-6B/tree/main)

You can automatically download a model from HF using the script `download-model.py`:

    python download-model.py organization/model

For example:

    python download-model.py facebook/opt-1.3b

To download a protected model, set env vars `HF_USER` and `HF_PASS` to your Hugging Face username and password (or [User Access Token](https://huggingface.co/settings/tokens)). The model's terms must first be accepted on the HF website.

#### GGML models

You can drop these directly into the `models/` folder, making sure that the file name contains `ggml` somewhere and ends in `.bin`.

#### GPT-4chan

<details>
<summary>
Instructions
</summary>

[GPT-4chan](https://huggingface.co/ykilcher/gpt-4chan) has been shut down from Hugging Face, so you need to download it elsewhere. You have two options:

* Torrent: [16-bit](https://archive.org/details/gpt4chan_model_float16) / [32-bit](https://archive.org/details/gpt4chan_model)
* Direct download: [16-bit](https://theswissbay.ch/pdf/_notpdf_/gpt4chan_model_float16/) / [32-bit](https://theswissbay.ch/pdf/_notpdf_/gpt4chan_model/)

The 32-bit version is only relevant if you intend to run the model in CPU mode. Otherwise, you should use the 16-bit version.

After downloading the model, follow these steps:

1. Place the files under `models/gpt4chan_model_float16` or `models/gpt4chan_model`.
2. Place GPT-J 6B's config.json file in that same folder: [config.json](https://huggingface.co/EleutherAI/gpt-j-6B/raw/main/config.json).
3. Download GPT-J 6B's tokenizer files (they will be automatically detected when you attempt to load GPT-4chan):

```
python download-model.py EleutherAI/gpt-j-6B --text-only
```

When you load this model in default or notebook modes, the "HTML" tab will show the generated text in 4chan format.
</details>

## Starting the web UI

    conda activate textgen
    cd text-generation-webui
    python server.py

Then browse to 

`http://localhost:7860/?__theme=dark`

Optionally, you can use the following command-line flags:

#### Basic settings

| Flag                                       | Description |
|--------------------------------------------|-------------|
| `-h`, `--help`                             | Show this help message and exit. |
| `--notebook`                               | Launch the web UI in notebook mode, where the output is written to the same text box as the input. |
| `--chat`                                   | Launch the web UI in chat mode. |
| `--character CHARACTER`                    | The name of the character to load in chat mode by default. |
| `--model MODEL`                            | Name of the model to load by default. |
| `--lora LORA [LORA ...]`                   | The list of LoRAs to load. If you want to load more than one LoRA, write the names separated by spaces. |
| `--model-dir MODEL_DIR`                    | Path to directory with all the models. |
| `--lora-dir LORA_DIR`                      | Path to directory with all the loras. |
| `--model-menu`                             | Show a model menu in the terminal when the web UI is first launched. |
| `--no-stream`                              | Don't stream the text output in real time. |
| `--settings SETTINGS_FILE`                 | Load the default interface settings from this yaml file. See `settings-template.yaml` for an example. If you create a file called `settings.yaml`, this file will be loaded by default without the need to use the `--settings` flag. |
| `--extensions EXTENSIONS [EXTENSIONS ...]` | The list of extensions to load. If you want to load more than one extension, write the names separated by spaces. |
| `--verbose`                                | Print the prompts to the terminal. |

#### Model loader

| Flag                                       | Description |
|--------------------------------------------|-------------|
| `--loader LOADER`                          | Choose the model loader manually, otherwise, it will get autodetected. Valid options: transformers, autogptq, gptq-for-llama, exllama, exllama_hf, llamacpp, rwkv, flexgen |

#### Accelerate/transformers

| Flag                                        | Description |
|---------------------------------------------|-------------|
| `--cpu`                                     | Use the CPU to generate text. Warning: Training on CPU is extremely slow.|
| `--auto-devices`                            | Automatically split the model across the available GPU(s) and CPU. |
|  `--gpu-memory GPU_MEMORY [GPU_MEMORY ...]` | Maximum GPU memory in GiB to be allocated per GPU. Example: `--gpu-memory 10` for a single GPU, `--gpu-memory 10 5` for two GPUs. You can also set values in MiB like `--gpu-memory 3500MiB`. |
| `--cpu-memory CPU_MEMORY`                   | Maximum CPU memory in GiB to allocate for offloaded weights. Same as above.|
| `--disk`                                    | If the model is too large for your GPU(s) and CPU combined, send the remaining layers to the disk. |
| `--disk-cache-dir DISK_CACHE_DIR`           | Directory to save the disk cache to. Defaults to `cache/`. |
| `--load-in-8bit`                            | Load the model with 8-bit precision.|
| `--threshold`                               | Threshold for 8bit precision for older cards. It will use more memory while performing infrerence so watch out. NaN == too high. OOM == too low.|
| `--bf16`                                    | Load the model with bfloat16 precision. Requires NVIDIA Ampere GPU. |
| `--no-cache`                                | Set `use_cache` to False while generating text. This reduces the VRAM usage a bit with a performance cost. |
| `--xformers`                                | Use xformer's memory efficient attention. This should increase your tokens/s. |
| `--sdp-attention`                           | Use torch 2.0's sdp attention. |
| `--flash-attention`                         | Use HazyResearch's Flash Attention. This drastically reduces the VRAM cost |
| `--trust-remote-code`                       | Set trust_remote_code=True while loading a model. Necessary for ChatGLM and Falcon. |

#### Accelerate 4-bit

⚠️ Requires minimum compute of 7.0 on Windows at the moment.

| Flag                                        | Description |
|---------------------------------------------|-------------|
| `--load-in-4bit`                            | Load the model with 4-bit precision (using bitsandbytes). |
| `--compute_dtype COMPUTE_DTYPE`             | compute dtype for 4-bit. Valid options: bfloat16, float16, float32. |
| `--quant_type QUANT_TYPE`                   | quant_type for 4-bit. Valid options: nf4, fp4. |
| `--use_double_quant`                        | use_double_quant for 4-bit. |

#### llama.cpp

| Flag        | Description |
|-------------|-------------|
| `--threads` | Number of threads to use. |
| `--n_batch` | Maximum number of prompt tokens to batch together when calling llama_eval. |
| `--no-mmap` | Prevent mmap from being used. |
| `--mlock`   | Force the system to keep the model in RAM. |
| `--cache-capacity CACHE_CAPACITY`   | Maximum cache capacity. Examples: 2000MiB, 2GiB. When provided without units, bytes will be assumed. |
| `--n-gpu-layers N_GPU_LAYERS` | Number of layers to offload to the GPU. Only works if llama-cpp-python was compiled with BLAS. Set this to 1000000000 to offload all layers to the GPU. |
| `--n_ctx N_CTX` | Size of the prompt context. |
| `--llama_cpp_seed SEED` | Seed for llama-cpp models. Default 0 (random). |

#### AutoGPTQ

| Flag             | Description |
|------------------|-------------|
| `--triton`                     | Use triton. |
| `--quant_attn`  | Ennable the use of fused attention, faster but slightly more vram. |
| `--fused_mlp`        | Triton mode only: enable the use of fused MLP, which will use lots more vram. |
| `autogptq_act_order`                   | For models that don't have a quantize_config.json, this parameter is used to define whether to use group size and act_order together  |

#### ExLlama

| Flag             | Description |
|------------------|-------------|
|`--gpu-split`     | Comma-separated list of VRAM (in GB) to use per GPU device for model layers, e.g. `20,7,7` |
|`--nohalf2`       | Disable half2 so pascal can somewhat use exllama. its still not good  |
|`--max_seq_len MAX_SEQ_LEN`           | Maximum sequence length. |
|`--compress_pos_emb COMPRESS_POS_EMB` | Positional embeddings compression factor. Should typically be set to max_seq_len / 2048. |

#### GPTQ-for-LLaMa

| Flag                      | Description |
|---------------------------|-------------|
| `--wbits WBITS`           | GPTQ: Load a pre-quantized model with specified precision in bits. 2, 3, 4 and 8 are supported. |
| `--model_type MODEL_TYPE` | GPTQ: Model type of pre-quantized model. Currently LLaMA, OPT, GPT-NeoX, and GPT-J are supported. |
| `--groupsize GROUPSIZE`   | GPTQ: Group size. |
| `--pre_layer PRE_LAYER [PRE_LAYER ...]`  | The number of layers to allocate to the GPU. Setting this parameter enables CPU offloading for 4-bit models. For multi-gpu, write the numbers separated by spaces, eg `--pre_layer 30 60`. |
| `--checkpoint CHECKPOINT` | The path to the quantized checkpoint file. If not specified, it will be automatically detected. |
| `--autograd`   | GPTQ: Autograd implementation to use 4bit lora and run multiple models. Will now automatically select loader. |
| `--v1`   | GPTQ: Explicitly declare a GPTQv1 model to load into autograd. |
| `---mlp_attn`    | MLP attention hijack. Slightly faster inference. |
| `--quant_attn`         | (triton) Enable quant attention.
| `--warmup_autotune`    | (triton) Enable warmup autotune.
| `--fused_mlp`          | (triton) Enable fused mlp.
| `--autogptq`           | Load with autogptq. Look in shared.py for more options like triton or using act order w/ groupsize kernel


#### FlexGen

| Flag             | Description |
|------------------|-------------|
| `--percent PERCENT [PERCENT ...]` | FlexGen: allocation percentages. Must be 6 numbers separated by spaces (default: 0, 100, 100, 0, 100, 0). |
| `--compress-weight`               | FlexGen: Whether to compress weight (default: False).|
| `--pin-weight [PIN_WEIGHT]`       | FlexGen: whether to pin weights (setting this to False reduces CPU memory by 20%). |

#### DeepSpeed

| Flag                                  | Description |
|---------------------------------------|-------------|
| `--deepspeed`                         | Enable the use of DeepSpeed ZeRO-3 for inference via the Transformers integration. |
| `--nvme-offload-dir NVME_OFFLOAD_DIR` | DeepSpeed: Directory to use for ZeRO-3 NVME offloading. |
| `--local_rank LOCAL_RANK`             | DeepSpeed: Optional argument for distributed setups. |

#### RWKV

| Flag                            | Description |
|---------------------------------|-------------|
| `--rwkv-strategy RWKV_STRATEGY` | RWKV: The strategy to use while loading the model. Examples: "cpu fp32", "cuda fp16", "cuda fp16i8". |
| `--rwkv-cuda-on`                | RWKV: Compile the CUDA kernel for better performance. |

#### Gradio

| Flag                                  | Description |
|---------------------------------------|-------------|
| `--listen`                            | Make the web UI reachable from your local network. |
| `--listen-host LISTEN_HOST`           | The hostname that the server will use. |
| `--listen-port LISTEN_PORT`           | The listening port that the server will use. |
| `--share`                             | Create a public URL. This is useful for running the web UI on Google Colab or similar. |
| `--auto-launch`                       | Open the web UI in the default browser upon launch. |
| `--gradio-auth USER:PWD`              | set gradio authentication like "username:password"; or comma-delimit multiple like "u1:p1,u2:p2,u3:p3" |
| `--gradio-auth-path GRADIO_AUTH_PATH` | Set the gradio authentication file path. The file should contain one or more user:password pairs in this format: "u1:p1,u2:p2,u3:p3" |

#### API

| Flag                                  | Description |
|---------------------------------------|-------------|
| `--api`                               | Enable the API extension. |
| `--public-api`                        | Create a public URL for the API using Cloudfare. |
| `--api-blocking-port BLOCKING_PORT`   | The listening port for the blocking API. |
| `--api-streaming-port STREAMING_PORT` | The listening port for the streaming API. |

#### Multimodal

| Flag                                  | Description |
|---------------------------------------|-------------|
| `--multimodal-pipeline PIPELINE`      | The multimodal pipeline to use. Examples: `llava-7b`, `llava-13b`. |

Out of memory errors? [Check the low VRAM guide](docs/Low-VRAM-guide.md).

## Presets

Inference settings presets can be created under `presets/` as yaml files. These files are detected automatically at startup.

The presets that are included by default are the result of a contest that received 7215 votes. More details can be found [here](https://github.com/oobabooga/oobabooga.github.io/blob/main/arena/results.md).

## Contributing

* Pull requests, suggestions, and issue reports are welcome. 
* Make sure to carefully [search](https://github.com/oobabooga/text-generation-webui/issues) existing issues before starting a new one.
* If you have some experience with git, testing an open pull request and leaving a comment on whether it works as expected or not is immensely helpful.
* A simple way to contribute, even if you are not a programmer, is to leave a 👍 on an issue or pull request that you find relevant.

## Credits

- Gradio dropdown menu refresh button, code for reloading the interface: https://github.com/AUTOMATIC1111/stable-diffusion-webui
- Godlike preset: https://github.com/KoboldAI/KoboldAI-Client/wiki/Settings-Presets
- Code for some of the sliders: https://github.com/PygmalionAI/gradio-ui/
