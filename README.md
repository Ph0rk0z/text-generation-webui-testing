# Text generation web UI Testing
### Here there be dragons. (But V1 and V2 GPTQ support)


- Allow 4bit loras and use of the --autograd implementation.
- Use GPT-J 4-bits (GPTQv1/v2)
- GPT-NeoXT 4-bits (GPTQv1/v2)
- 8 bit threshold hardcoded to 1.5 for now (Pre 7.0 compute)
- load 4-bit lora from web ui
- V1 Models work in --autograd (declare with --v1)
- V2 Models work in both.
- Offloading works in autograd with --gpu-memory but doesn't 100% hodl while generating
- Offloading for llama GPTQ works
- Offloading for all other GPTQ models getting worked on
- Probably not as fast as the old version (still figuring this out)
- 4bit loras only work in autograd


#### Depends on:
https://github.com/Ph0rk0z/GPTQ-Merged (dual module branch)

https://github.com/sterlind/peft

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

* Dropdown menu for switching between models
* Notebook mode that resembles OpenAI's playground
* Chat mode for conversation and role playing
* Instruct mode compatible with Alpaca, Vicuna, and Open Assistant formats **\*NEW!\***
* Nice HTML output for GPT-4chan
* Markdown output for [GALACTICA](https://github.com/paperswithcode/galai), including LaTeX rendering
* [Custom chat characters](https://github.com/oobabooga/text-generation-webui/wiki/Custom-chat-characters)
* Advanced chat features (send images, get audio responses with TTS)
* Very efficient text streaming
* Parameter presets
* 8-bit mode
* Layers splitting across GPU(s), CPU, and disk
* CPU mode
* [FlexGen](https://github.com/oobabooga/text-generation-webui/wiki/FlexGen)
* [DeepSpeed ZeRO-3](https://github.com/oobabooga/text-generation-webui/wiki/DeepSpeed)
* API [with](https://github.com/oobabooga/text-generation-webui/blob/main/api-example-stream.py) streaming and [without](https://github.com/oobabooga/text-generation-webui/blob/main/api-example.py) streaming
* [LLaMA model, including 4-bit GPTQ](https://github.com/oobabooga/text-generation-webui/wiki/LLaMA-model)
* [llama.cpp](https://github.com/oobabooga/text-generation-webui/wiki/llama.cpp-models) **\*NEW!\***
* [RWKV model](https://github.com/oobabooga/text-generation-webui/wiki/RWKV-model)
* [LoRA (loading and training)](https://github.com/oobabooga/text-generation-webui/wiki/Using-LoRAs)
* Softprompts
* [Extensions](https://github.com/oobabooga/text-generation-webui/wiki/Extensions)

## Installation

### One-click installers

[oobabooga-windows.zip](https://github.com/oobabooga/text-generation-webui/releases/download/installers/oobabooga-windows.zip)

Just download the zip above, extract it, and double click on "install". The web UI and all its dependencies will be installed in the same folder.

* To download a model, double click on "download-model"
* To start the web UI, double click on "start-webui" 

Source codes: https://github.com/oobabooga/one-click-installers

> **Note**
> 
> Thanks to [@jllllll](https://github.com/jllllll) and [@ClayShoaf](https://github.com/ClayShoaf), the Windows 1-click installer now sets up 8-bit and 4-bit requirements out of the box. No additional installation steps are necessary.

> **Note**
> 
> There is no need to run the installer as admin.

### Manual installation using Conda

Recommended if you have some experience with the command-line.

On Windows, I additionally recommend carrying out the installation on WSL instead of the base system: [WSL installation guide](https://github.com/oobabooga/text-generation-webui/wiki/WSL-installation-guide).

#### 0. Install Conda

https://docs.conda.io/en/latest/miniconda.html

On Linux or WSL, it can be automatically installed with these two commands:

```
curl -sL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > "Miniconda3.sh"
bash Miniconda3.sh
```
Source: https://educe-ubc.github.io/conda.html

#### 0.1 (Ubuntu/WSL) Install build tools

```
sudo apt install build-essential
```


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

The up to date commands can be found here: https://pytorch.org/get-started/locally/. 

#### 2.1 Special instructions

* MacOS users: https://github.com/oobabooga/text-generation-webui/pull/393
* AMD users: https://rentry.org/eq3hg

#### 3. Install the web UI

```
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui
pip install -r requirements.txt
```

> **Note**
> 
> For bitsandbytes and `--load-in-8bit` to work on Linux/WSL, this dirty fix is currently necessary: https://github.com/oobabooga/text-generation-webui/issues/400#issuecomment-1474876859


### Alternative: manual Windows installation

As an alternative to the recommended WSL method, you can install the web UI natively on Windows using this guide. It will be a lot harder and the performance may be slower: [Windows installation guide](https://github.com/oobabooga/text-generation-webui/wiki/Windows-installation-guide).

### Alternative: Docker

```
cp .env.example .env
docker compose up --build
```

Make sure to edit `.env.example` and set the appropriate CUDA version for your GPU, which can be found on [developer.nvidia.com](https://developer.nvidia.com/cuda-gpus).

You need to have docker compose v2.17 or higher installed in your system. For installation instructions, see [Docker compose installation](https://github.com/oobabooga/text-generation-webui/wiki/Docker-compose-installation).

Contributed by [@loeken](https://github.com/loeken) in [#633](https://github.com/oobabooga/text-generation-webui/pull/633)

### Updating the requirements

From time to time, the `requirements.txt` changes. To update, use this command:

```
conda activate textgen
cd text-generation-webui
pip install -r requirements.txt --upgrade
```
## Downloading models

Models should be placed inside the `models` folder.

[Hugging Face](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads) is the main place to download models. These are some examples:

* [Pythia](https://huggingface.co/models?sort=downloads&search=eleutherai%2Fpythia+deduped)
* [OPT](https://huggingface.co/models?search=facebook/opt)
* [GALACTICA](https://huggingface.co/models?search=facebook/galactica)
* [GPT-J 6B](https://huggingface.co/EleutherAI/gpt-j-6B/tree/main)

You can automatically download a model from HF using the script `download-model.py`:

    python download-model.py organization/model

For example:

    python download-model.py facebook/opt-1.3b

If you want to download a model manually, note that all you need are the json, txt, and pytorch\*.bin (or model*.safetensors) files. The remaining files are not necessary.

#### GPT-4chan

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
| `--model MODEL`                            | Name of the model to load by default. |
| `--lora LORA`                              | Name of the LoRA to apply to the model by default. |
| `--model-dir MODEL_DIR`                    | Path to directory with all the models. |
| `--lora-dir LORA_DIR`                      | Path to directory with all the loras. |
| `--model-menu`                             | Show a model menu in the terminal when the web UI is first launched. |
| `--no-stream`                              | Don't stream the text output in real time. |
| `--settings SETTINGS_FILE`                 | Load the default interface settings from this json file. See `settings-template.json` for an example. If you create a file called `settings.json`, this file will be loaded by default without the need to use the `--settings` flag. |
| `--extensions EXTENSIONS [EXTENSIONS ...]` | The list of extensions to load. If you want to load more than one extension, write the names separated by spaces. |
| `--verbose`                                | Print the prompts to the terminal. |

#### Accelerate/transformers

| Flag                                        | Description |
|---------------------------------------------|-------------|
| `--cpu`                                     | Use the CPU to generate text. Warning: Training on CPU is extremely slow.|
| `--auto-devices`                            | Automatically split the model across the available GPU(s) and CPU. |
|  `--gpu-memory GPU_MEMORY [GPU_MEMORY ...]` | Maxmimum GPU memory in GiB to be allocated per GPU. Example: `--gpu-memory 10` for a single GPU, `--gpu-memory 10 5` for two GPUs. You can also set values in MiB like `--gpu-memory 3500MiB`. |
| `--cpu-memory CPU_MEMORY`                   | Maximum CPU memory in GiB to allocate for offloaded weights. Same as above.|
| `--disk`                                    | If the model is too large for your GPU(s) and CPU combined, send the remaining layers to the disk. |
| `--disk-cache-dir DISK_CACHE_DIR`           | Directory to save the disk cache to. Defaults to `cache/`. |
| `--load-in-8bit`                            | Load the model with 8-bit precision.|
| `--threshold`                               | Threshold for 8bit precision for older cards. It will use more memory while performing infrerence so watch out. NaN == too high. OOM == too low.|
| `--bf16`                                    | Load the model with bfloat16 precision. Requires NVIDIA Ampere GPU. |
| `--no-cache`                                | Set `use_cache` to False while generating text. This reduces the VRAM usage a bit with a performance cost. |
| `--xformers`                                | Use xformer's memory efficient attention. This should increase your tokens/s. |
| `--sdp-attention`                           | Use torch 2.0's sdp attention. |

#### llama.cpp

| Flag        | Description |
|-------------|-------------|
| `--threads` | Number of threads to use in llama.cpp. |

#### GPTQ

| Flag                      | Description |
|---------------------------|-------------|
| `--wbits WBITS`           | GPTQ: Load a pre-quantized model with specified precision in bits. 2, 3, 4 and 8 are supported. |
| `--model_type MODEL_TYPE` | GPTQ: Model type of pre-quantized model. Currently LLaMA, OPT, and GPT-J are supported. |
| `--groupsize GROUPSIZE`   | GPTQ: Group size. |
| `--pre_layer PRE_LAYER`   | GPTQ: The number of layers to allocate to the GPU. Setting this parameter enables CPU offloading for 4-bit models. |
| `--autograd`   | GPTQ: Autograd implementation to use 4bit lora and run multiple models |
| `--v1`   | GPTQ: Explicitly declare a GPTQv1 model to load into autograd. |


#### FlexGen

| Flag             | Description |
|------------------|-------------|
| `--flexgen`                       | Enable the use of FlexGen offloading. |
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
| `--listen-port LISTEN_PORT`           | The listening port that the server will use. |
| `--share`                             | Create a public URL. This is useful for running the web UI on Google Colab or similar. |
| `--auto-launch`                       | Open the web UI in the default browser upon launch. |
| `--gradio-auth-path GRADIO_AUTH_PATH` | Set the gradio authentication file path. The file should contain one or more user:password pairs in this format: "u1:p1,u2:p2,u3:p3" |

Out of memory errors? [Check the low VRAM guide](https://github.com/oobabooga/text-generation-webui/wiki/Low-VRAM-guide).

## Presets

Inference settings presets can be created under `presets/` as text files. These files are detected automatically at startup.

By default, 10 presets by NovelAI and KoboldAI are included. These were selected out of a sample of 43 presets after applying a K-Means clustering algorithm and selecting the elements closest to the average of each cluster.

[Visualization](https://user-images.githubusercontent.com/112222186/228956352-1addbdb9-2456-465a-b51d-089f462cd385.png)

## System requirements

Check the [wiki](https://github.com/oobabooga/text-generation-webui/wiki/System-requirements) for some examples of VRAM and RAM usage in both GPU and CPU mode.

## Contributing

Pull requests, suggestions, and issue reports are welcome. 

You are also welcome to review open pull requests.

Before reporting a bug, make sure that you have:

1. Created a conda environment and installed the dependencies exactly as in the *Installation* section above.
2. [Searched](https://github.com/oobabooga/text-generation-webui/issues) to see if an issue already exists for the issue you encountered.

## Credits

- Gradio dropdown menu refresh button, code for reloading the interface: https://github.com/AUTOMATIC1111/stable-diffusion-webui
- Verbose preset: Anonymous 4chan user.
- NovelAI and KoboldAI presets: https://github.com/KoboldAI/KoboldAI-Client/wiki/Settings-Presets
- Code for early stopping in chat mode, code for some of the sliders: https://github.com/PygmalionAI/gradio-ui/
