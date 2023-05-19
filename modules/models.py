import gc
import json
import logging
import os
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import transformers
from accelerate import infer_auto_device_map, init_empty_weights
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                          AutoModelForSeq2SeqLM, AutoTokenizer,
                          BitsAndBytesConfig, LlamaTokenizer)

import modules.shared as shared
from modules import llama_attn_hijack

transformers.logging.set_verbosity_error()

local_rank = None
if shared.args.deepspeed:
    import deepspeed
    from transformers.deepspeed import (HfDeepSpeedConfig,
                                        is_deepspeed_zero3_enabled)

    from modules.deepspeed_parameters import generate_ds_config

    # Distributed setup
    local_rank = shared.args.local_rank if shared.args.local_rank is not None else int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()
    ds_config = generate_ds_config(shared.args.bf16, 1 * world_size, shared.args.nvme_offload_dir)
    dschf = HfDeepSpeedConfig(ds_config)  # Keep this object alive for the Transformers integration


# Some models require special treatment in various parts of the code.
# This function detects those models
def find_model_type(model_name):
    path_to_model = Path(f'{shared.args.model_dir}/{model_name}')
    if not path_to_model.exists():
        return 'None'

    model_name_lower = model_name.lower()
    if 'rwkv-' in model_name_lower:
        return 'rwkv'
    elif len(list(path_to_model.glob('*ggml*.bin'))) > 0:
        return 'llamacpp'
    elif re.match('.*ggml.*\.bin', model_name_lower):
        return 'llamacpp'
    elif 'chatglm' in model_name_lower:
        return 'chatglm'
    elif 'galactica' in model_name_lower:
        return 'galactica'
    elif 'llava' in model_name_lower:
        return 'llava'
    elif 'oasst' in model_name_lower:
        return 'oasst'
    elif any((k in model_name_lower for k in ['gpt4chan', 'gpt-4chan'])):
        return 'gpt4chan'
    else:
        config = AutoConfig.from_pretrained(path_to_model, trust_remote_code=shared.args.trust_remote_code)
        # Not a "catch all", but fairly accurate
        if config.to_dict().get("is_encoder_decoder", False):
            return 'HF_seq2seq'
        else:
            return 'HF_generic'


def load_model(model_name):
    logging.info(f"Loading {model_name}...")
    t0 = time.time()

    shared.model_type = find_model_type(model_name)
    if shared.model_type == 'None':
        logging.error('The path to the model does not exist. Exiting.')
        return None, None

    if shared.args.wbits > 0:
        load_func = GPTQ_loader
    elif shared.model_type == 'llamacpp':
        load_func = llamacpp_loader
    elif shared.model_type == 'rwkv':
        load_func = RWKV_loader
    elif shared.args.flexgen:
        load_func = flexgen_loader
    else:
        load_func = huggingface_loader

    output = load_func(model_name)
    if type(output) is tuple:
        model, tokenizer = output
    else:
        model = output
        if model is None:
            return None, None
        else:
            tokenizer = load_tokenizer(model_name, model)

    # Hijack attention with xformers
    if any((shared.args.xformers, shared.args.sdp_attention)):
        llama_attn_hijack.hijack_llama_attention()

    logging.info(f"Loaded the model in {(time.time()-t0):.2f} seconds.\n")
    return model, tokenizer


def load_tokenizer(model_name, model):
    tokenizer = None
    if shared.model_type == 'gpt4chan' and Path(f"{shared.args.model_dir}/gpt-j-6B/").exists():
        tokenizer = AutoTokenizer.from_pretrained(Path(f"{shared.args.model_dir}/gpt-j-6B/"))
    elif type(model) is transformers.LlamaForCausalLM:
        # Try to load an universal LLaMA tokenizer
        if shared.model_type not in ['llava', 'oasst']:
            for p in [Path(f"{shared.args.model_dir}/llama-tokenizer/"), Path(f"{shared.args.model_dir}/oobabooga_llama-tokenizer/")]:
                if p.exists():
                    logging.info(f"Loading the universal LLaMA tokenizer from {p}...")
                    tokenizer = LlamaTokenizer.from_pretrained(p, clean_up_tokenization_spaces=True)
                    return tokenizer

        # Otherwise, load it from the model folder and hope that these
        # are not outdated tokenizer files.
        tokenizer = LlamaTokenizer.from_pretrained(Path(f"{shared.args.model_dir}/{model_name}/"), clean_up_tokenization_spaces=True)
        try:
            tokenizer.eos_token_id = 2
            tokenizer.bos_token_id = 1
            tokenizer.pad_token_id = 0
        except:
            pass
    else:
        path_to_model = Path(f"{shared.args.model_dir}/{model_name}/")
        if path_to_model.exists():
            tokenizer = AutoTokenizer.from_pretrained(path_to_model, trust_remote_code=shared.args.trust_remote_code)

    return tokenizer


def huggingface_loader(model_name):
    if shared.model_type == 'chatglm':
        LoaderClass = AutoModel
    elif shared.model_type == 'HF_seq2seq':
        LoaderClass = AutoModelForSeq2SeqLM
    else:
        LoaderClass = AutoModelForCausalLM

    # Load the model in simple 16-bit mode by default
    if not any([shared.args.cpu, shared.args.load_in_8bit, shared.args.auto_devices, shared.args.disk, shared.args.deepspeed, shared.args.gpu_memory is not None, shared.args.cpu_memory is not None]):
        model = LoaderClass.from_pretrained(Path(f"{shared.args.model_dir}/{model_name}"), low_cpu_mem_usage=True, torch_dtype=torch.bfloat16 if shared.args.bf16 else torch.float16, trust_remote_code=shared.args.trust_remote_code)
        if torch.has_mps:
            device = torch.device('mps')
            model = model.to(device)
        else:
            model = model.cuda()

    # DeepSpeed ZeRO-3
    elif shared.args.deepspeed:
        model = LoaderClass.from_pretrained(Path(f"{shared.args.model_dir}/{model_name}"), torch_dtype=torch.bfloat16 if shared.args.bf16 else torch.float16)
        model = deepspeed.initialize(model=model, config_params=ds_config, model_parameters=None, optimizer=None, lr_scheduler=None)[0]
        model.module.eval()  # Inference
        logging.info(f"DeepSpeed ZeRO-3 is enabled: {is_deepspeed_zero3_enabled()}")


    # Custom
    else:
        params = {
            "low_cpu_mem_usage": True,
            "trust_remote_code": shared.args.trust_remote_code
        }

        if not any((shared.args.cpu, torch.cuda.is_available(), torch.has_mps)):
            logging.warning("torch.cuda.is_available() returned False. This means that no GPU has been detected. Falling back to CPU mode.")
            shared.args.cpu = True

        if shared.args.cpu:
            params["torch_dtype"] = torch.float32
        else:
            params["device_map"] = 'auto'
            if shared.args.load_in_8bit and any((shared.args.auto_devices, shared.args.gpu_memory)):
                params['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=float(shared.args.threshold), llm_int8_enable_fp32_cpu_offload=True)
            elif shared.args.load_in_8bit:
                params['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=float(shared.args.threshold))
            elif shared.args.bf16:
                params["torch_dtype"] = torch.bfloat16
            else:
                params["torch_dtype"] = torch.float16

            params['max_memory'] = get_max_memory_dict()
            if shared.args.disk:
                params["offload_folder"] = shared.args.disk_cache_dir

        checkpoint = Path(f'{shared.args.model_dir}/{model_name}')
        if shared.args.load_in_8bit and params.get('max_memory', None) is not None and params['device_map'] == 'auto':
            config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=shared.args.trust_remote_code)
            with init_empty_weights():
                model = LoaderClass.from_config(config, trust_remote_code=shared.args.trust_remote_code)

            model.tie_weights()
            params['device_map'] = infer_auto_device_map(
                model,
                dtype=torch.int8,
                max_memory=params['max_memory'],
                no_split_module_classes=model._no_split_modules
            )

        model = LoaderClass.from_pretrained(checkpoint, **params)

    return model


def flexgen_loader(model_name):
    from flexgen.flex_opt import CompressionConfig, ExecutionEnv, OptLM, Policy

    # Initialize environment
    env = ExecutionEnv.create(shared.args.disk_cache_dir)

    # Offloading policy
    policy = Policy(1, 1,
                    shared.args.percent[0], shared.args.percent[1],
                    shared.args.percent[2], shared.args.percent[3],
                    shared.args.percent[4], shared.args.percent[5],
                    overlap=True, sep_layer=True, pin_weight=shared.args.pin_weight,
                    cpu_cache_compute=False, attn_sparsity=1.0,
                    compress_weight=shared.args.compress_weight,
                    comp_weight_config=CompressionConfig(
                        num_bits=4, group_size=64,
                        group_dim=0, symmetric=False),
                    compress_cache=False,
                    comp_cache_config=CompressionConfig(
                        num_bits=4, group_size=64,
                        group_dim=2, symmetric=False))

    model = OptLM(f"facebook/{model_name}", env, shared.args.model_dir, policy)
    return model


def RWKV_loader(model_name):
    from modules.RWKV import RWKVModel, RWKVTokenizer

    model = RWKVModel.from_pretrained(Path(f'{shared.args.model_dir}/{model_name}'), dtype="fp32" if shared.args.cpu else "bf16" if shared.args.bf16 else "fp16", device="cpu" if shared.args.cpu else "cuda")
    tokenizer = RWKVTokenizer.from_pretrained(Path(shared.args.model_dir))
    return model, tokenizer


def llamacpp_loader(model_name):
    from modules.llamacpp_model import LlamaCppModel

    path = Path(f'{shared.args.model_dir}/{model_name}')
    if path.is_file():
        model_file = path
    else:
        model_file = list(Path(f'{shared.args.model_dir}/{model_name}').glob('*ggml*.bin'))[0]

    logging.info(f"llama.cpp weights detected: {model_file}\n")
    model, tokenizer = LlamaCppModel.from_pretrained(model_file)
    return model, tokenizer


def GPTQ_loader(model_name):

    #AutoGPTQ
    if shared.args.autogptq:
        from modules.AutoGPTQ_loader import load_quantized

        model = load_quantized(model_name)
    #Autograd and GPTQ
    else:
        from modules.GPTQ_loader import load_quantized

        model = load_quantized(model_name)

    return model


def get_max_memory_dict():
    max_memory = {}
    if shared.args.gpu_memory:
        memory_map = list(map(lambda x: x.strip(), shared.args.gpu_memory))
        for i in range(len(memory_map)):
            max_memory[i] = f'{memory_map[i]}GiB' if not re.match('.*ib$', memory_map[i].lower()) else memory_map[i]

        max_cpu_memory = shared.args.cpu_memory.strip() if shared.args.cpu_memory is not None else '99GiB'
        max_memory['cpu'] = f'{max_cpu_memory}GiB' if not re.match('.*ib$', max_cpu_memory.lower()) else max_cpu_memory

    # If --auto-devices is provided standalone, try to get a reasonable value
    # for the maximum memory of device :0
    elif shared.args.auto_devices:
        total_mem = (torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
        suggestion = round((total_mem - 1000) / 1000) * 1000
        if total_mem - suggestion < 800:
            suggestion -= 1000

        suggestion = int(round(suggestion / 1000))
        logging.warning(f"Auto-assiging --gpu-memory {suggestion} for your GPU to try to prevent out-of-memory errors. You can manually set other values.")
        max_memory = {0: f'{suggestion}GiB', 'cpu': f'{shared.args.cpu_memory or 99}GiB'}

    return max_memory if len(max_memory) > 0 else None


def clear_torch_cache():
    gc.collect()
    if not shared.args.cpu:
        torch.cuda.empty_cache()


def unload_model():
    shared.model = shared.tokenizer = None
    clear_torch_cache()


def reload_model():
    unload_model()
    shared.model, shared.tokenizer = load_model(shared.model_name)


def load_soft_prompt(name):
    if name == 'None':
        shared.soft_prompt = False
        shared.soft_prompt_tensor = None
    else:
        with zipfile.ZipFile(Path(f'softprompts/{name}.zip')) as zf:
            zf.extract('tensor.npy')
            zf.extract('meta.json')
            j = json.loads(open('meta.json', 'r').read())
            logging.info(f"\nLoading the softprompt \"{name}\".")
            for field in j:
                if field != 'name':
                    if type(j[field]) is list:
                        logging.info(f"{field}: {', '.join(j[field])}")
                    else:
                        logging.info(f"{field}: {j[field]}")

            logging.info()
            tensor = np.load('tensor.npy')
            Path('tensor.npy').unlink()
            Path('meta.json').unlink()

        tensor = torch.Tensor(tensor).to(device=shared.model.device, dtype=shared.model.dtype)
        tensor = torch.reshape(tensor, (1, tensor.shape[0], tensor.shape[1]))
        shared.soft_prompt = True
        shared.soft_prompt_tensor = tensor

    return name
