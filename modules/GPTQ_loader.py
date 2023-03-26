import re
import sys
from pathlib import Path

import accelerate
import torch

import modules.shared as shared

sys.path.insert(0, str(Path("repositories/GPTQ-Merged")))
#sys.path.insert(0, str(Path("repositories/GPTQ-for-LLaMa")))
import llama
#import llama_inference_offload
import opt
import gptneox
import gptj

def load_quantized(model_name):
    if not shared.args.gptq_model_type:
        # Try to determine model type from model name
        model_type = model_name.split('-')[0].lower()
        if model_type not in ('llama', 'opt', 'gptneox', 'gptj'):
            print("Can't determine model type from model name. Please specify it manually using --gptq-model-type "
                  "argument")
            exit()
    else:
        model_type = shared.args.gptq_model_type.lower()

    if model_type == 'llama':
        if not shared.args.gptq_pre_layer:
            load_quant = llama.load_quant
        else:
            load_quant = llama_inference_offload.load_quant
    elif model_type == 'opt':
        load_quant = opt.load_quant
    elif model_type == 'gptneox':
        load_quant = gptneox.load_quant
    elif model_type == 'gptj':
        load_quant = gptj.load_quant
    else:
        print("Unknown pre-quantized model type specified. Only 'llama', 'opt', 'gptj', 'gptneox' are supported")
        exit()

    path_to_model = Path(f'models/{model_name}')
    if path_to_model.name.lower().startswith('llama-7b'):
        pt_model = f'llama-7b-{shared.args.gptq_bits}bit'
    elif path_to_model.name.lower().startswith('llama-13b'):
        pt_model = f'llama-13b-{shared.args.gptq_bits}bit'
    elif path_to_model.name.lower().startswith('llama-30b'):
        pt_model = f'llama-30b-{shared.args.gptq_bits}bit'
    elif path_to_model.name.lower().startswith('llama-65b'):
        pt_model = f'llama-65b-{shared.args.gptq_bits}bit'
    else:
        pt_model = f'{model_name}-{shared.args.gptq_bits}bit'

    # Try to find the .safetensors or .pt both in models/ and in the subfolder
    pt_path = None
    for path in [Path(p+ext) for ext in ['.safetensors', '.pt'] for p in [f"models/{pt_model}", f"{path_to_model}/{pt_model}"]]:
        if path.exists():
            print(f"Found {path}")
            pt_path = path
            break

    if not pt_path:
        print(f"Could not find {pt_model}, exiting...")
        exit()

    if shared.args.autograd:
      import autograd_4bit
      from autograd_4bit import Autograd4bitQuantLinear
      if model_type == 'opt':
         from autograd_4bit import load_opt_model_4bit_low_ram
         model, tokenizer = load_opt_model_4bit_low_ram(path_to_model, f"models/{pt_model}.pt" )

      elif model_type == 'llama':
           from autograd_4bit import load_llama_model_4bit_low_ram
           model, tokenizer = load_llama_model_4bit_low_ram(path_to_model, f"models/{pt_model}.pt" )

      else:
          print (f" Error: {model_type}, {path_to_model}")

      print (shared.args.lora, shared.lora_name)

      if not shared.args.lora or shared.lora_name == "None":
         print('Apply auto switch and half. Lora:', shared.lora_name)
         for n, m in model.named_modules():
           if isinstance(m, Autograd4bitQuantLinear):
              m.zeros = m.zeros.half()
              m.scales = m.scales.half()
              m.bias = m.bias.half()
         autograd_4bit.use_new = True
         autograd_4bit.auto_switch = True
       
    # qwopqwop200's offload
    elif shared.args.gptq_pre_layer:
        model = load_quant(str(path_to_model), str(pt_path), shared.args.gptq_bits, shared.args.gptq_pre_layer)
    else:
        model = load_quant(str(path_to_model), str(pt_path), shared.args.gptq_bits)
        print ('Load Quant')
        # accelerate offload (doesn't work properly)
        if shared.args.gpu_memory:
            memory_map = list(map(lambda x : x.strip(), shared.args.gpu_memory))
            max_cpu_memory = shared.args.cpu_memory.strip() if shared.args.cpu_memory is not None else '99GiB'
            max_memory = {}
            for i in range(len(memory_map)):
                max_memory[i] = f'{memory_map[i]}GiB' if not re.match('.*ib$', memory_map[i].lower()) else memory_map[i]
            max_memory['cpu'] = max_cpu_memory

            device_map = accelerate.infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=["LlamaDecoderLayer"])
            print("Using the following device map for the 4-bit model:", device_map)
            # https://huggingface.co/docs/accelerate/package_reference/big_modeling#accelerate.dispatch_model
            model = accelerate.dispatch_model(model, device_map=device_map, offload_buffers=True)

        # No offload
        elif not shared.args.cpu:
            model = model.to(torch.device('cuda:0'))

    return model
