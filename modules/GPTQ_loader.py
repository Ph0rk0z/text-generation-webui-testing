import inspect
import re
import sys
from pathlib import Path

import accelerate
import torch
import transformers
import modules.shared as shared

from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, str(Path("repositories/GPTQ-Merged/src/alpaca_lora_4bit")))
sys.path.insert(0, str(Path("repositories/GPTQ-Merged/src/gptq_llama")))

#from gptq_llama import quant, modelutils
from gptq_llama import llama_inference_offload 
#from offload import load_quant_offload
from modelutils import find_layers
from quant import make_quant

def _load_quant(model, checkpoint, wbits, groupsize=-1, faster_kernel=False, exclude_layers=['lm_head', 'embed_out'], kernel_switch_threshold=128):

    def noop(*args, **kwargs):
        pass

    config = AutoConfig.from_pretrained(model)
    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = AutoModelForCausalLM.from_config(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in exclude_layers:
        if name in layers:
            del layers[name]

    gptq_args = inspect.getfullargspec(make_quant).args

    make_quant_kwargs = {
        'module': model,
        'names': layers,
        'bits': wbits,
    }
    if 'groupsize' in gptq_args:
        make_quant_kwargs['groupsize'] = groupsize
    if 'faster' in gptq_args:
        make_quant_kwargs['faster'] = faster_kernel
    if 'kernel_switch_threshold' in gptq_args:
        make_quant_kwargs['kernel_switch_threshold'] = kernel_switch_threshold

    make_quant(**make_quant_kwargs)
    #make_quant(model, layers, wbits, groupsize)

    del layers

    print('Loading GPTQ model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint), strict=False)
    else:
        model.load_state_dict(torch.load(checkpoint), strict=False)
    model.seqlen = 2048
    print('Done.')

    return model



def load_quantized(model_name):
    if not(shared.args.gptq_bits):
       shared.args.gptq_bits = shared.args.wbits
    if (shared.args.model_type):
        shared.args.gptq_model_type = shared.args.model_type
    if not shared.args.gptq_model_type:
        # Try to determine model type from model name
        model_type = model_name.split('-')[0].lower()
        if model_type not in ('llama', 'opt', 'gptneox', 'gptj'):
            print("Can't determine model type from model name. Please specify it manually using --model_type argument")
            exit()
    else:
        model_type = shared.args.gptq_model_type.lower()

    if shared.args.pre_layer and model_type == 'llama':
        load_quant = llama_inference_offload.load_quant
#    elif shared.args.pre_layer and model_type== 'gptj':
#        load_quant = load_quant_offload
    elif model_type in ('llama', 'opt', 'gptneox'):
        if shared.args.pre_layer:
            print("Warning: ignoring --pre_layer because it only works for llama or gptj model type.")
        load_quant = _load_quant

    path_to_model = Path(f'models/{model_name}')
    found_pts = list(path_to_model.glob("*.pt"))
    found_safetensors = list(path_to_model.glob("*.safetensors"))
    pt_path = None
    
    if len(found_pts) > 0:
        pt_path = found_pts[-1]
    elif len(found_safetensors) > 0:
        pt_path = found_safetensors[-1]
    else: 
        pt_model = f'{model_name}-{shared.args.gptq_bits}bit'
    
      # Try to find the .safetensors or .pt both in the model dir and in the subfolder

        for path in [Path(p + ext) for ext in ['.safetensors', '.pt'] for p in [f"{shared.args.model_dir}/{pt_model}", f"{path_to_model}/{pt_model}"]]:
            if path.exists():
                pt_path = path
                break

    if not pt_path:
        print(f"Could not find {pt_model}, exiting...")
        exit()
    else:
        print(f"Found the following quantized model: {pt_path}")

    if shared.args.autograd:
      import autograd_4bit
      from autograd_4bit import Autograd4bitQuantLinear
      from autograd_4bit import load_llama_model_4bit_low_ram, load_auto_model_4bit_low_ram, load_llama_model_4bit_low_ram_and_offload, load_auto_model_4bit_low_ram_and_offload
      if (model_type== 'llama'):
         if shared.args.gpu_memory:
            memory_map = list(map(lambda x: x.strip(), shared.args.gpu_memory))
            max_cpu_memory = shared.args.cpu_memory.strip() if shared.args.cpu_memory is not None else '99GiB'
            max_memory = {}
            for i in range(len(memory_map)):
                max_memory[i] = f'{memory_map[i]}GiB' if not re.match('.*ib$', memory_map[i].lower()) else memory_map[i]
            max_memory['cpu'] = max_cpu_memory
            model, tokenizer = load_llama_model_4bit_low_ram_and_offload(path_to_model, f"{pt_path}", lora_path=None, groupsize=shared.args.groupsize, seqlen=2048, max_memory=max_memory, is_v1_model=shared.args.v1)  
         else:
            #from monkeypatch.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
            #replace_llama_attn_with_flash_attn()

            model, tokenizer = load_llama_model_4bit_low_ram(path_to_model, f"{pt_path}", groupsize=shared.args.groupsize, is_v1_model=shared.args.v1)
            

      else:
         if shared.args.gpu_memory:
            memory_map = list(map(lambda x: x.strip(), shared.args.gpu_memory))
            max_cpu_memory = shared.args.cpu_memory.strip() if shared.args.cpu_memory is not None else '99GiB'
            max_memory = {}
            for i in range(len(memory_map)):
                max_memory[i] = f'{memory_map[i]}GiB' if not re.match('.*ib$', memory_map[i].lower()) else memory_map[i]
            max_memory['cpu'] = max_cpu_memory
            model, tokenizer = load_auto_model_4bit_low_ram_and_offload(path_to_model, f"{pt_path}", lora_path=None, groupsize=shared.args.groupsize, seqlen=2048, max_memory=max_memory, is_v1_model=shared.args.v1)                   
         else:
            model, tokenizer = load_auto_model_4bit_low_ram(path_to_model, f"{pt_path}", groupsize=shared.args.groupsize, is_v1_model=shared.args.v1)

      print (shared.args.lora, shared.lora_name)

      if not shared.args.lora or shared.lora_name == "None":
         print('Apply auto switch and half. Lora:', shared.lora_name)
         for n, m in model.named_modules():
           if isinstance(m, Autograd4bitQuantLinear):
              if (shared.args.v1 == True):
                  m.zeros = m.zeros.half()
              m.scales = m.scales.half()
              m.bias = m.bias.half()
         autograd_4bit.use_new = True
         autograd_4bit.auto_switch = True
         #if any((shared.args.xformers, shared.args.sdp_attention)):
         #   if (model_type== 'llama'):
         #      from modules import llama_attn_hijack    
         #      llama_attn_hijack.hijack_llama_attention()

    # qwopqwop200's offload 
    elif model_type == 'llama' and shared.args.pre_layer:
        model = load_quant(str(path_to_model), str(pt_path), shared.args.wbits, shared.args.groupsize, shared.args.pre_layer)
#    elif model_type == 'gptj' and shared.args.pre_layer:
#        from gptq_llama import gptj
#        model = load_quant(_load_quant, str(path_to_model), str(pt_path), shared.args.wbits, shared.args.groupsize, gpu_layers=[10,0])
    else:
        threshold = False if model_type == ('gptj') else 128
        model = load_quant(str(path_to_model), str(pt_path), shared.args.wbits, shared.args.groupsize, kernel_switch_threshold=threshold)

        # accelerate offload (doesn't work properly)
        if shared.args.gpu_memory:
            memory_map = list(map(lambda x: x.strip(), shared.args.gpu_memory))
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
