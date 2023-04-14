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

def finalize_autograd (model):
    import autograd_4bit
    from autograd_4bit import Autograd4bitQuantLinear
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
         #from amp_wrapper import AMPWrapper
         #wrapper = AMPWrapper(model)
         #wrapper.apply_generate()
    print('Apply auto switch and half. Lora:', shared.lora_name)


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

    try:
        from quant import autotune_warmup, make_quant_attn
        # triton branch
        make_quant_attn(model)
        if not shared.args.no_warmup_autotune:
            autotune_warmup(model)
    except ImportError:  # not triton branch
        pass

    model.seqlen = 2048
    print('Done.')

    return model



def load_quantized(model_name):

    # Find the model type
    if not shared.args.model_type:
        name = model_name.lower()
        if any((k in name for k in ['llama', 'alpaca', 'vicuna'])):
            model_type = 'llama'
        elif any((k in name for k in ['opt-', 'galactica'])):
            model_type = 'opt'
        elif any((k in name for k in ['gpt-j', 'pygmalion-6b'])):
            model_type = 'gptj'
        elif any((k in name for k in ['oasst', 'pythia-12b', 'lotus-12b', 'gpt-neoxt'])):
            model_type = 'gptj'
        else:
            print("Can't determine model type from model name. Please specify it manually using --model_type "
                  "argument")
            exit()
    else:
        model_type = shared.args.model_type.lower()

    # Select the appropriate load_quant function
    if shared.args.pre_layer and model_type == 'llama':
        load_quant = llama_inference_offload.load_quant
#    elif shared.args.pre_layer and model_type== 'gptj':
#        load_quant = load_quant_offload
    elif model_type in ('llama', 'opt', 'gptneox', 'gptj'):
        if shared.args.pre_layer:
            print("Warning: ignoring --pre_layer because it only works for llama or gptj model type.")
        load_quant = _load_quant

    # Locate the quantized model file
    path_to_model = Path(f'{shared.args.model_dir}/{model_name}')
    pt_path = None
    priority_name_list = [
        Path(f'{shared.args.model_dir}/{model_name}{hyphen}{shared.args.wbits}bit{group}{ext}')
        for group in ([f'-{shared.args.groupsize}g', ''] if shared.args.groupsize > 0 else [''])
        for ext in ['.safetensors', '.pt']
        for hyphen in ['-', f'/{model_name}-', '/']
    ]
    for path in priority_name_list:
        if path.exists():
            pt_path = path
            break

    # If the model hasn't been found with a well-behaved name, pick the last .pt
    # or the last .safetensors found in its folder as a last resort
    if not pt_path:
        path_to_model = Path(f'{shared.args.model_dir}/{model_name}')
        found_pts = list(path_to_model.glob("*.pt"))
        found_safetensors = list(path_to_model.glob("*.safetensors"))
        pt_path = None

        if len(found_pts) > 0:
            if len(found_pts) > 1:
                print('Warning: more than one .pt model has been found. The last one will be selected. It could be wrong.')
            pt_path = found_pts[-1]
        elif len(found_safetensors) > 0:
            if len(found_pts) > 1:
                print('Warning: more than one .safetensors model has been found. The last one will be selected. It could be wrong.')
            pt_path = found_safetensors[-1]

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
         if shared.args.gpu_memory or torch.cuda.device_count() > 1:
            if shared.args.gpu_memory:
                memory_map = list(map(lambda x: x.strip(), shared.args.gpu_memory))
                max_cpu_memory = shared.args.cpu_memory.strip() if shared.args.cpu_memory is not None else '99GiB'
                max_memory = {}
                for i in range(len(memory_map)):
                    max_memory[i] = f'{memory_map[i]}GiB' if not re.match('.*ib$', memory_map[i].lower()) else memory_map[i]
                max_memory['cpu'] = max_cpu_memory
            else:
                max_memory = accelerate.utils.get_balanced_memory(model)
            model, tokenizer = load_llama_model_4bit_low_ram_and_offload(path_to_model, f"{pt_path}", lora_path=None, groupsize=shared.args.groupsize, seqlen=2048, max_memory=max_memory, is_v1_model=shared.args.v1)  
         else:
            #from monkeypatch.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
            #replace_llama_attn_with_flash_attn()

            model, tokenizer = load_llama_model_4bit_low_ram(path_to_model, f"{pt_path}", groupsize=shared.args.groupsize, is_v1_model=shared.args.v1)
            

      else:
         if shared.args.gpu_memory or torch.cuda.device_count() > 1:
            if shared.args.gpu_memory:
                memory_map = list(map(lambda x: x.strip(), shared.args.gpu_memory))
                max_cpu_memory = shared.args.cpu_memory.strip() if shared.args.cpu_memory is not None else '99GiB'
                max_memory = {}
                for i in range(len(memory_map)):
                    max_memory[i] = f'{memory_map[i]}GiB' if not re.match('.*ib$', memory_map[i].lower()) else memory_map[i]
                max_memory['cpu'] = max_cpu_memory
            else:
                max_memory = accelerate.utils.get_balanced_memory(model)
            model, tokenizer = load_auto_model_4bit_low_ram_and_offload(path_to_model, f"{pt_path}", lora_path=None, groupsize=shared.args.groupsize, seqlen=2048, max_memory=max_memory, is_v1_model=shared.args.v1)                   
         else:
            model, tokenizer = load_auto_model_4bit_low_ram(path_to_model, f"{pt_path}", groupsize=shared.args.groupsize, is_v1_model=shared.args.v1)

      print ('Lora arguments:', shared.args.lora, shared.lora_name)

      if not shared.args.lora or shared.lora_name == "None":
         finalize_autograd(model)
         return model #let textgen handle the tokenizer

    # qwopqwop200's offload 
    elif model_type == 'llama' and shared.args.pre_layer:
        model = load_quant(str(path_to_model), str(pt_path), shared.args.wbits, shared.args.groupsize, shared.args.pre_layer)
    else:
        threshold = False if model_type == ('gptj') else 128
        model = _load_quant(str(path_to_model), str(pt_path), shared.args.wbits, shared.args.groupsize, kernel_switch_threshold=threshold)

        # accelerate offload (doesn't work properly)
        if shared.args.gpu_memory or torch.cuda.device_count() > 1:
            if shared.args.gpu_memory:
                memory_map = list(map(lambda x: x.strip(), shared.args.gpu_memory))
                max_cpu_memory = shared.args.cpu_memory.strip() if shared.args.cpu_memory is not None else '99GiB'
                max_memory = {}
                for i in range(len(memory_map)):
                    max_memory[i] = f'{memory_map[i]}GiB' if not re.match('.*ib$', memory_map[i].lower()) else memory_map[i]
                max_memory['cpu'] = max_cpu_memory
            else:
                max_memory = accelerate.utils.get_balanced_memory(model)

            device_map = accelerate.infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=["LlamaDecoderLayer", "GPTJBlock", "OPTDecoderLayer", "GPTNeoXLayer"])
            print("Using the following device map for the quantized model:", device_map)
            # https://huggingface.co/docs/accelerate/package_reference/big_modeling#accelerate.dispatch_model
            model = accelerate.dispatch_model(model, device_map=device_map, offload_buffers=True)

        # No offload
        elif not shared.args.cpu:
            model = model.to(torch.device('cuda:0'))

    
    return model
