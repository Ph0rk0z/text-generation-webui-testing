import inspect
import re
import sys
from pathlib import Path

import accelerate
import torch
import transformers
import modules.shared as shared

from accelerate.utils import is_xpu_available
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, str(Path("repositories/GPTQ-Merged/src/alpaca_lora_4bit")))
sys.path.insert(0, str(Path("repositories/GPTQ-Merged/src/gptq_llama")))
#sys.path.insert(0, str(Path("repositories/GPTQ-for-LLaMa")))
import modules.shared as shared
from modules.logging_colors import logger

#from gptq_llama import llama_inference_offload
#import llama_inference_offload 
#from offload import load_quant_offload

try:
    from gptq_llama import modelutils
    from modelutils import find_layers
except ImportError:
    from utils import find_layers

import time
from colorama import init, Fore, Back, Style

import autograd_4bit
from autograd_4bit import Autograd4bitQuantLinear, make_quant_for_4bit_autograd

try:
    from gptq_llama import quant
    from quant import make_quant
    is_triton = False
except ImportError:
    import quant
    is_triton = True

#Calculates max memory from arguments
def calculate_device_mem (model):
    if shared.args.gpu_memory or torch.cuda.device_count() > 1:
        if shared.args.gpu_memory:
            memory_map = list(map(lambda x: x.strip(), shared.args.gpu_memory))
            max_cpu_memory = shared.args.cpu_memory.strip() if shared.args.cpu_memory is not None else '99GiB'
            max_memory = {}
            for i in range(len(memory_map)):
                max_memory[i] = f'{memory_map[i]}GiB' if not re.match('.*ib$', memory_map[i].lower()) else memory_map[i]
            max_memory['cpu'] = f'{max_cpu_memory}GiB' if not re.match('.*ib$', max_cpu_memory.lower()) else max_cpu_memory
        else:
            max_memory = accelerate.utils.get_balanced_memory(model)
    return max_memory

#Autograd Loader to load the model with offloading or not.
def load_autograd (config_path, model_path):


    print(Style.BRIGHT + Fore.CYAN + "Autograd Loading Model ...")
    t0 = time.time()

    rope = {}
    if shared.args.compress_pos_emb > 1:
        rope['rope_scaling'] = {'type': 'linear', 'factor': shared.args.compress_pos_emb}
    elif shared.args.alpha_value > 1:
        rope['rope_scaling'] = {'type': 'dynamic', 'factor': shared.args.alpha_value}


    with accelerate.init_empty_weights():
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=shared.args.trust_remote_code, **rope)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=shared.args.trust_remote_code)
        model = model.eval()
        layers = find_layers(model)
        for name in ['lm_head', 'embed_out']:
            if name in layers:
                del layers[name]
        make_quant_for_4bit_autograd(model, layers, groupsize=shared.args.groupsize, is_v1_model=shared.args.v1)


    if shared.args.gpu_memory or torch.cuda.device_count() > 1:
        print(Style.BRIGHT + Fore.YELLOW + 'Autograd Dispatching model ...')
        # rotary_emb fix
        for n, m in model.named_modules():
            if 'rotary_emb' in n:
                cos_cached = m.cos_cached.clone().cpu()
                sin_cached = m.sin_cached.clone().cpu()
                break
        
        device_map = accelerate.infer_auto_device_map(model, max_memory=calculate_device_mem(model), no_split_module_classes=["LlamaDecoderLayer", "GPTJBlock", "OPTDecoderLayer", "GPTNeoXLayer"])
        print("Using the following device map for the quantized model:", device_map)      
        accelerate.load_checkpoint_in_model(model, checkpoint=model_path, device_map=device_map)
        model = accelerate.dispatch_model(model, device_map=device_map, offload_buffers=True, main_device=0)
        torch.cuda.empty_cache()
        print(Style.BRIGHT + Fore.YELLOW + 'Total {:.2f} Gib VRAM used.'.format(torch.cuda.memory_allocated() / 1024 / 1024))
        # rotary_emb fix
        for n, m in model.named_modules():
            if 'rotary_emb' in n:
                if getattr(m, '_hf_hook', None):
                    if isinstance(m._hf_hook, accelerate.hooks.SequentialHook):
                        hooks = m._hf_hook.hooks
                    else:
                        hooks = [m._hf_hook]
                    for hook in hooks:
                        if hook.offload:
                            if n + '.sin_cached' not in hook.weights_map.dataset.state_dict.keys():
                                hook.weights_map.dataset.state_dict[n + '.sin_cached'] = sin_cached.clone().cpu()
                                hook.weights_map.dataset.state_dict[n + '.cos_cached'] = cos_cached.clone().cpu()      
    else: 
        device_map="auto"
        print("Using the following device map for the quantized model:", device_map)
        model = accelerate.load_checkpoint_and_dispatch(
            model=model,
            checkpoint=model_path,
            device_map=device_map
        )
    print(Style.BRIGHT + Fore.GREEN + f"Loaded the model in {(time.time()-t0):.2f} seconds.")
    return model

#Autograd finalizer
def finalize_autograd (model):
  
    #model.half() #can't benchmark with lora
    for n, m in model.named_modules():
       if isinstance(m, Autograd4bitQuantLinear):
          if (shared.args.v1 == True):
              m.zeros = m.zeros.half()
          m.scales = m.scales.half()
          m.bias = m.bias.half()

    if (shared.args.fused_mlp) or (shared.args.quant_attn):
       if (shared.args.quant_attn): 
           from model_attn_mlp_patch import make_quant_attn
           make_quant_attn(model, is_v1_model=shared.args.v1)
           print(Style.BRIGHT + Fore.YELLOW + 'Autograd: quant_attn (May fail on newer cards with RoPE)')
       if (shared.args.fused_mlp):
           from model_attn_mlp_patch import make_fused_mlp
           make_fused_mlp(model, is_v1_model=shared.args.v1)
           print(Style.BRIGHT + Fore.YELLOW + 'Autograd: fused_mlp')
    else:
       from amp_wrapper import AMPWrapper
       wrapper = AMPWrapper(model)
       wrapper.apply_generate()

    print(Style.BRIGHT + Fore.RED + 'Finalizing Autograd Lora:', shared.lora_names)

# This function is a replacement for the load_quant function in the
# GPTQ-for_LLaMa repository. It supports more models and branches.
def _load_quant(model, checkpoint, wbits, groupsize=-1, faster_kernel=False, exclude_layers=None, kernel_switch_threshold=128, eval=True):
    exclude_layers = exclude_layers or ['lm_head', 'embed_out']

    def noop(*args, **kwargs):
        pass

    rope = {}
    if shared.args.compress_pos_emb > 1:
        rope['rope_scaling'] = {'type': 'linear', 'factor': shared.args.compress_pos_emb}
    elif shared.args.alpha_value > 1:
        rope['rope_scaling'] = {'type': 'dynamic', 'factor': shared.args.alpha_value}

    config = AutoConfig.from_pretrained(model, trust_remote_code=shared.args.trust_remote_code, **rope)
    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=shared.args.trust_remote_code)
    torch.set_default_dtype(torch.float)
    if eval:
        model = model.eval()

    layers = find_layers(model)
    for name in exclude_layers:
        if name in layers:
            del layers[name]

    if not is_triton:
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
        #print (make_quant_kwargs)
    else:
        quant.make_quant_linear(model, layers, wbits, groupsize)

    del layers

    logger.warning('Loading GPTQ model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint), strict=False)
    else:
        model.load_state_dict(torch.load(checkpoint), strict=False)

    if is_triton:
        if shared.args.quant_attn:
            quant.make_quant_attn(model)

        if eval and shared.args.fused_mlp:
            quant.make_fused_mlp(model)

        if shared.args.warmup_autotune:
            quant.autotune_warmup_linear(model, transpose=not eval)
            if eval and shared.args.fused_mlp:
                quant.autotune_warmup_fused(model)

    #model.seqlen = 2048
    
    return model

def offload_quant(model, checkpoint, wbits, groupsize, pre_layer):
    from gptq_llama import llama_inference_offload
    from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaConfig
    transformers.models.llama.modeling_llama.LlamaModel = llama_inference_offload.Offload_LlamaModel

    rope = {}
    if shared.args.compress_pos_emb > 1:
        rope['rope_scaling'] = {'type': 'linear', 'factor': shared.args.compress_pos_emb}
    elif shared.args.alpha_value > 1:
        rope['rope_scaling'] = {'type': 'dynamic', 'factor': shared.args.alpha_value}
    
    config = LlamaConfig.from_pretrained(model, trust_remote_code=shared.args.trust_remote_code, **rope )
    def noop(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = noop 
    torch.nn.init.uniform_ = noop 
    torch.nn.init.normal_ = noop 

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LlamaForCausalLM (config)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    make_quant(model, layers, wbits, groupsize)

    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint), strict=False)
    else:
        model.load_state_dict(torch.load(checkpoint), strict=False)
    
    
    if (isinstance(pre_layer, int)):
        pre_layer = [pre_layer]
    last_layer = 0
    for index, dev_layer in enumerate(pre_layer):
        target = torch.device(f'cuda:{index}')
        for i in range(last_layer, dev_layer):
            model.model.layers[i].to(target)
            model.model.layers[i].target_device = target
        last_layer = dev_layer
    for i in range(last_layer, len(model.model.layers)):
        model.model.layers[i].target_device = torch.device('cpu')
    model.model.embed_tokens.to(torch.device('cuda:0'))
    model.model.norm.to(torch.device('cuda:0'))
    model.lm_head.to(torch.device('cuda:0'))
    model.model.preload = last_layer
    print('Done.')
    return model



# Used to locate the .pt/.safetensors quantized file
def find_quantized_model_file(model_name):
    if shared.args.checkpoint:
        return Path(shared.args.checkpoint)

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
        for ext in ['.pt', '.safetensors']:
            found = list(path_to_model.glob(f"*{ext}"))
            if len(found) > 0:
                if len(found) > 1:
                    logger.warning(f'More than one {ext} model has been found. The last one will be selected. It could be wrong.')

                pt_path = found[-1]
                break

    return pt_path


# The function that loads the model in modules/models.py
def load_quantized(model_name):
    if shared.args.model_type is None:
        logger.error("The model could not be loaded because its type could not be inferred from its name.")
        logger.error("Please specify the type manually using the --model_type argument.")
        return None

    # Select the appropriate load_quant function
    model_type = shared.args.model_type.lower()
    if shared.args.pre_layer and model_type == 'llama':
        load_quant = offload_quant
    elif model_type in ('llama', 'opt', 'gptneox', 'gptj'):
        if shared.args.pre_layer:
            logger.warning("Ignoring --pre_layer because it only works for llama model type.")

        load_quant = _load_quant
    else:
        logger.error("Unknown pre-quantized model type specified. Only 'llama', 'opt', 'gptj, and gptneox' are supported")
        exit()

    # Find the quantized model weights file (.pt/.safetensors)
    path_to_model = Path(f'{shared.args.model_dir}/{model_name}')
    pt_path = find_quantized_model_file(model_name)
    if not pt_path:
        logger.error("Could not find the quantized model in .pt or .safetensors format, exiting...")
        exit()
    else:
        logger.info(f"Found the following quantized model: {pt_path}")

    # Autograd Model Load
    if shared.args.autograd:
       model = load_autograd ( str(path_to_model), str(pt_path)) 

       if not shared.args.lora or len(shared.lora_names) == 0:
          finalize_autograd(model)
          print(Style.BRIGHT + Fore.GREEN + 'No Lora. Finalized in loader...')   
       return model #let textgen handle the tokenizer

    # qwopqwop200's offload 
    elif model_type == 'llama' and shared.args.pre_layer:
        if len(shared.args.pre_layer) == 1:
            pre_layer = shared.args.pre_layer[0]
        else:
            pre_layer = shared.args.pre_layer

        model = load_quant(str(path_to_model), str(pt_path), shared.args.wbits, shared.args.groupsize, pre_layer)        

    else:
        threshold = False if model_type == ('gptj') else 128
        model = _load_quant(str(path_to_model), str(pt_path), shared.args.wbits, shared.args.groupsize, kernel_switch_threshold=threshold)

        # accelerate offload (doesn't work properly)
        if shared.args.gpu_memory or torch.cuda.device_count() > 1 or (is_xpu_available() and torch.xpu.device_count() > 1):
            device_map = accelerate.infer_auto_device_map(model, max_memory=calculate_device_mem(model), no_split_module_classes=["LlamaDecoderLayer", "GPTJBlock", "OPTDecoderLayer", "GPTNeoXLayer"])
            logger.info("Using the following device map for the quantized model:", device_map)
            # https://huggingface.co/docs/accelerate/package_reference/big_modeling#accelerate.dispatch_model
            model = accelerate.dispatch_model(model, device_map=device_map, offload_buffers=True)

        # No offload
        elif not shared.args.cpu:
            if is_xpu_available():
                model = model.to(torch.device("xpu:0"))
            else:
                model = model.to(torch.device('cuda:0'))

    
    return model
