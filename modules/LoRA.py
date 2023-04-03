from pathlib import Path

import torch

import modules.shared as shared
from modules.models import load_model
from modules.text_generation import clear_torch_cache


def reload_model():
    shared.lora_name = shared.args.lora = None
    shared.model = shared.tokenizer = None
    clear_torch_cache()
    shared.model, shared.tokenizer = load_model(shared.model_name)


def add_lora_to_model(lora_name):

    if not(shared.args.autograd):
       from peft import PeftModel

    # If a LoRA had been previously loaded, or if we want
    # to unload a LoRA, reload the model
    if shared.lora_name != "None" or lora_name == "None":
        reload_model()
    shared.lora_name = lora_name
    if shared.args.autograd and shared.lora_name != "None":
       import sys

       sys.path.insert(0, 'repositories/GPTQ-Merged')
       import autograd_4bit, quant
       from autograd_4bit import Autograd4bitQuantLinear
       
       sys.path.insert(0, 'repositories/peft-GPTQ/src')
       from peft import PeftModel
       from peft.tuners.lora import Linear4bitLt
       print('Loading', lora_name)
       shared.model = PeftModel.from_pretrained(shared.model, Path(f"loras/{lora_name}"), device_map={'': 0}, torch_dtype=torch.float32)
       
       for n, m in shared.model.named_modules():
           if isinstance(m, Autograd4bitQuantLinear) or isinstance(m, Linear4bitLt):
               m.zeros = m.zeros.half()
               m.scales = m.scales.half()
               m.bias = m.bias.half()
       autograd_4bit.use_new = True
       autograd_4bit.auto_switch = True
       print('Apply auto switch Lora 4-bit', lora_name)

    else:
      if lora_name != "None":
         print(f"Adding the LoRA {lora_name} to the model...")
         params = {}
         if not shared.args.cpu:
             params['dtype'] = shared.model.dtype
             if hasattr(shared.model, "hf_device_map"):
                 params['device_map'] = {"base_model.model."+k: v for k, v in shared.model.hf_device_map.items()}
             elif shared.args.load_in_8bit:
                 params['device_map'] = {'': 0}
            
         shared.model = PeftModel.from_pretrained(shared.model, Path(f"loras/{lora_name}"), **params)
         if not shared.args.load_in_8bit and not shared.args.cpu:
             shared.model.half()
             if not hasattr(shared.model, "hf_device_map"):
                 if torch.has_mps:
                     device = torch.device('mps')
                     shared.model = shared.model.to(device)
                 else:
                     shared.model = shared.model.cuda()
