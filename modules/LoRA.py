from pathlib import Path

import torch
import sys
import modules.shared as shared
from modules.logging_colors import logger

def autograd_add (lora_path):

    #Loras Do not Stack yet.

    sys.path.insert(0, str(Path("repositories/GPTQ-Merged/src/alpaca_lora_4bit")))
    from monkeypatch.peft_tuners_lora_monkey_patch import replace_peft_model_with_gptq_lora_model
    replace_peft_model_with_gptq_lora_model()
      
    from peft import PeftModel
    #Not sure what happens in offload    
    print('Autograd Add Lora', lora_path)
    shared.model = PeftModel.from_pretrained(shared.model, lora_path, device_map={'': 0}, torch_dtype=torch.float32)
    
    from modules.GPTQ_loader import finalize_autograd
    finalize_autograd(shared.model)
    print('Finalize Lora\nNote: Only one lora works with 4bit for the time being.\n Please only add one at a time and remove all loras before switching!')





def add_lora_to_model(lora_names):
    prior_set = set(shared.lora_names)
    added_set = set(lora_names) - prior_set
    removed_set = prior_set - set(lora_names)
    shared.lora_names = list(lora_names)
    
    if len(removed_set) > 0 and (shared.args.autograd or shared.args.autogptq):
       from modules.models import reload_model
       reload_model() #remove lora
       return
    if shared.args.autograd and len(lora_names) > 0:
       lora_path = Path(f"{shared.args.lora_dir}/{lora_names[0]}")
       autograd_add(lora_path)
       logger.info('Autograd Lora Added:', lora_path, Path(f"{shared.args.lora_dir}/{lora_names[0]}"))
       return

    from peft import PeftModel

    # AutoGPTQ lora for inference
    if shared.args.autogptq and len(lora_names) > 0:
        from auto_gptq import AutoGPTQForCausalLM, get_gptq_peft_model
        from auto_gptq.utils.peft_utils import GPTQLoraConfig
        
        peft_config = GPTQLoraConfig(
            inference_mode=True,
        )
        logger.info('AutoGPTQ Lora Added:', lora_path, Path(f"{shared.args.lora_dir}/{lora_names[0]}"))
        #logger.info("#####", peft_config)
        model = get_gptq_peft_model(shared.model, peft_config, Path(f"{shared.args.lora_dir}/{lora_names[0]}"))
        return

    # If no LoRA needs to be added or removed, exit
    if len(added_set) == 0 and len(removed_set) == 0:
        return

    # Add a LoRA when another LoRA is already present
    if len(removed_set) == 0 and len(prior_set) > 0:
        logger.info(f"Adding the LoRA(s) named {added_set} to the model...")
        for lora in added_set:
            shared.model.load_adapter(Path(f"{shared.args.lora_dir}/{lora}"), lora)

        return

    # If any LoRA needs to be removed, start over
    if len(removed_set) > 0:
        shared.model.disable_adapter()
    #    if shared.args.autograd:
    #       from modules.models import reload_model
    #       reload_model() #remove lora
    
        shared.model = shared.model.base_model.model


    if len(lora_names) > 0:
        logger.info("Applying the following LoRAs to {}: {}".format(shared.model_name, ', '.join(lora_names)))
        params = {}
        if not shared.args.cpu:
            params['dtype'] = shared.model.dtype
            if hasattr(shared.model, "hf_device_map"):
                params['device_map'] = {"base_model.model." + k: v for k, v in shared.model.hf_device_map.items()}
            elif shared.args.load_in_8bit:
                params['device_map'] = {'': 0}

#        if shared.args.autograd:
 #          lora_path = Path(f"{shared.args.lora_dir}/{lora_names[0]}")
 #          autograd_add(lora_path)
 #          print ('Lora Added:', lora_path, Path(f"{shared.args.lora_dir}/{lora_names[0]}"))
 #       else:
        shared.model = PeftModel.from_pretrained(shared.model, Path(f"{shared.args.lora_dir}/{lora_names[0]}"), **params)

        for lora in lora_names[1:]:
            shared.model.load_adapter(Path(f"{shared.args.lora_dir}/{lora}"), lora)
      
        if not shared.args.load_in_8bit and not shared.args.cpu:
            #if shared.args.autograd: 
            #   from modules.GPTQ_loader import finalize_autograd
            #   finalize_autograd(shared.model)
            #   print('Finalize Lora')
            #else:
            shared.model.half()
            if not hasattr(shared.model, "hf_device_map"):
                if torch.has_mps:
                    device = torch.device('mps')
                    shared.model = shared.model.to(device)
                else:
                    shared.model = shared.model.cuda()

