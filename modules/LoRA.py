from pathlib import Path

import torch
import sys
import modules.shared as shared

def autograd_add (lora_path):

    #not sure this will stack loras yet

    sys.path.insert(0, str(Path("repositories/GPTQ-Merged/src/alpaca_lora_4bit")))
    import autograd_4bit, quant
    from autograd_4bit import Autograd4bitQuantLinear
       
    sys.path.insert(0, 'repositories/peft/src')
    from peft import PeftModel
    
    print('Autograd Add Lora', lora_path)
    shared.model = PeftModel.from_pretrained(shared.model, lora_path, device_map={'': 0}, torch_dtype=torch.float32)
    
    from modules.GPTQ_loader import finalize_autograd
    finalize_autograd(shared.model)
    print('Finalize Lora')


def add_lora_to_model(lora_names):
    prior_set = set(shared.lora_names)
    added_set = set(lora_names) - prior_set
    removed_set = prior_set - set(lora_names)
    shared.lora_names = list(lora_names)
    
    if len(removed_set) > 0 and shared.args.autograd:
       from modules.models import reload_model
       reload_model() #remove lora
       return
    if shared.args.autograd and len(lora_names) > 0:
       lora_path = Path(f"{shared.args.lora_dir}/{lora_names[0]}")
       autograd_add(lora_path)
       print ('Lora Added:', lora_path, Path(f"{shared.args.lora_dir}/{lora_names[0]}"))
       return
    if not(shared.args.autograd):
       from peft import PeftModel

    # Nothing to do = skip.
    if len(added_set) == 0 and len(removed_set) == 0:
        return
      
       # Only adding, and already peft? Do it the easy way.
    if len(removed_set) == 0 and len(prior_set) > 0:
        print(f"Adding the LoRA(s) named {added_set} to the model...")
        for lora in added_set:
            shared.model.load_adapter(Path(f"{shared.args.lora_dir}/{lora}"), lora)
            return

       # If removing anything, disable all and re-add.
    if len(removed_set) > 0:
        shared.model.disable_adapter()

    if len(lora_names) > 0:
        print("Applying the following LoRAs to {}: {}".format(shared.model_name, ', '.join(lora_names)))
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
            #if not shared.args.monkey_patch: is this where my multi lora broke?
            shared.model.half()
            if not hasattr(shared.model, "hf_device_map"):
                if torch.has_mps:
                    device = torch.device('mps')
                    shared.model = shared.model.to(device)
                else:
                    shared.model = shared.model.cuda()

