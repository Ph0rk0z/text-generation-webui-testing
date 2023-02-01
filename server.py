import re
import gc
import time
import glob
import torch
import argparse
import json
import io
import base64
import sys
import os
from pathlib import Path
from PIL import Image
import copy
import gradio as gr
import warnings
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from modules.html_generator import *
from modules.ui import *
from modules.stopping_criteria import _SentinelTokenStoppingCriteria

transformers.logging.set_verbosity_error()

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Name of the model to load by default.')
parser.add_argument('--notebook', action='store_true', help='Launch the web UI in notebook mode, where the output is written to the same text box as the input.')
parser.add_argument('--chat', action='store_true', help='Launch the web UI in chat mode.')
parser.add_argument('--cai-chat', action='store_true', help='Launch the web UI in chat mode with a style similar to Character.AI\'s. If the file img_bot.png or img_bot.jpg exists in the same folder as server.py, this image will be used as the bot\'s profile picture. Similarly, img_me.png or img_me.jpg will be used as your profile picture.')
parser.add_argument('--cpu', action='store_true', help='Use the CPU to generate text.')
parser.add_argument('--load-in-8bit', action='store_true', help='Load the model with 8-bit precision.')
parser.add_argument('--auto-devices', action='store_true', help='Automatically split the model across the available GPU(s) and CPU.')
parser.add_argument('--disk', action='store_true', help='If the model is too large for your GPU(s) and CPU combined, send the remaining layers to the disk.')
parser.add_argument('--disk-cache-dir', type=str, help='Directory to save the disk cache to. Defaults to "cache/".')
parser.add_argument('--gpu-memory', type=int, help='Maximum GPU memory in GiB to allocate. This is useful if you get out of memory errors while trying to generate text. Must be an integer number.')
parser.add_argument('--cpu-memory', type=int, help='Maximum CPU memory in GiB to allocate for offloaded weights. Must be an integer number. Defaults to 99.')
parser.add_argument('--deepspeed', action='store_true', help='Enable the use of DeepSpeed ZeRO-3 for inference via the Transformers integration.')
parser.add_argument('--nvme-offload-dir', type=str, help='Directory to use for DeepSpeed ZeRO-3 NVME offloading.')
parser.add_argument('--local_rank', type=int, default=0, help='Optional argument for DeepSpeed distributed setups.')
parser.add_argument('--no-stream', action='store_true', help='Don\'t stream the text output in real time. This improves the text generation performance.')
parser.add_argument('--settings', type=str, help='Load the default interface settings from this json file. See settings-template.json for an example.')
parser.add_argument('--extensions', type=str, help='The list of extensions to load. If you want to load more than one extension, write the names separated by commas and between quotation marks, "like,this".')
parser.add_argument('--listen', action='store_true', help='Make the web UI reachable from your local network.')
parser.add_argument('--listen-port', type=int, help='The listening port that the server will use.')
parser.add_argument('--share', action='store_true', help='Create a public URL. This is useful for running the web UI on Google Colab or similar.')
parser.add_argument('--verbose', action='store_true', help='Print the prompts to the terminal.')
args = parser.parse_args()

if (args.chat or args.cai_chat) and not args.no_stream:
    print("Warning: chat mode currently becomes somewhat slower with text streaming on.\nConsider starting the web UI with the --no-stream option.\n")
    
settings = {
    'max_new_tokens': 200,
    'max_new_tokens_min': 1,
    'max_new_tokens_max': 2000,
    'preset': 'NovelAI-Sphinx Moth',
    'name1': 'Person 1',
    'name2': 'Person 2',
    'context': 'This is a conversation between two people.',
    'prompt': 'Common sense questions and answers\n\nQuestion: \nFactual answer:',
    'prompt_gpt4chan': '-----\n--- 865467536\nInput text\n--- 865467537\n',
    'stop_at_newline': True,
    'history_size': 0,
    'history_size_min': 0,
    'history_size_max': 64,
    'preset_pygmalion': 'Pygmalion',
    'name1_pygmalion': 'You',
    'name2_pygmalion': 'Kawaii',
    'context_pygmalion': "Kawaii's persona: Kawaii is a cheerful person who loves to make others smile. She is an optimist who loves to spread happiness and positivity wherever she goes.\n<START>",
    'stop_at_newline_pygmalion': False,
}

if args.settings is not None and Path(args.settings).exists():
    new_settings = json.loads(open(Path(args.settings), 'r').read())
    for item in new_settings:
        settings[item] = new_settings[item]


if args.deepspeed:
    import deepspeed
    from transformers.deepspeed import HfDeepSpeedConfig, is_deepspeed_zero3_enabled

    # Distributed setup
    if args.local_rank is not None:
        local_rank = args.local_rank
    else:
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()

    # DeepSpeed configration
    # https://huggingface.co/docs/transformers/main_classes/deepspeed
    train_batch_size = 1 * world_size
    if args.nvme_offload_dir:
        ds_config = {
            "fp16": {
                "enabled": True,
            },
            "bf16": {
                "enabled": False,
            },
            "zero_optimization": {
                "stage": 3,
                "offload_param": {
                    "device": "nvme",
                    "nvme_path": args.nvme_offload_dir,
                    "pin_memory": True,
                    "buffer_count": 5,
                    "buffer_size": 1e9,
                    "max_in_cpu": 1e9
                },
                "overlap_comm": True,
                "reduce_bucket_size": "auto",
                "contiguous_gradients": True,
                "sub_group_size": 1e8,
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_max_live_parameters": "auto",
                "stage3_max_reuse_distance": "auto",
            },
            "aio": {
                "block_size": 262144,
                "queue_depth": 32,
                "thread_count": 1,
                "single_submit": False,
                "overlap_events": True
            },
            "steps_per_print": 2000,
            "train_batch_size": train_batch_size,
            "train_micro_batch_size_per_gpu": 1,
            "wall_clock_breakdown": False
        }
    else:
        ds_config = {
            "fp16": {
                "enabled": True,
            },
            "bf16": {
                "enabled": False,
            },
            "zero_optimization": {
                "stage": 3,
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": "auto",
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_max_live_parameters": "auto",
                "stage3_max_reuse_distance": "auto",
            },
            "steps_per_print": 2000,
            "train_batch_size": train_batch_size,
            "train_micro_batch_size_per_gpu": 1,
            "wall_clock_breakdown": False
        }
    dschf = HfDeepSpeedConfig(ds_config) # Keep this object alive for the Transformers integration


def load_model(model_name):
    print(f"Loading {model_name}...")
    t0 = time.time()

    # Default settings
    if not (args.cpu or args.load_in_8bit or args.auto_devices or args.disk or args.gpu_memory is not None or args.cpu_memory is not None or args.deepspeed):
        if Path(f"torch-dumps/{model_name}.pt").exists():
            print("Loading in .pt format...")
            model = torch.load(Path(f"torch-dumps/{model_name}.pt"))
        elif model_name.lower().startswith(('gpt-neo', 'opt-', 'galactica')) and any(size in model_name.lower() for size in ('13b', '20b', '30b')):
            model = AutoModelForCausalLM.from_pretrained(Path(f"models/{model_name}"), device_map='auto', load_in_8bit=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(Path(f"models/{model_name}"), low_cpu_mem_usage=True, torch_dtype=torch.float16).cuda()

    # DeepSpeed ZeRO-3
    elif args.deepspeed:
        model = AutoModelForCausalLM.from_pretrained(Path(f"models/{model_name}", no_split_module_classes=["GPTJBlock"]))
        model = deepspeed.initialize(model=model,
                                     config_params=ds_config,
                                     model_parameters=None,
                                     optimizer=None,
                                     lr_scheduler=None)[0]
        model.module.eval() # Inference
        print(f"DeepSpeed ZeRO-3 is enabled: {is_deepspeed_zero3_enabled()}")

    # Custom
    else:
        command = "AutoModelForCausalLM.from_pretrained"
        settings = []

        if args.cpu:
            settings.append("low_cpu_mem_usage=True")
            settings.append("torch_dtype=torch.float32")
        else:
            settings.append("device_map='auto'")
            settings.append("load_in_8bit=True" if args.load_in_8bit else "torch_dtype=torch.float16")

            if args.gpu_memory:
                settings.append(f"max_memory={{0: '{args.gpu_memory or '99'}GiB', 'cpu': '{args.cpu_memory or '99'}GiB'}}")
            elif not args.load_in_8bit:
                total_mem = (torch.cuda.get_device_properties(0).total_memory/(1024*1024))
                suggestion = round((total_mem-1000)/1000)*1000
                if total_mem-suggestion < 800:
                    suggestion -= 1000
                suggestion = int(round(suggestion/1000))
                print(f"\033[1;32;1mAuto-assiging --gpu-memory {suggestion} for your GPU to try to prevent out-of-memory errors.\nYou can manually set other values.\033[0;37;0m")
                settings.append(f"max_memory={{0: '{suggestion}GiB', 'cpu': '{args.cpu_memory or '99'}GiB'}}")
            if args.disk:
                settings.append(f"offload_folder='{args.disk_cache_dir or 'cache'}'")

        settings = ', '.join(set(settings))
        command = f"{command}(Path(f'models/{model_name}'), {settings})"
        model = eval(command)

    # Loading the tokenizer
    if model_name.lower().startswith(('gpt4chan', 'gpt-4chan', '4chan')) and Path(f"models/gpt-j-6B/").exists():
        tokenizer = AutoTokenizer.from_pretrained(Path("models/gpt-j-6B/"))
    else:
        tokenizer = AutoTokenizer.from_pretrained(Path(f"models/{model_name}/"))
    tokenizer.truncation_side = 'left'

    print(f"Loaded the model in {(time.time()-t0):.2f} seconds.")
    return model, tokenizer

# Removes empty replies from gpt4chan outputs
def fix_gpt4chan(s):
    for i in range(10):
        s = re.sub("--- [0-9]*\n>>[0-9]*\n---", "---", s)
        s = re.sub("--- [0-9]*\n *\n---", "---", s)
        s = re.sub("--- [0-9]*\n\n\n---", "---", s)
    return s

# Fix the LaTeX equations in galactica
def fix_galactica(s):
    s = s.replace(r'\[', r'$')
    s = s.replace(r'\]', r'$')
    s = s.replace(r'\(', r'$')
    s = s.replace(r'\)', r'$')
    s = s.replace(r'$$', r'$')
    return s

def encode(prompt, tokens_to_generate=0, add_special_tokens=True):
    if args.cpu:
        input_ids = tokenizer.encode(str(prompt), return_tensors='pt', truncation=True, max_length=2048-tokens_to_generate, add_special_tokens=add_special_tokens)
    else:
        torch.cuda.empty_cache()
        input_ids = tokenizer.encode(str(prompt), return_tensors='pt', truncation=True, max_length=2048-tokens_to_generate, add_special_tokens=add_special_tokens).cuda()
    return input_ids

def decode(output_ids):
    reply = tokenizer.decode(output_ids, skip_special_tokens=True)
    reply = reply.replace(r'<|endoftext|>', '')
    return reply

def formatted_outputs(reply, model_name):
    if not (args.chat or args.cai_chat):
        if model_name.lower().startswith('galactica'):
            reply = fix_galactica(reply)
            return reply, reply, generate_basic_html(reply)
        elif model_name.lower().startswith(('gpt4chan', 'gpt-4chan', '4chan')):
            reply = fix_gpt4chan(reply)
            return reply, 'Only applicable for GALACTICA models.', generate_4chan_html(reply)
        else:
            return reply, 'Only applicable for GALACTICA models.', generate_basic_html(reply)
    else:
        return reply

def generate_reply(question, tokens, inference_settings, selected_model, eos_token=None, stopping_string=None):
    global model, tokenizer, model_name, loaded_preset, preset

    original_question = question
    if not (args.chat or args.cai_chat):
        question = apply_extensions(question, "input")
    if args.verbose:
        print(f"\n\n{question}\n--------------------\n")

    if selected_model != model_name:
        model_name = selected_model
        model = tokenizer = None
        if not args.cpu:
            gc.collect()
            torch.cuda.empty_cache()
        model, tokenizer = load_model(model_name)
    if inference_settings != loaded_preset:
        with open(Path(f'presets/{inference_settings}.txt'), 'r') as infile:
            preset = infile.read()
        loaded_preset = inference_settings

    cuda = "" if args.cpu else ".cuda()"
    n = tokenizer.eos_token_id if eos_token is None else tokenizer.encode(eos_token, return_tensors='pt')[0][-1]
    if args.deepspeed:
        input_ids = encode(question, tokens).to(device=local_rank)
    else:
        input_ids = encode(question, tokens)
    if stopping_string is not None:
        # The stopping_criteria code below was copied from
        # https://github.com/PygmalionAI/gradio-ui/blob/master/src/model.py
        t = encode(stopping_string, 0, add_special_tokens=False)
        stopping_criteria_list = transformers.StoppingCriteriaList([
            _SentinelTokenStoppingCriteria(
                sentinel_token_ids=t,
                starting_idx=len(input_ids[0])
            )
        ])
    else:
        stopping_criteria_list = None

    # Generate the entire reply at once
    if args.no_stream:
        t0 = time.time()
        if args.deepspeed:
            with torch.no_grad():
                output = eval(f"model.generate(input_ids, eos_token_id={n}, stopping_criteria=stopping_criteria_list, {preset})")
        else:
            output = eval(f"model.generate(input_ids, eos_token_id={n}, stopping_criteria=stopping_criteria_list, {preset}){cuda}")
        reply = decode(output[0])
        t1 = time.time()
        print(f"Output generated in {(t1-t0):.2f} seconds ({(len(output[0])-len(input_ids[0]))/(t1-t0):.2f} it/s)")
        if not (args.chat or args.cai_chat):
            reply = original_question + apply_extensions(reply[len(question):], "output")
        yield formatted_outputs(reply, model_name)

    # Generate the reply 1 token at a time
    else:
        yield formatted_outputs(original_question, model_name)
        preset = preset.replace('max_new_tokens=tokens', 'max_new_tokens=8')
        for i in tqdm(range(tokens//8+1)):
            if args.deepspeed:
                with torch.no_grad():
                    output = eval(f"model.generate(input_ids, eos_token_id={n}, stopping_criteria=stopping_criteria_list, {preset})")
            else:
                output = eval(f"model.generate(input_ids, eos_token_id={n}, stopping_criteria=stopping_criteria_list, {preset}){cuda}")
            reply = decode(output[0])
            if not (args.chat or args.cai_chat):
                reply = original_question + apply_extensions(reply[len(question):], "output")
            yield formatted_outputs(reply, model_name)
            input_ids = output
            if output[0][-1] == n:
                break

def apply_extensions(text, typ):
    global available_extensions, extension_state
    for ext in sorted(extension_state, key=lambda x : extension_state[x][1]):
        if extension_state[ext][0] == True:
            ext_string = f"extensions.{ext}.script"
            if typ == "input" and hasattr(eval(ext_string), "input_modifier"):
                text = eval(f"{ext_string}.input_modifier(text)")
            elif typ == "output" and hasattr(eval(ext_string), "output_modifier"):
                text = eval(f"{ext_string}.output_modifier(text)")
            elif typ == "bot_prefix" and hasattr(eval(ext_string), "bot_prefix_modifier"):
                text = eval(f"{ext_string}.bot_prefix_modifier(text)")
    return text

def update_extensions_parameters(*kwargs):
    i = 0
    for ext in sorted(extension_state, key=lambda x : extension_state[x][1]):
        if extension_state[ext][0] == True:
            params = eval(f"extensions.{ext}.script.params")
            for param in params:
                if len(kwargs) >= i+1:
                    params[param] = eval(f"kwargs[{i}]")
                    i += 1

def create_extensions_block():
    extensions_ui_elements = []
    default_values = []
    gr.Markdown('## Extensions parameters')
    for ext in sorted(extension_state, key=lambda x : extension_state[x][1]):
        if extension_state[ext][0] == True:
            params = eval(f"extensions.{ext}.script.params")
            for param in params:
                _id = f"{ext}-{param}"
                default_value = settings[_id] if _id in settings else params[param]
                default_values.append(default_value)
                if type(params[param]) == str:
                    extensions_ui_elements.append(gr.Textbox(value=default_value, label=f"{ext}-{param}"))
                elif type(params[param]) in [int, float]:
                    extensions_ui_elements.append(gr.Number(value=default_value, label=f"{ext}-{param}"))
                elif type(params[param]) == bool:
                    extensions_ui_elements.append(gr.Checkbox(value=default_value, label=f"{ext}-{param}"))

    update_extensions_parameters(*default_values)
    btn_extensions = gr.Button("Apply")
    btn_extensions.click(update_extensions_parameters, [*extensions_ui_elements], [])

def get_available_models():
    return sorted(set([item.replace('.pt', '') for item in map(lambda x : str(x.name), list(Path('models/').glob('*'))+list(Path('torch-dumps/').glob('*'))) if not item.endswith('.txt')]), key=str.lower)

def get_available_presets():
    return sorted(set(map(lambda x : '.'.join(str(x.name).split('.')[:-1]), Path('presets').glob('*.txt'))), key=str.lower)

def get_available_characters():
    return ["None"] + sorted(set(map(lambda x : '.'.join(str(x.name).split('.')[:-1]), Path('characters').glob('*.json'))), key=str.lower)

def get_available_extensions():
    return sorted(set(map(lambda x : x.parts[1], Path('extensions').glob('*/script.py'))), key=str.lower)

available_models = get_available_models()
available_presets = get_available_presets()
available_characters = get_available_characters()
available_extensions = get_available_extensions()
extension_state = {}
if args.extensions is not None:
    for i,ext in enumerate(args.extensions.split(',')):
        if ext in available_extensions:
            print(f'Loading the extension "{ext}"... ', end='')
            ext_string = f"extensions.{ext}.script"
            exec(f"import {ext_string}")
            extension_state[ext] = [True, i]
            print(f'Ok.')

# Choosing the default model
if args.model is not None:
    model_name = args.model
else:
    if len(available_models) == 0:
        print("No models are available! Please download at least one.")
        sys.exit(0)
    elif len(available_models) == 1:
        i = 0
    else:
        print("The following models are available:\n")
        for i,model in enumerate(available_models):
            print(f"{i+1}. {model}")
        print(f"\nWhich one do you want to load? 1-{len(available_models)}\n")
        i = int(input())-1
        print()
    model_name = available_models[i]
model, tokenizer = load_model(model_name)
loaded_preset = None

# UI settings
default_text = settings['prompt_gpt4chan'] if model_name.lower().startswith(('gpt4chan', 'gpt-4chan', '4chan')) else settings['prompt']
description = f"\n\n# Text generation lab\nGenerate text using Large Language Models.\n"
css = ".my-4 {margin-top: 0} .py-6 {padding-top: 2.5rem} #refresh-button {flex: none; margin: 0; padding: 0; min-width: 50px; border: none; box-shadow: none; border-radius: 0} #download-label, #upload-label {min-height: 0}"
buttons = {}
gen_events = []

if args.chat or args.cai_chat:
    history = {'internal': [], 'visible': []}
    character = None

    # This gets the new line characters right.
    def clean_chat_message(text):
        text = text.replace('\n', '\n\n')
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        return text

    def generate_chat_prompt(text, tokens, name1, name2, context, history_size, impersonate=False):
        text = clean_chat_message(text)

        rows = [f"{context.strip()}\n"]
        i = len(history['internal'])-1
        count = 0
        while i >= 0 and len(encode(''.join(rows), tokens)[0]) < 2048-tokens:
            rows.insert(1, f"{name2}: {history['internal'][i][1].strip()}\n")
            count += 1
            if not (history['internal'][i][0] == '<|BEGIN-VISIBLE-CHAT|>'):
                rows.insert(1, f"{name1}: {history['internal'][i][0].strip()}\n")
                count += 1
            i -= 1
            if history_size != 0 and count >= history_size:
                break

        if not impersonate:
            rows.append(f"{name1}: {text}\n")
            rows.append(apply_extensions(f"{name2}:", "bot_prefix"))
            limit = 3
        else:
            rows.append(f"{name1}:")
            limit = 2

        while len(rows) > limit and len(encode(''.join(rows), tokens)[0]) >= 2048-tokens:
            rows.pop(1)
            rows.pop(1)

        question = ''.join(rows)
        return question

    def extract_message_from_reply(question, reply, current, other, check, extensions=False):
        next_character_found = False
        substring_found = False

        previous_idx = [m.start() for m in re.finditer(f"(^|\n){current}:", question)]
        idx = [m.start() for m in re.finditer(f"(^|\n){current}:", reply)]
        idx = idx[len(previous_idx)-1]

        if extensions:
            reply = reply[idx + 1 + len(apply_extensions(f"{current}:", "bot_prefix")):]
        else:
            reply = reply[idx + 1 + len(f"{current}:"):]

        if check:
            reply = reply.split('\n')[0].strip()
        else:
            idx = reply.find(f"\n{other}:")
            if idx != -1:
                reply = reply[:idx]
                next_character_found = True
            reply = clean_chat_message(reply)

            # Detect if something like "\nYo" is generated just before
            # "\nYou:" is completed
            tmp = f"\n{other}:"
            for j in range(1, len(tmp)):
                if reply[-j:] == tmp[:j]:
                    substring_found = True

        return reply, next_character_found, substring_found

    def chatbot_wrapper(text, tokens, inference_settings, selected_model, name1, name2, context, check, history_size):
        original_text = text
        text = apply_extensions(text, "input")
        question = generate_chat_prompt(text, tokens, name1, name2, context, history_size)
        history['internal'].append(['', ''])
        history['visible'].append(['', ''])
        eos_token = '\n' if check else None
        for reply in generate_reply(question, tokens, inference_settings, selected_model, eos_token=eos_token, stopping_string=f"\n{name1}:"):
            reply, next_character_found, substring_found = extract_message_from_reply(question, reply, name2, name1, check, extensions=True)
            history['internal'][-1] = [text, reply]
            history['visible'][-1] = [original_text, apply_extensions(reply, "output")]
            if not substring_found:
                yield history['visible']
            if next_character_found:
                break
        yield history['visible']

    def impersonate_wrapper(text, tokens, inference_settings, selected_model, name1, name2, context, check, history_size):
        question = generate_chat_prompt(text, tokens, name1, name2, context, history_size, impersonate=True)
        eos_token = '\n' if check else None
        for reply in generate_reply(question, tokens, inference_settings, selected_model, eos_token=eos_token, stopping_string=f"\n{name2}:"):
            reply, next_character_found, substring_found = extract_message_from_reply(question, reply, name1, name2, check, extensions=False)
            if not substring_found:
                yield apply_extensions(reply, "output")
            if next_character_found:
                break
        yield apply_extensions(reply, "output")

    def cai_chatbot_wrapper(text, tokens, inference_settings, selected_model, name1, name2, context, check, history_size):
        for _history in chatbot_wrapper(text, tokens, inference_settings, selected_model, name1, name2, context, check, history_size):
            yield generate_chat_html(_history, name1, name2, character)

    def regenerate_wrapper(text, tokens, inference_settings, selected_model, name1, name2, context, check, history_size):
        last = history['visible'].pop()
        history['internal'].pop()
        text = last[0]
        if args.cai_chat:
            for i in cai_chatbot_wrapper(text, tokens, inference_settings, selected_model, name1, name2, context, check, history_size):
                yield i
        else:
            for i in chatbot_wrapper(text, tokens, inference_settings, selected_model, name1, name2, context, check, history_size):
                yield i

    def remove_last_message(name1, name2):
        if not history['internal'][-1][0] == '<|BEGIN-VISIBLE-CHAT|>':
            last = history['visible'].pop()
            history['internal'].pop()
        else:
            last = ['', '']
        if args.cai_chat:
            return generate_chat_html(history['visible'], name1, name2, character), last[0]
        else:
            return history['visible'], last[0]

    def send_last_reply_to_input():
        if len(history['visible']) > 0:
            return history['visible'][-1][1]
        else:
            return ''

    def replace_last_reply(text, name1, name2):
        if len(history['visible']) > 0:
            history['visible'][-1][1] = text
            history['internal'][-1][1] = apply_extensions(text, "input")

        if args.cai_chat:
            return generate_chat_html(history['visible'], name1, name2, character)
        else:
            return history['visible']

    def clear_html():
        return generate_chat_html([], "", "", character)

    def clear_chat_log(_character, name1, name2):
        global history
        if _character != 'None':
            for i in range(len(history['internal'])):
                if '<|BEGIN-VISIBLE-CHAT|>' in history['internal'][i][0]:
                    history['visible'] = [['', history['internal'][i][1]]]
                    history['internal'] = history['internal'][:i+1]
                    break
        else:
            history['internal'] = []
            history['visible'] = []
        if args.cai_chat:
            return generate_chat_html(history['visible'], name1, name2, character)
        else:
            return history['visible'] 

    def redraw_html(name1, name2):
        global history
        return generate_chat_html(history['visible'], name1, name2, character)

    def tokenize_dialogue(dialogue, name1, name2):
        _history = []

        dialogue = re.sub('<START>', '', dialogue)
        dialogue = re.sub('<start>', '', dialogue)
        dialogue = re.sub('(\n|^)[Aa]non:', '\\1You:', dialogue)
        dialogue = re.sub('(\n|^)\[CHARACTER\]:', f'\\1{name2}:', dialogue)
        idx = [m.start() for m in re.finditer(f"(^|\n)({name1}|{name2}):", dialogue)]
        if len(idx) == 0:
            return _history

        messages = []
        for i in range(len(idx)-1):
            messages.append(dialogue[idx[i]:idx[i+1]].strip())
        messages.append(dialogue[idx[-1]:].strip())

        entry = ['', '']
        for i in messages:
            if i.startswith(f'{name1}:'):
                entry[0] = i[len(f'{name1}:'):].strip()
            elif i.startswith(f'{name2}:'):
                entry[1] = i[len(f'{name2}:'):].strip()
                if not (len(entry[0]) == 0 and len(entry[1]) == 0):
                    _history.append(entry)
                entry = ['', '']

        print(f"\033[1;32;1m\nDialogue tokenized to:\033[0;37;0m\n", end='')
        for row in _history:
            for column in row:
                print("\n")
                for line in column.strip().split('\n'):
                    print("|  "+line+"\n")
                print("|\n")
            print("------------------------------")

        return _history

    def save_history():
        if not Path('logs').exists():
            Path('logs').mkdir()
        with open(Path('logs/conversation.json'), 'w') as f:
            f.write(json.dumps({'data': history['internal'], 'data_visible': history['visible']}))
        return Path('logs/conversation.json')

    def upload_history(file, name1, name2):
        global history
        file = file.decode('utf-8')
        try:
            j = json.loads(file)
            if 'data' in j:
                history['internal'] = j['data']
                if 'data_visible' in j:
                    history['visible'] = j['data_visible']
                else:
                    history['visible'] = copy.deepcopy(history['internal'])
            # Compatibility with Pygmalion AI's official web UI
            elif 'chat' in j:
                history['internal'] = [':'.join(x.split(':')[1:]).strip() for x in j['chat']]
                if len(j['chat']) > 0 and j['chat'][0].startswith(f'{name2}:'):
                    history['internal'] = [['<|BEGIN-VISIBLE-CHAT|>', history['internal'][0]]] + [[history['internal'][i], history['internal'][i+1]] for i in range(1, len(history['internal'])-1, 2)]
                else:
                    history['internal'] = [[history['internal'][i], history['internal'][i+1]] for i in range(0, len(history['internal'])-1, 2)]
        except:
            history['internal'] = tokenize_dialogue(file, name1, name2)
            history['visible'] = copy.deepcopy(history['internal'])

    def load_character(_character, name1, name2):
        global history, character
        context = ""
        history['internal'] = []
        history['visible'] = []
        if _character != 'None':
            character = _character
            data = json.loads(open(Path(f'characters/{_character}.json'), 'r').read())
            name2 = data['char_name']
            if 'char_persona' in data and data['char_persona'] != '':
                context += f"{data['char_name']}'s Persona: {data['char_persona']}\n"
            if 'world_scenario' in data and data['world_scenario'] != '':
                context += f"Scenario: {data['world_scenario']}\n"
            context = f"{context.strip()}\n<START>\n"
            if 'example_dialogue' in data and data['example_dialogue'] != '':
                history['internal'] = tokenize_dialogue(data['example_dialogue'], name1, name2)
            if 'char_greeting' in data and len(data['char_greeting'].strip()) > 0:
                history['internal'] += [['<|BEGIN-VISIBLE-CHAT|>', data['char_greeting']]]
                history['visible'] += [['', apply_extensions(data['char_greeting'], "output")]]
            else:
                history['internal'] += [['<|BEGIN-VISIBLE-CHAT|>', "Hello there!"]]
                history['visible'] += [['', "Hello there!"]]
        else:
            character = None
            context = settings['context_pygmalion']
            name2 = settings['name2_pygmalion']

        if args.cai_chat:
            return name2, context, generate_chat_html(history['visible'], name1, name2, character)
        else:
            return name2, context, history['visible']

    def upload_character(json_file, img, tavern=False):
        json_file = json_file if type(json_file) == str else json_file.decode('utf-8')
        data = json.loads(json_file)
        outfile_name = data["char_name"]
        i = 1
        while Path(f'characters/{outfile_name}.json').exists():
            outfile_name = f'{data["char_name"]}_{i:03d}'
            i += 1
        if tavern:
            outfile_name = f'TavernAI-{outfile_name}'
        with open(Path(f'characters/{outfile_name}.json'), 'w') as f:
            f.write(json_file)
        if img is not None:
            img = Image.open(io.BytesIO(img))
            img.save(Path(f'characters/{outfile_name}.png'))
        print(f'New character saved to "characters/{outfile_name}.json".')
        return outfile_name

    def upload_tavern_character(img, name1, name2):
        _img = Image.open(io.BytesIO(img))
        _img.getexif()
        decoded_string = base64.b64decode(_img.info['chara'])
        _json = json.loads(decoded_string)
        _json = {"char_name": _json['name'], "char_persona": _json['description'], "char_greeting": _json["first_mes"], "example_dialogue": _json['mes_example'], "world_scenario": _json['scenario']}
        _json['example_dialogue'] = _json['example_dialogue'].replace('{{user}}', name1).replace('{{char}}', _json['char_name'])
        return upload_character(json.dumps(_json), img, tavern=True)

    def upload_your_profile_picture(img):
        img = Image.open(io.BytesIO(img))
        img.save(Path(f'img_me.png'))
        print(f'Profile picture saved to "img_me.png"')

    suffix = '_pygmalion' if 'pygmalion' in model_name.lower() else ''
    with gr.Blocks(css=css+".h-\[40vh\] {height: 66.67vh} .gradio-container {max-width: 800px; margin-left: auto; margin-right: auto}", analytics_enabled=False) as interface:
        if args.cai_chat:
            display = gr.HTML(value=generate_chat_html([], "", "", character))
        else:
            display = gr.Chatbot()
        textbox = gr.Textbox(label='Input')
        buttons["Generate"] = gr.Button("Generate")
        with gr.Row():
            buttons["Stop"] = gr.Button("Stop")
            buttons["Regenerate"] = gr.Button("Regenerate")
            buttons["Remove last"] = gr.Button("Remove last")
            buttons["Clear"] = gr.Button("Clear history")
        with gr.Row():
            buttons["Send last reply to input"] = gr.Button("Send last reply to input")
            buttons["Replace last reply"] = gr.Button("Replace last reply")
            buttons["Impersonate"] = gr.Button("Impersonate")

        with gr.Row():
            with gr.Column():
                length_slider = gr.Slider(minimum=settings['max_new_tokens_min'], maximum=settings['max_new_tokens_max'], step=1, label='max_new_tokens', value=settings['max_new_tokens'])
                with gr.Row():
                    model_menu = gr.Dropdown(choices=available_models, value=model_name, label='Model')
                    create_refresh_button(model_menu, lambda : None, lambda : {"choices": get_available_models()}, "refresh-button")
            with gr.Column():
                history_size_slider = gr.Slider(minimum=settings['history_size_min'], maximum=settings['history_size_max'], step=1, label='Chat history size in prompt (0 for no limit)', value=settings['history_size'])
                with gr.Row():
                    preset_menu = gr.Dropdown(choices=available_presets, value=settings[f'preset{suffix}'], label='Generation parameters preset')
                    create_refresh_button(preset_menu, lambda : None, lambda : {"choices": get_available_presets()}, "refresh-button")

        name1 = gr.Textbox(value=settings[f'name1{suffix}'], lines=1, label='Your name')
        name2 = gr.Textbox(value=settings[f'name2{suffix}'], lines=1, label='Bot\'s name')
        context = gr.Textbox(value=settings[f'context{suffix}'], lines=2, label='Context')
        with gr.Row():
            character_menu = gr.Dropdown(choices=available_characters, value="None", label='Character')
            create_refresh_button(character_menu, lambda : None, lambda : {"choices": get_available_characters()}, "refresh-button")

        with gr.Row():
            check = gr.Checkbox(value=settings[f'stop_at_newline{suffix}'], label='Stop generating at new line character?')
        with gr.Row():
            with gr.Tab('Chat history'):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown('Upload')
                        upload = gr.File(type='binary')
                    with gr.Column():
                        gr.Markdown('Download')
                        download = gr.File()
                        buttons["Download"] = gr.Button(value="Click me")
            with gr.Tab('Upload character'):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown('1. Select the JSON file')
                        upload_char = gr.File(type='binary')
                    with gr.Column():
                        gr.Markdown('2. Select your character\'s profile picture (optional)')
                        upload_img = gr.File(type='binary')
                buttons["Upload character"] = gr.Button(value="Submit")
            with gr.Tab('Upload your profile picture'):
                upload_img_me = gr.File(type='binary')
            with gr.Tab('Upload TavernAI Character Card'):
                upload_img_tavern = gr.File(type='binary')

        if args.extensions is not None:
            create_extensions_block()

        input_params = [textbox, length_slider, preset_menu, model_menu, name1, name2, context, check, history_size_slider]
        if args.cai_chat:
            gen_events.append(buttons["Generate"].click(cai_chatbot_wrapper, input_params, display, show_progress=args.no_stream, api_name="textgen"))
            gen_events.append(textbox.submit(cai_chatbot_wrapper, input_params, display, show_progress=args.no_stream))
        else:
            gen_events.append(buttons["Generate"].click(chatbot_wrapper, input_params, display, show_progress=args.no_stream, api_name="textgen"))
            gen_events.append(textbox.submit(chatbot_wrapper, input_params, display, show_progress=args.no_stream))
        gen_events.append(buttons["Regenerate"].click(regenerate_wrapper, input_params, display, show_progress=args.no_stream))
        gen_events.append(buttons["Impersonate"].click(impersonate_wrapper, input_params, textbox, show_progress=args.no_stream))

        buttons["Send last reply to input"].click(send_last_reply_to_input, [], textbox, show_progress=args.no_stream)
        buttons["Replace last reply"].click(replace_last_reply, [textbox, name1, name2], display, show_progress=args.no_stream)
        buttons["Clear"].click(clear_chat_log, [character_menu, name1, name2], display)
        buttons["Remove last"].click(remove_last_message, [name1, name2], [display, textbox], show_progress=False)
        buttons["Stop"].click(None, None, None, cancels=gen_events)
        buttons["Download"].click(save_history, inputs=[], outputs=[download])
        buttons["Upload character"].click(upload_character, [upload_char, upload_img], [character_menu])
        for i in ["Generate", "Regenerate", "Replace last reply"]:
            buttons[i].click(lambda x: "", textbox, textbox, show_progress=False)
        textbox.submit(lambda x: "", textbox, textbox, show_progress=False)
        character_menu.change(load_character, [character_menu, name1, name2], [name2, context, display])
        upload_img_tavern.upload(upload_tavern_character, [upload_img_tavern, name1, name2], [character_menu])
        upload.upload(upload_history, [upload, name1, name2], [])
        upload_img_me.upload(upload_your_profile_picture, [upload_img_me], [])

        if args.cai_chat:
            upload.upload(redraw_html, [name1, name2], [display])
            upload_img_me.upload(redraw_html, [name1, name2], [display])
        else:
            upload.upload(lambda : history['visible'], [], [display])
            upload_img_me.upload(lambda : history['visible'], [], [display])

elif args.notebook:
    with gr.Blocks(css=css, analytics_enabled=False) as interface:
        gr.Markdown(description)
        with gr.Tab('Raw'):
            textbox = gr.Textbox(value=default_text, lines=23)
        with gr.Tab('Markdown'):
            markdown = gr.Markdown()
        with gr.Tab('HTML'):
            html = gr.HTML()
        buttons["Generate"] = gr.Button("Generate")
        buttons["Stop"] = gr.Button("Stop")

        length_slider = gr.Slider(minimum=settings['max_new_tokens_min'], maximum=settings['max_new_tokens_max'], step=1, label='max_new_tokens', value=settings['max_new_tokens'])
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    model_menu = gr.Dropdown(choices=available_models, value=model_name, label='Model')
                    create_refresh_button(model_menu, lambda : None, lambda : {"choices": get_available_models()}, "refresh-button")
            with gr.Column():
                with gr.Row():
                    preset_menu = gr.Dropdown(choices=available_presets, value=settings['preset'], label='Generation parameters preset')
                    create_refresh_button(preset_menu, lambda : None, lambda : {"choices": get_available_presets()}, "refresh-button")

        if args.extensions is not None:
            create_extensions_block()

        gen_events.append(buttons["Generate"].click(generate_reply, [textbox, length_slider, preset_menu, model_menu], [textbox, markdown, html], show_progress=args.no_stream, api_name="textgen"))
        gen_events.append(textbox.submit(generate_reply, [textbox, length_slider, preset_menu, model_menu], [textbox, markdown, html], show_progress=args.no_stream))
        buttons["Stop"].click(None, None, None, cancels=gen_events)

else:
    with gr.Blocks(css=css, analytics_enabled=False) as interface:
        gr.Markdown(description)
        with gr.Row():
            with gr.Column():
                textbox = gr.Textbox(value=default_text, lines=15, label='Input')
                length_slider = gr.Slider(minimum=settings['max_new_tokens_min'], maximum=settings['max_new_tokens_max'], step=1, label='max_new_tokens', value=settings['max_new_tokens'])
                with gr.Row():
                    preset_menu = gr.Dropdown(choices=available_presets, value=settings['preset'], label='Generation parameters preset')
                    create_refresh_button(preset_menu, lambda : None, lambda : {"choices": get_available_presets()}, "refresh-button")
                with gr.Row():
                    model_menu = gr.Dropdown(choices=available_models, value=model_name, label='Model')
                    create_refresh_button(model_menu, lambda : None, lambda : {"choices": get_available_models()}, "refresh-button")
                buttons["Generate"] = gr.Button("Generate")
                with gr.Row():
                    with gr.Column():
                        buttons["Continue"] = gr.Button("Continue")
                    with gr.Column():
                        buttons["Stop"] = gr.Button("Stop")
                if args.extensions is not None:
                    create_extensions_block()

            with gr.Column():
                with gr.Tab('Raw'):
                    output_textbox = gr.Textbox(lines=15, label='Output')
                with gr.Tab('Markdown'):
                    markdown = gr.Markdown()
                with gr.Tab('HTML'):
                    html = gr.HTML()

        gen_events.append(buttons["Generate"].click(generate_reply, [textbox, length_slider, preset_menu, model_menu], [output_textbox, markdown, html], show_progress=args.no_stream, api_name="textgen"))
        gen_events.append(textbox.submit(generate_reply, [textbox, length_slider, preset_menu, model_menu], [output_textbox, markdown, html], show_progress=args.no_stream))
        gen_events.append(buttons["Continue"].click(generate_reply, [output_textbox, length_slider, preset_menu, model_menu], [output_textbox, markdown, html], show_progress=args.no_stream))
        buttons["Stop"].click(None, None, None, cancels=gen_events)

interface.queue()
if args.listen:
    interface.launch(share=args.share, server_name="0.0.0.0", server_port=args.listen_port)
else:
    interface.launch(share=args.share, server_port=args.listen_port)
