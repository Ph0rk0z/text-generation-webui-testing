import importlib
import math
import re
import traceback
from functools import partial
from pathlib import Path

import gradio as gr
import psutil
import torch
from transformers import is_torch_xpu_available

from modules import loaders, shared, ui, utils
from modules.logging_colors import logger
from modules.LoRA import add_lora_to_model
from modules.models import load_model, unload_model
from modules.models_settings import (
    apply_model_settings_to_state,
    get_model_metadata,
    save_instruction_template,
    save_model_settings,
    update_model_parameters
)
from modules.utils import gradio


def create_ui():
    mu = shared.args.multi_user

    # Finding the default values for the GPU and CPU memories
    total_mem = []
    if is_torch_xpu_available():
        for i in range(torch.xpu.device_count()):
            total_mem.append(math.floor(torch.xpu.get_device_properties(i).total_memory / (1024 * 1024)))
    else:
        for i in range(torch.cuda.device_count()):
            total_mem.append(math.floor(torch.cuda.get_device_properties(i).total_memory / (1024 * 1024)))

    default_gpu_mem = []
    if shared.args.gpu_memory is not None and len(shared.args.gpu_memory) > 0:
        for i in shared.args.gpu_memory:
            if 'mib' in i.lower():
                default_gpu_mem.append(int(re.sub('[a-zA-Z ]', '', i)))
            else:
                default_gpu_mem.append(int(re.sub('[a-zA-Z ]', '', i)) * 1000)

    while len(default_gpu_mem) < len(total_mem):
        default_gpu_mem.append(0)

    total_cpu_mem = math.floor(psutil.virtual_memory().total / (1024 * 1024))
    if shared.args.cpu_memory is not None:
        default_cpu_mem = re.sub('[a-zA-Z ]', '', shared.args.cpu_memory)
    else:
        default_cpu_mem = 0

    with gr.Tab("Model", elem_id="model-tab"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            shared.gradio['model_menu'] = gr.Dropdown(choices=utils.get_available_models(), value=lambda: shared.model_name, label='Model', elem_classes='slim-dropdown', interactive=not mu)
                            ui.create_refresh_button(shared.gradio['model_menu'], lambda: None, lambda: {'choices': utils.get_available_models()}, 'refresh-button', interactive=not mu)
                            shared.gradio['load_model'] = gr.Button("Load", visible=not shared.settings['autoload_model'], elem_classes='refresh-button', interactive=not mu)
                            shared.gradio['unload_model'] = gr.Button("Unload", elem_classes='refresh-button', interactive=not mu)
                            shared.gradio['reload_model'] = gr.Button("Reload", elem_classes='refresh-button', interactive=not mu)
                            shared.gradio['save_model_settings'] = gr.Button("Save settings", elem_classes='refresh-button', interactive=not mu)

                    with gr.Column():
                        with gr.Row():
                            shared.gradio['lora_menu'] = gr.Dropdown(multiselect=True, choices=utils.get_available_loras(), value=shared.lora_names, label='LoRA(s)', elem_classes='slim-dropdown', interactive=not mu)
                            ui.create_refresh_button(shared.gradio['lora_menu'], lambda: None, lambda: {'choices': utils.get_available_loras(), 'value': shared.lora_names}, 'refresh-button', interactive=not mu)
                            shared.gradio['lora_menu_apply'] = gr.Button(value='Apply LoRAs', elem_classes='refresh-button', interactive=not mu)

        with gr.Row():
            with gr.Column():
                shared.gradio['loader'] = gr.Dropdown(label="Model loader", choices=loaders.loaders_and_params.keys(), value=None)
                with gr.Box():
                    with gr.Row():
                        with gr.Column():
                            with gr.Blocks():
                                for i in range(len(total_mem)):
                                    shared.gradio[f'gpu_memory_{i}'] = gr.Slider(label=f"gpu-memory in MiB for device :{i}", maximum=total_mem[i], value=default_gpu_mem[i])

                            shared.gradio['cpu_memory'] = gr.Slider(label="cpu-memory in MiB", maximum=total_cpu_mem, value=default_cpu_mem)
                            shared.gradio['auto_devices'] = gr.Checkbox(label="auto-devices", value=shared.args.auto_devices)
                          
                        # 8 Bit
                            shared.gradio['load_in_8bit'] = gr.Checkbox(label="load-in-8bit", value=shared.args.load_in_8bit)
                            shared.gradio['threshold'] = gr.Slider(label="8bit threshold", minimum=0.0, maximum=10.0, value=shared.args.threshold, info='Threshold for 8bit on older cards if you did not patch BnB' )
                        # Rope
                            #with gr.Blocks():
                            shared.gradio['alpha_value'] = gr.Slider(label='alpha_value', minimum=1, maximum=32, step=0.1, info='Positional embeddings alpha factor for NTK RoPE scaling. Recommended values (NTKv1): 1.75 for 1.5x context, 2.5 for 2x context. Use either this or compress_pos_emb, not both.', value=shared.args.alpha_value)
                            shared.gradio['rope_freq_base'] = gr.Slider(label='rope_freq_base', minimum=0, maximum=1000000, step=1000, info='If greater than 0, will be used instead of alpha_value. Those two are related by rope_freq_base = 10000 * alpha_value ^ (64 / 63)', value=shared.args.rope_freq_base)
                            shared.gradio['compress_pos_emb'] = gr.Slider(label='compress_pos_emb', minimum=1, maximum=8, step=1, info='Positional embeddings compression factor. Should be set to (context length) / (model\'s original context length). Equal to 1/rope_freq_scale.', value=shared.args.compress_pos_emb)
            
                        with gr.Column():
                            # Transformers 4 bit
                            shared.gradio['transformers_info'] = gr.Markdown('load-in-4bit params:')                      
                            shared.gradio['load_in_4bit'] = gr.Checkbox(label="load-in-4bit", value=shared.args.load_in_4bit)
                            shared.gradio['use_double_quant'] = gr.Checkbox(label="use_double_quant", value=shared.args.use_double_quant)
                    

                            shared.gradio['compute_dtype'] = gr.Dropdown(label="compute_dtype", choices=["bfloat16", "float16", "float32"], value=shared.args.compute_dtype)
                            shared.gradio['quant_type'] = gr.Dropdown(label="quant_type", choices=["nf4", "fp4"], value=shared.args.quant_type)               
                            shared.gradio['bf16'] = gr.Checkbox(label="bf16", value=shared.args.bf16, info='Use BF16')
                            shared.gradio['use_flash_attention_2'] = gr.Checkbox(label="use_flash_attention_2", value=shared.args.use_flash_attention_2, info='Transformers Flash attenton 2')
                            # Low End                        
                            shared.gradio['no_cache'] = gr.Checkbox(label="no_cache", value=shared.args.no_cache, info='Disable Generation Cache')
                            shared.gradio['cpu'] = gr.Checkbox(label="cpu", value=shared.args.cpu, info='Offload to CPU')
                            shared.gradio['disk'] = gr.Checkbox(label="disk", value=shared.args.disk, info='Offload to Disk')

                    with gr.Row():
                        with gr.Column():
                        # ExLlama
                            shared.gradio['gpu_split'] = gr.Textbox(label='gpu-split', info='Comma-separated list of VRAM (in GB) to use per GPU. Example: 20,7,7')
                            shared.gradio['max_seq_len'] = gr.Slider(label='max_seq_len', minimum=0, maximum=shared.settings['truncation_length_max'], step=256, info='Context length. Try lowering this if you run out of memory while loading the model.', value=shared.args.max_seq_len)
                            shared.gradio['autosplit'] = gr.Checkbox(label="autosplit", value=shared.args.autosplit, info='Automatically split the model tensors across the available GPUs.')
                            shared.gradio['no_flash_attn'] = gr.Checkbox(label="no_flash_attn", value=shared.args.no_flash_attn, info='Force flash-attention to not be used.')
                            shared.gradio['cache_8bit'] = gr.Checkbox(label="cache_8bit", value=shared.args.cache_8bit, info='Use 8-bit cache to save VRAM.')
                            shared.gradio['cache_4bit'] = gr.Checkbox(label="cache_4bit", value=shared.args.cache_4bit, info='Use Q4 cache to save VRAM.')

                    with gr.Row():
                        with gr.Column():
                        # GPTQ
                            shared.gradio['wbits'] = gr.Dropdown(label="wbits", choices=["None", 1, 2, 3, 4, 8], value=shared.args.wbits if shared.args.wbits > 0 else "None")
                            shared.gradio['groupsize'] = gr.Dropdown(label="groupsize", choices=["None", 32, 64, 128, 1024], value=shared.args.groupsize if shared.args.groupsize > 0 else "None")
                            shared.gradio['model_type'] = gr.Dropdown(label="model_type", choices=["None"], value=shared.args.model_type or "None")

                                                           
                        # AutoGPTQ
                            shared.gradio['quant_attn'] = gr.Checkbox(label="quant_attn", value=shared.args.quant_attn, info='Enable fused attention. AutoGPTQ/Autograd/ExLlama')
                            shared.gradio['fused_mlp'] = gr.Checkbox(label="fused_mlp", value=shared.args.fused_mlp, info='Enable fused MLP. AutoGPTQ (triton)/Autograd/ExLlama')
                            shared.gradio['disable_exllama'] = gr.Checkbox(label="disable_exllama", value=shared.args.warmup_autotune, info='Disable exllama kernel. Use for P40/P6000')
                            shared.gradio['disable_exllamav2'] = gr.Checkbox(label="disable_exllamav2", value=shared.args.disable_exllamav2, info='Disable ExLlamav2 kernel.')
                        with gr.Column():

                        # Autograd
                            shared.gradio['autograd'] = gr.Checkbox(label="Autograd", value=shared.args.autograd, info='4bit Lora Support and Inference')
                            shared.gradio['v1'] = gr.Checkbox(label="GPTQv1 Model (Autograd Only)", value=shared.args.v1, info='V1 Models. Pre Groupsize')
                            shared.gradio['pre_layer'] = gr.Textbox(label="pre_layer", value=shared.args.pre_layer[0] if shared.args.pre_layer is not None else 0, info='Llama only: Pre layer. For multi-gpu, write the numbers separated by spaces, ie: 30 60 More layers than the model will cause error')
                            shared.gradio['autogptq_act_order'] = gr.Checkbox(label="autogptq_act_order", value=shared.args.autogptq_act_order, info='Enable act_order and groupsize together')
                            shared.gradio['triton'] = gr.Checkbox(label="triton", value=shared.args.triton, info='Autogptq Triton Backend')
                            shared.gradio['warmup_autotune'] = gr.Checkbox(label="warmup_autotune", value=shared.args.warmup_autotune, info='Enable warmup autotune if using triton')

                   # with gr.Row():
                       # with gr.Column():
                        # Attention Hijacks
                            shared.gradio['attention_info'] = gr.Markdown('Hijack Attention via:')
                            shared.gradio['xformers'] = gr.Checkbox(label="xformers", value=shared.args.quant_attn, info='Hijack attention with xformers')
                            shared.gradio['sdp_attention'] = gr.Checkbox(label="sdp_attention", value=shared.args.sdp_attention, info='Hijack attention via Torch 2.0 SDP attention')
                            shared.gradio['flash_attention'] = gr.Checkbox(label="flash_attention", value=shared.args.flash_attention, info='Flash attention 2. Ampere and up.')
                    with gr.Row():
                            with gr.Column():
                        # Llama.cpp
                                
                                shared.gradio['main_gpu'] = gr.Number(label='Main GPU', value=shared.args.main_gpu)
                                shared.gradio['n_gpu_layers'] = gr.Slider(label="n-gpu-layers", minimum=0, maximum=256, value=shared.args.n_gpu_layers)
                                shared.gradio['n_ctx'] = gr.Slider(minimum=0, maximum=shared.settings['truncation_length_max'], step=256, label="n_ctx", value=shared.args.n_ctx, info='Context length. Try lowering this if you run out of memory while loading the model.')
                                shared.gradio['threads'] = gr.Slider(label="threads", minimum=0, step=1, maximum=96, value=shared.args.threads)
                                shared.gradio['threads_batch'] = gr.Slider(label="threads_batch", minimum=0, step=1, maximum=32, value=shared.args.threads_batch)
                                shared.gradio['n_batch'] = gr.Slider(label="n_batch", minimum=1, maximum=2048, value=shared.args.n_batch)
                                shared.gradio['attention_sink_size'] = gr.Number(label="attention_sink_size", value=shared.args.attention_sink_size, info='StreamingLLM: number of sink tokens. Only used if the trimmed prompt doesn\'t share a prefix with the old prompt.')

                            with gr.Column():
                                shared.gradio['tensor_split'] = gr.Textbox(label='tensor_split', info='Split the model across multiple GPUs, comma-separated list of proportions, e.g. 18,17')

                                shared.gradio['numa'] = gr.Checkbox(label="numa support", value=shared.args.numa)
                                shared.gradio['no_mmap'] = gr.Checkbox(label="no-mmap", value=shared.args.no_mmap)
                                shared.gradio['mlock'] = gr.Checkbox(label="mlock", value=shared.args.mlock)
                                shared.gradio['no_mul_mat_q'] = gr.Checkbox(label="no_mul_mat_q", value=shared.args.no_mul_mat_q, info='Disable the mulmat kernels.')
                                shared.gradio['row_split'] = gr.Checkbox(label="row_split", value=shared.args.row_split, info='Split the model by rows across GPUs. This may improve multi-gpu performance.')
                                shared.gradio['logits_all'] = gr.Checkbox(label="logits_all", value=shared.args.logits_all, info='Needs to be set for perplexity evaluation to work. Otherwise, ignore it, as it makes prompt processing slower.')
                                shared.gradio['no_offload_kqv'] = gr.Checkbox(label="no_offload_kqv", value=shared.args.no_offload_kqv, info='Do not offload the  K, Q, V to the GPU. This saves VRAM but reduces the performance.')
                                shared.gradio['streaming_llm'] = gr.Checkbox(label="streaming_llm", value=shared.args.streaming_llm, info='(experimental) Activate StreamingLLM to avoid re-evaluating the entire prompt when old messages are removed.')
                                                           
                    with gr.Row():  
                        with gr.Column():
                            shared.gradio['cfg_cache'] = gr.Checkbox(label="cfg-cache", value=shared.args.cfg_cache, info='Create an additional cache for CFG negative prompts.')
                            shared.gradio['num_experts_per_token'] = gr.Number(label="Number of experts per token", value=shared.args.num_experts_per_token, info='Only applies to MoE models like Mixtral.')              
                        # Security
                            shared.gradio['trust_remote_code'] = gr.Checkbox(label="trust-remote-code", value=shared.args.trust_remote_code, info='To enable this option, start the web UI with the --trust-remote-code flag. It is necessary for some models.')
                            shared.gradio['no_use_fast'] = gr.Checkbox(label="no_use_fast", value=shared.args.no_use_fast, info='Set use_fast=False while loading the tokenizer.')

                        # Infos
                            shared.gradio['gptq_for_llama_info'] = gr.Markdown('GPTQ-for-LLaMa. The original GPTQ. Can be used with Autograd for 4-bit lora and sometimes faster inference. Also lora training in 4bits')
                            shared.gradio['hqq_backend'] = gr.Dropdown(label="hqq_backend", choices=["PYTORCH", "PYTORCH_COMPILE", "ATEN"], value=shared.args.hqq_backend)
                            shared.gradio['exllamav2_info'] = gr.Markdown("ExLlamav2_HF is recommended over ExLlamav2 for better integration with extensions and more consistent sampling behavior across loaders.")
                            shared.gradio['llamacpp_HF_info'] = gr.Markdown('llamacpp_HF is a wrapper that lets you use llama.cpp like a Transformers model, which means it can use the Transformers samplers. Requires model tokenizer.')
                            shared.gradio['autogptq_info'] = gr.Markdown('AutoGPTQ supports GPTQ quantized models of various types.')
                            shared.gradio['quipsharp_info'] = gr.Markdown('QuIP# only works on Linux.')


            with gr.Column():
                with gr.Row():
                    shared.gradio['autoload_model'] = gr.Checkbox(value=shared.settings['autoload_model'], label='Autoload the model', info='Whether to load the model as soon as it is selected in the Model dropdown.', interactive=not mu)

                with gr.Tab("Download"):
                    shared.gradio['custom_model_menu'] = gr.Textbox(label="Download model or LoRA", info="Enter the Hugging Face username/model path, for instance: facebook/galactica-125m. To specify a branch, add it at the end after a \":\" character like this: facebook/galactica-125m:main. To download a single file, enter its name in the second box.", interactive=not mu)
                    shared.gradio['download_specific_file'] = gr.Textbox(placeholder="File name (for GGUF models)", show_label=False, max_lines=1, interactive=not mu)
                    with gr.Row():
                        shared.gradio['download_model_button'] = gr.Button("Download", variant='primary', interactive=not mu)
                        shared.gradio['get_file_list'] = gr.Button("Get file list", interactive=not mu)

                with gr.Tab("llamacpp_HF creator"):
                    with gr.Row():
                        shared.gradio['gguf_menu'] = gr.Dropdown(choices=utils.get_available_ggufs(), value=lambda: shared.model_name, label='Choose your GGUF', elem_classes='slim-dropdown', interactive=not mu)
                        ui.create_refresh_button(shared.gradio['gguf_menu'], lambda: None, lambda: {'choices': utils.get_available_ggufs()}, 'refresh-button', interactive=not mu)

                    shared.gradio['unquantized_url'] = gr.Textbox(label="Enter the URL for the original (unquantized) model", info="Example: https://huggingface.co/lmsys/vicuna-13b-v1.5", max_lines=1)
                    shared.gradio['create_llamacpp_hf_button'] = gr.Button("Submit", variant="primary", interactive=not mu)
                    gr.Markdown("This will move your gguf file into a subfolder of `models` along with the necessary tokenizer files.")

                with gr.Tab("Customize instruction template"):
                    with gr.Row():
                        shared.gradio['customized_template'] = gr.Dropdown(choices=utils.get_available_instruction_templates(), value='None', label='Select the desired instruction template', elem_classes='slim-dropdown')
                        ui.create_refresh_button(shared.gradio['customized_template'], lambda: None, lambda: {'choices': utils.get_available_instruction_templates()}, 'refresh-button', interactive=not mu)

                    shared.gradio['customized_template_submit'] = gr.Button("Submit", variant="primary", interactive=not mu)
                    gr.Markdown("This allows you to set a customized template for the model currently selected in the \"Model loader\" menu. Whenever the model gets loaded, this template will be used in place of the template specified in the model's medatada, which sometimes is wrong.")

                with gr.Row():
                    shared.gradio['model_status'] = gr.Markdown('No model is loaded' if shared.model_name == 'None' else 'Ready')


def create_event_handlers():
    shared.gradio['loader'].change(
        loaders.make_loader_params_visible, gradio('loader'), gradio(loaders.get_all_params())).then(
        lambda value: gr.update(choices=loaders.get_model_types(value)), gradio('loader'), gradio('model_type'))

    # In this event handler, the interface state is read and updated
    # with the model defaults (if any), and then the model is loaded
    # unless "autoload_model" is unchecked
    shared.gradio['model_menu'].change(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        apply_model_settings_to_state, gradio('model_menu', 'interface_state'), gradio('interface_state')).then(
        ui.apply_interface_values, gradio('interface_state'), gradio(ui.list_interface_input_elements()), show_progress=False).then(
        update_model_parameters, gradio('interface_state'), None).then(
        load_model_wrapper, gradio('model_menu', 'loader', 'autoload_model'), gradio('model_status'), show_progress=False).success(
        update_truncation_length, gradio('truncation_length', 'interface_state'), gradio('truncation_length')).then(
        lambda x: x, gradio('loader'), gradio('filter_by_loader'))

    shared.gradio['load_model'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        update_model_parameters, gradio('interface_state'), None).then(
        partial(load_model_wrapper, autoload=True), gradio('model_menu', 'loader'), gradio('model_status'), show_progress=False).success(
        update_truncation_length, gradio('truncation_length', 'interface_state'), gradio('truncation_length')).then(
        lambda x: x, gradio('loader'), gradio('filter_by_loader'))

    shared.gradio['reload_model'].click(
        unload_model, None, None).then(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        update_model_parameters, gradio('interface_state'), None).then(
        partial(load_model_wrapper, autoload=True), gradio('model_menu', 'loader'), gradio('model_status'), show_progress=False).success(
        update_truncation_length, gradio('truncation_length', 'interface_state'), gradio('truncation_length')).then(
        lambda x: x, gradio('loader'), gradio('filter_by_loader'))

    shared.gradio['unload_model'].click(
        unload_model, None, None).then(
        lambda: "Model unloaded", None, gradio('model_status'))

    shared.gradio['save_model_settings'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        save_model_settings, gradio('model_menu', 'interface_state'), gradio('model_status'), show_progress=False)

    shared.gradio['lora_menu_apply'].click(load_lora_wrapper, gradio('lora_menu'), gradio('model_status'), show_progress=False)
    shared.gradio['download_model_button'].click(download_model_wrapper, gradio('custom_model_menu', 'download_specific_file'), gradio('model_status'), show_progress=True)
    shared.gradio['get_file_list'].click(partial(download_model_wrapper, return_links=True), gradio('custom_model_menu', 'download_specific_file'), gradio('model_status'), show_progress=True)
    shared.gradio['autoload_model'].change(lambda x: gr.update(visible=not x), gradio('autoload_model'), gradio('load_model'))
    shared.gradio['create_llamacpp_hf_button'].click(create_llamacpp_hf, gradio('gguf_menu', 'unquantized_url'), gradio('model_status'), show_progress=True)
    shared.gradio['customized_template_submit'].click(save_instruction_template, gradio('model_menu', 'customized_template'), gradio('model_status'), show_progress=True)


def load_model_wrapper(selected_model, loader, autoload=False):
    if not autoload:
        yield f"The settings for `{selected_model}` have been updated.\n\nClick on \"Load\" to load it."
        return

    if selected_model == 'None':
        yield "No model selected"
    else:
        try:
            yield f"Loading `{selected_model}`..."
            unload_model()
            if selected_model != '':
                shared.model, shared.tokenizer = load_model(selected_model, loader)

            if shared.model is not None:
                output = f"Successfully loaded `{selected_model}`."

                settings = get_model_metadata(selected_model)
                if 'instruction_template' in settings:
                    output += '\n\nIt seems to be an instruction-following model with template "{}". In the chat tab, instruct or chat-instruct modes should be used.'.format(settings['instruction_template'])

                yield output
            else:
                yield f"Failed to load `{selected_model}`."
        except:
            exc = traceback.format_exc()
            logger.error('Failed to load the model.')
            print(exc)
            yield exc.replace('\n', '\n\n')


def load_lora_wrapper(selected_loras):
    yield ("Applying the following LoRAs to {}:\n\n{}".format(shared.model_name, '\n'.join(selected_loras)))
    add_lora_to_model(selected_loras)
    yield ("Successfuly applied the LoRAs")


def download_model_wrapper(repo_id, specific_file, progress=gr.Progress(), return_links=False, check=False):
    try:
        downloader = importlib.import_module("download-model").ModelDownloader()

        progress(0.0)
        model, branch = downloader.sanitize_model_and_branch_names(repo_id, None)

        yield ("Getting the download links from Hugging Face")
        links, sha256, is_lora, is_llamacpp = downloader.get_download_links_from_huggingface(model, branch, text_only=False, specific_file=specific_file)
        if return_links:
            output = "```\n"
            for link in links:
                output += f"{Path(link).name}" + "\n"

            output += "```"
            yield output
            return

        yield ("Getting the output folder")
        output_folder = downloader.get_output_folder(model, branch, is_lora, is_llamacpp=is_llamacpp)
        if check:
            progress(0.5)

            yield ("Checking previously downloaded files")
            downloader.check_model_files(model, branch, links, sha256, output_folder)
            progress(1.0)
        else:
            yield (f"Downloading file{'s' if len(links) > 1 else ''} to `{output_folder}`")
            downloader.download_model_files(model, branch, links, sha256, output_folder, progress_bar=progress, threads=4, is_llamacpp=is_llamacpp)

            yield (f"Model successfully saved to `{output_folder}/`.")
    except:
        progress(1.0)
        yield traceback.format_exc().replace('\n', '\n\n')


def create_llamacpp_hf(gguf_name, unquantized_url, progress=gr.Progress()):
    try:
        downloader = importlib.import_module("download-model").ModelDownloader()

        progress(0.0)
        model, branch = downloader.sanitize_model_and_branch_names(unquantized_url, None)

        yield ("Getting the tokenizer files links from Hugging Face")
        links, sha256, is_lora, is_llamacpp = downloader.get_download_links_from_huggingface(model, branch, text_only=True)
        output_folder = Path(shared.args.model_dir) / (re.sub(r'(?i)\.gguf$', '', gguf_name) + "-HF")

        yield (f"Downloading tokenizer to `{output_folder}`")
        downloader.download_model_files(model, branch, links, sha256, output_folder, progress_bar=progress, threads=4, is_llamacpp=False)

        # Move the GGUF
        (Path(shared.args.model_dir) / gguf_name).rename(output_folder / gguf_name)

        yield (f"Model saved to `{output_folder}/`.\n\nYou can now load it using llamacpp_HF.")
    except:
        progress(1.0)
        yield traceback.format_exc().replace('\n', '\n\n')


def update_truncation_length(current_length, state):
    if 'loader' in state:
        if state['loader'].lower().startswith('exllama'):
            return state['max_seq_len']
        elif state['loader'] in ['llama.cpp', 'llamacpp_HF', 'ctransformers']:
            return state['n_ctx']

    return current_length
