import functools

import gradio as gr

from modules import shared

loaders_and_params = {
    'AutoGPTQ': [
        'triton',
        'quant_attn',
        'fused_mlp',
        'wbits',
        'groupsize',
        'warmup_autotune',
        'autogptq_act_order',
        'gpu_memory',
        'cpu_memory',
        'cpu',
        'disk',
#        'autogptq_device_map',
        'trust_remote_code',
        'autogptq_info',
        'attention_info',
        'flash_attention',
        'xformers',
        'sdp_attention',
        'no_cache',

    ],
    'GPTQ-for-LLaMa': [
        'wbits',
        'groupsize',
        'model_type',
        'pre_layer',
        'autograd',
        'v1',
        'quant_attn',
        'fused_mlp',
        'warmup_autotune',
        'gptq_for_llama_info',
        'attention_info',
        'flash_attention',
        'xformers',
        'sdp_attention',
        'no_cache',

    ],
    'llama.cpp': [
        'n_ctx',
        'n_gpu_layers',
        'n_batch',
        'threads',
        'no_mmap',
        'mlock',
        'llama_cpp_seed',
    ],
    'Transformers': [
        'cpu_memory',
        'gpu_memory',
        'trust_remote_code',
        'load_in_8bit',
        'threshold',
        'bf16',
        'cpu',
        'disk',
        'auto_devices',
        'load_in_4bit',
        'use_double_quant',
        'quant_type',
        'compute_dtype',
        'trust_remote_code',
        'transformers_info',
        'attention_info',
        'flash_attention',
        'xformers',
        'sdp_attention',
        'no_cache',

    ],
    'ExLlama' : [
        'gpu_split',
        'nohalf2',
        'max_seq_len',
        'compress_pos_emb',
        'exllama_info',
    ],
    'ExLlama_HF' : [
        'gpu_split',
        'nohalf2',
        'quant_attn',
        'fused_mlp',
        'max_seq_len',
        'compress_pos_emb',
        'exllama_HF_info',
    ]
}


def get_gpu_memory_keys():
    return [k for k in shared.gradio if k.startswith('gpu_memory')]


@functools.cache
def get_all_params():
    all_params = set()
    for k in loaders_and_params:
        for el in loaders_and_params[k]:
            all_params.add(el)

    if 'gpu_memory' in all_params:
        all_params.remove('gpu_memory')
        for k in get_gpu_memory_keys():
            all_params.add(k)

    return sorted(all_params)


def make_loader_params_visible(loader):
    params = []
    all_params = get_all_params()
    if loader in loaders_and_params:
        params = loaders_and_params[loader]

        if 'gpu_memory' in params:
            params.remove('gpu_memory')
            params += get_gpu_memory_keys()

    return [gr.update(visible=True) if k in params else gr.update(visible=False) for k in all_params]
