

# 【环境1】

- NVIDIA A10 

eval-显存：19GB
  - 14min 跑完第一个 math/train.jsonl
  - outputs: 3000+ token/s

```
➜  /workspace nvidia-smi
Sun Mar  1 09:48:41 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.65.06              Driver Version: 580.65.06      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A10                     On  |   00000000:00:08.0 Off |                    0 |
|  0%   70C    P0            140W /  150W |   19463MiB /  23028MiB |     54%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
➜  /workspace 
```

# 【环境2】

- NVIDIA L40  

eval-显存：39GB
- eval-耗时：15min

```
 /workspace nvidia-smi 
Sun Mar  1 10:10:38 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.65.06              Driver Version: 580.65.06      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA L40                     On  |   00000000:23:00.0 Off |                    0 |
| N/A   44C    P0            222W /  300W |   39145MiB /  46068MiB |     63%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
➜  /workspace 
```


```
(alignment) ➜  assignment5-alignment git:(main) ✗ python eval.py       
INFO 03-01 10:09:10 __init__.py:190] Automatically detected platform cuda.
Using CUDA device cuda:0
`torch_dtype` is deprecated! Use `dtype` instead!
INFO 03-01 10:09:21 config.py:542] This model supports multiple tasks: {'score', 'reward', 'classify', 'embed', 'generate'}. Defaulting to 'generate'.
INFO 03-01 10:09:21 llm_engine.py:234] Initializing a V0 LLM engine (v0.7.2) with config: model='models/Qwen2.5-Math-1.5B', speculative_config=None, tokenizer='models/Qwen2.5-Math-1.5B', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=4096, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto,  device_config=cuda:0, decoding_config=DecodingConfig(guided_decoding_backend='xgrammar'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=models/Qwen2.5-Math-1.5B, num_scheduler_steps=1, multi_step_stream_outputs=True, enable_prefix_caching=True, chunked_prefill_enabled=False, use_async_output_proc=True, disable_mm_preprocessor_cache=False, mm_processor_kwargs=None, pooler_config=None, compilation_config={"splitting_ops":[],"compile_sizes":[],"cudagraph_capture_sizes":[256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"max_capture_size":256}, use_cached_outputs=False, 
INFO 03-01 10:09:22 cuda.py:230] Using Flash Attention backend.
INFO 03-01 10:09:22 model_runner.py:1110] Starting to load model models/Qwen2.5-Math-1.5B...
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:13<00:00, 13.45s/it]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:13<00:00, 13.45s/it]

INFO 03-01 10:09:37 model_runner.py:1115] Loading model weights took 2.8797 GB
INFO 03-01 10:09:39 worker.py:267] Memory profiling takes 1.39 seconds
INFO 03-01 10:09:39 worker.py:267] the current vLLM instance can use total_gpu_memory (44.39GiB) x gpu_memory_utilization (0.85) = 37.73GiB
INFO 03-01 10:09:39 worker.py:267] model weights take 2.88GiB; non_torch_memory takes 0.08GiB; PyTorch activation peak memory takes 1.40GiB; the rest of the memory reserved for KV Cache is 33.37GiB.
INFO 03-01 10:09:39 executor_base.py:110] # CUDA blocks: 78115, # CPU blocks: 9362
INFO 03-01 10:09:39 executor_base.py:115] Maximum concurrency for 4096 tokens per request: 305.14x
INFO 03-01 10:09:40 model_runner.py:1434] Capturing cudagraphs for decoding. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI. If out-of-memory error occurs during cudagraph capture, consider decreasing `gpu_memory_utilization` or switching to eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
Capturing CUDA graph shapes: 100%|███████████████████████████████████████████████████████████| 35/35 [00:17<00:00,  1.97it/s]
INFO 03-01 10:09:58 model_runner.py:1562] Graph capturing finished in 18 secs, took 0.20 GiB
INFO 03-01 10:09:58 llm_engine.py:431] init engine (profile, create kv cache, warmup model) took 20.92 seconds
Evaluating data/pre-processed/math/train.jsonl Train Set...
Processed prompts: 100%|█████| 12000/12000 [09:51<00:00, 20.29it/s, est. speed input: 3367.19 toks/s, output: 7217.58 toks/s]
{
│   'total': 12000,
│   'answer_correct': 351,
│   'format_correct': 2064,
│   'reward_1': 351,
│   'formatted_but_answer_wrong': 1713,
│   'answer_accuracy': 0.02925
}
Evaluating data/pre-processed/math/test.jsonl Test Set...
Processed prompts: 100%|█████████| 500/500 [00:26<00:00, 18.65it/s, est. speed input: 3060.57 toks/s, output: 6927.87 toks/s]
{
│   'total': 500,
│   'answer_correct': 11,
│   'format_correct': 87,
│   'reward_1': 11,
│   'formatted_but_answer_wrong': 76,
│   'answer_accuracy': 0.022
}
Evaluating data/pre-processed/gsm8k/train.jsonl Train Set...
Processed prompts: 100%|███████| 7473/7473 [03:52<00:00, 32.17it/s, est. speed input: 4902.37 toks/s, output: 7117.09 toks/s]
{
│   'total': 7473,
│   'answer_correct': 241,
│   'format_correct': 1447,
│   'reward_1': 241,
│   'formatted_but_answer_wrong': 1206,
│   'answer_accuracy': 0.0322494312859628
}
Evaluating data/pre-processed/gsm8k/test.jsonl Test Set...
Processed prompts: 100%|███████| 1319/1319 [00:41<00:00, 31.41it/s, est. speed input: 4838.65 toks/s, output: 6800.11 toks/s]
{
│   'total': 1319,
│   'answer_correct': 38,
│   'format_correct': 264,
│   'reward_1': 38,
│   'formatted_but_answer_wrong': 226,
│   'answer_accuracy': 0.02880970432145565
}
Saved evaluation overview to: outputs/eval_overview.csv
[rank0]:[W301 10:25:11.850725689 ProcessGroupNCCL.cpp:1250] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())
```