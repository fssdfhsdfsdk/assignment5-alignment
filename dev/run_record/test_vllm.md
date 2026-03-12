

```
(alignment) ➜  dev git:(main) ✗ unset LD_LIBRARY_PATH    
(alignment) ➜  dev git:(main) ✗ python test_vllm.py  
INFO 03-01 10:30:28 __init__.py:190] Automatically detected platform cuda.
`torch_dtype` is deprecated! Use `dtype` instead!
INFO 03-01 10:30:40 config.py:542] This model supports multiple tasks: {'generate', 'classify', 'embed', 'reward', 'score'}. Defaulting to 'generate'.
INFO 03-01 10:30:40 llm_engine.py:234] Initializing a V0 LLM engine (v0.7.2) with config: model='../models/Qwen2.5-Math-1.5B', speculative_config=None, tokenizer='../models/Qwen2.5-Math-1.5B', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=4096, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto,  device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='xgrammar'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=../models/Qwen2.5-Math-1.5B, num_scheduler_steps=1, multi_step_stream_outputs=True, enable_prefix_caching=False, chunked_prefill_enabled=False, use_async_output_proc=True, disable_mm_preprocessor_cache=False, mm_processor_kwargs=None, pooler_config=None, compilation_config={"splitting_ops":[],"compile_sizes":[],"cudagraph_capture_sizes":[256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"max_capture_size":256}, use_cached_outputs=False, 
INFO 03-01 10:30:42 cuda.py:230] Using Flash Attention backend.
INFO 03-01 10:30:42 model_runner.py:1110] Starting to load model ../models/Qwen2.5-Math-1.5B...
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:11<00:00, 11.17s/it]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:11<00:00, 11.17s/it]

INFO 03-01 10:30:55 model_runner.py:1115] Loading model weights took 2.8797 GB
INFO 03-01 10:30:57 worker.py:267] Memory profiling takes 1.48 seconds
INFO 03-01 10:30:57 worker.py:267] the current vLLM instance can use total_gpu_memory (22.06GiB) x gpu_memory_utilization (0.90) = 19.85GiB
INFO 03-01 10:30:57 worker.py:267] model weights take 2.88GiB; non_torch_memory takes 0.05GiB; PyTorch activation peak memory takes 1.40GiB; the rest of the memory reserved for KV Cache is 15.52GiB.
INFO 03-01 10:30:57 executor_base.py:110] # CUDA blocks: 36332, # CPU blocks: 9362
INFO 03-01 10:30:57 executor_base.py:115] Maximum concurrency for 4096 tokens per request: 141.92x
INFO 03-01 10:30:59 model_runner.py:1434] Capturing cudagraphs for decoding. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI. If out-of-memory error occurs during cudagraph capture, consider decreasing `gpu_memory_utilization` or switching to eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
Capturing CUDA graph shapes: 100%|███████████████████████████████████████████████████████████| 35/35 [00:19<00:00,  1.82it/s]
INFO 03-01 10:31:18 model_runner.py:1562] Graph capturing finished in 19 secs, took 0.18 GiB
INFO 03-01 10:31:18 llm_engine.py:431] init engine (profile, create kv cache, warmup model) took 23.09 seconds
Processed prompts: 100%|████████████████| 4/4 [00:01<00:00,  3.86it/s, est. speed input: 21.22 toks/s, output: 163.94 toks/s]
Prompt: 'Hello, my name is', Generated text: " Kailash. I'm having trouble understanding how to solve the following problem. Could you please help me?"
Prompt: 'The president of the United States is', Generated text: ' the highest-ranking official in the executive branch and is responsible for the overall administration of the country.'
Prompt: 'The capital of France is', Generated text: ' also the capital of another country. Which of the following options correctly fills in the blank to maintain the grammatical structure of the sentence?'
Prompt: 'The future of AI is', Generated text: " a topic of great interest, with various projects exploring different aspects. Suppose a middle school math club decides to explore the concept of AI and decides to model the growth of their AI club's membership over the next few years using a simple linear model. The club currently has 15 members and expects to gain 3 new members each year. If the club starts with 15 members and gains 3 new members each year, how many members will the club have at the end of 5 years?"
[rank0]:[W301 10:31:20.456031806 ProcessGroupNCCL.cpp:1250] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())
```

Prompt: 'Hello, my name is'
 - Generated text: " Kailash. I'm having trouble understanding how to solve the following problem. Could you please help me?"

Prompt: 'The president of the United States is'
 - Generated text: ' the highest-ranking official in the executive branch and is responsible for the overall administration of the country.'

Prompt: 'The capital of France is'
 - Generated text: ' also the capital of another country. Which of the following options correctly fills in the blank to maintain the grammatical structure of the sentence?'

Prompt: 'The future of AI is'
 - Generated text: " a topic of great interest, with various projects exploring different aspects. Suppose a middle school math club decides to explore the concept of AI and decides to model the growth of their AI club's membership over the next few years using a simple linear model. The club currently has 15 members and expects to gain 3 new members each year. If the club starts with 15 members and gains 3 new members each year, how many members will the club have at the end of 5 years?"