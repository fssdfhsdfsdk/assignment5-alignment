

# 【1】ok

```
(alignment) ➜  assignment5-alignment git:(main) ✗ python eval.py                         
INFO 03-01 09:41:12 __init__.py:190] Automatically detected platform cuda.
Using CUDA device cuda:1
`torch_dtype` is deprecated! Use `dtype` instead!
INFO 03-01 09:41:26 config.py:542] This model supports multiple tasks: {'generate', 'classify', 'score', 'embed', 'reward'}. Defaulting to 'generate'.
INFO 03-01 09:41:26 llm_engine.py:234] Initializing a V0 LLM engine (v0.7.2) with config: model='models/Qwen2.5-Math-1.5B', speculative_config=None, tokenizer='models/Qwen2.5-Math-1.5B', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=4096, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto,  device_config=cuda:1, decoding_config=DecodingConfig(guided_decoding_backend='xgrammar'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=models/Qwen2.5-Math-1.5B, num_scheduler_steps=1, multi_step_stream_outputs=True, enable_prefix_caching=True, chunked_prefill_enabled=False, use_async_output_proc=True, disable_mm_preprocessor_cache=False, mm_processor_kwargs=None, pooler_config=None, compilation_config={"splitting_ops":[],"compile_sizes":[],"cudagraph_capture_sizes":[256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"max_capture_size":256}, use_cached_outputs=False, 
INFO 03-01 09:41:32 cuda.py:230] Using Flash Attention backend.
INFO 03-01 09:41:33 model_runner.py:1110] Starting to load model models/Qwen2.5-Math-1.5B...
[rank0]: Traceback (most recent call last):
[rank0]:   File "/workspace/Stanford-CS336/assignment5-alignment/eval.py", line 22, in <module>
[rank0]:     vllm = init_vllm(
[rank0]:            ^^^^^^^^^^
[rank0]:   File "/workspace/Stanford-CS336/assignment5-alignment/cs336_alignment/vllm_utils.py", line 16, in init_vllm
[rank0]:     return LLM(
[rank0]:            ^^^^
[rank0]:   File "/workspace/Stanford-CS336/assignment5-alignment/.venv/lib/python3.12/site-packages/vllm/utils.py", line 1051, in inner
[rank0]:     return fn(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/workspace/Stanford-CS336/assignment5-alignment/.venv/lib/python3.12/site-packages/vllm/entrypoints/llm.py", line 242, in __init__
[rank0]:     self.llm_engine = self.engine_class.from_engine_args(
[rank0]:                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/workspace/Stanford-CS336/assignment5-alignment/.venv/lib/python3.12/site-packages/vllm/engine/llm_engine.py", line 484, in from_engine_args
[rank0]:     engine = cls(
[rank0]:              ^^^^
[rank0]:   File "/workspace/Stanford-CS336/assignment5-alignment/.venv/lib/python3.12/site-packages/vllm/engine/llm_engine.py", line 273, in __init__
[rank0]:     self.model_executor = executor_class(vllm_config=vllm_config, )
[rank0]:                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/workspace/Stanford-CS336/assignment5-alignment/.venv/lib/python3.12/site-packages/vllm/executor/executor_base.py", line 51, in __init__
[rank0]:     self._init_executor()
[rank0]:   File "/workspace/Stanford-CS336/assignment5-alignment/.venv/lib/python3.12/site-packages/vllm/executor/uniproc_executor.py", line 42, in _init_executor
[rank0]:     self.collective_rpc("load_model")
[rank0]:   File "/workspace/Stanford-CS336/assignment5-alignment/.venv/lib/python3.12/site-packages/vllm/executor/uniproc_executor.py", line 51, in collective_rpc
[rank0]:     answer = run_method(self.driver_worker, method, args, kwargs)
[rank0]:              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/workspace/Stanford-CS336/assignment5-alignment/.venv/lib/python3.12/site-packages/vllm/utils.py", line 2220, in run_method
[rank0]:     return func(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/workspace/Stanford-CS336/assignment5-alignment/.venv/lib/python3.12/site-packages/vllm/worker/worker.py", line 183, in load_model
[rank0]:     self.model_runner.load_model()
[rank0]:   File "/workspace/Stanford-CS336/assignment5-alignment/.venv/lib/python3.12/site-packages/vllm/worker/model_runner.py", line 1112, in load_model
[rank0]:     self.model = get_model(vllm_config=self.vllm_config)
[rank0]:                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/workspace/Stanford-CS336/assignment5-alignment/.venv/lib/python3.12/site-packages/vllm/model_executor/model_loader/__init__.py", line 14, in get_model
[rank0]:     return loader.load_model(vllm_config=vllm_config)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/workspace/Stanford-CS336/assignment5-alignment/.venv/lib/python3.12/site-packages/vllm/model_executor/model_loader/loader.py", line 383, in load_model
[rank0]:     model = _initialize_model(vllm_config=vllm_config)
[rank0]:             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/workspace/Stanford-CS336/assignment5-alignment/.venv/lib/python3.12/site-packages/vllm/model_executor/model_loader/loader.py", line 125, in _initialize_model
[rank0]:     return model_class(vllm_config=vllm_config, prefix=prefix)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/workspace/Stanford-CS336/assignment5-alignment/.venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen2.py", line 453, in __init__
[rank0]:     self.model = Qwen2Model(vllm_config=vllm_config,
[rank0]:                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/workspace/Stanford-CS336/assignment5-alignment/.venv/lib/python3.12/site-packages/vllm/compilation/decorators.py", line 151, in __init__
[rank0]:     old_init(self, vllm_config=vllm_config, prefix=prefix, **kwargs)
[rank0]:   File "/workspace/Stanford-CS336/assignment5-alignment/.venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen2.py", line 298, in __init__
[rank0]:     self.embed_tokens = VocabParallelEmbedding(
[rank0]:                         ^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/workspace/Stanford-CS336/assignment5-alignment/.venv/lib/python3.12/site-packages/vllm/model_executor/layers/vocab_parallel_embedding.py", line 263, in __init__
[rank0]:     self.linear_method.create_weights(self,
[rank0]:   File "/workspace/Stanford-CS336/assignment5-alignment/.venv/lib/python3.12/site-packages/vllm/model_executor/layers/vocab_parallel_embedding.py", line 31, in create_weights
[rank0]:     weight = Parameter(torch.empty(sum(output_partition_sizes),
[rank0]:                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/workspace/Stanford-CS336/assignment5-alignment/.venv/lib/python3.12/site-packages/torch/utils/_device.py", line 106, in __torch_function__
[rank0]:     return func(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^
[rank0]: RuntimeError: CUDA error: invalid device ordinal
[rank0]: CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
[rank0]: For debugging consider passing CUDA_LAUNCH_BLOCKING=1
[rank0]: Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

[rank0]:[W301 09:41:34.541498631 ProcessGroupNCCL.cpp:1250] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())
```

问题代码：环境只有1个GPU，必须是rank=0

```
if __name__ == "__main__":
    vllm = init_vllm(
        model_id=MODEL_NAME,
        device=str(get_device(rank=1)),
        seed=42,
        gpu_memory_utilization=0.85,
    )

```

The `RuntimeError: CUDA error: invalid device ordinal` usually means your code is trying to access a GPU index that doesn't exist or isn't "visible" to your current process.

In your log, I see:

`Using CUDA device cuda:1`

This suggests that somewhere in your script (likely `eval.py` or `vllm_utils.py`), the device is being explicitly set to index **1**. If your environment only has one GPU visible, PyTorch indexes it as `0`. Even if you were physically assigned "GPU 1" on a cluster, if you used `export CUDA_VISIBLE_DEVICES=1`, PyTorch re-maps that physical GPU to logical index `0`.