

# 【1】ok

```
(alignment) ➜  assignment5-alignment git:(main) ✗ python download_model.py \
  --repo-id Qwen/Qwen2.5-Math-1.5B \
  --save-dir models/Qwen2.5-Math-1.5B \
  --method snapshot --no-symlinks --verify
[download] snapshot_download repo_id=Qwen/Qwen2.5-Math-1.5B -> models/Qwen2.5-Math-1.5B
/workspace/Stanford-CS336/assignment5-alignment/.venv/lib/python3.12/site-packages/huggingface_hub/utils/_validators.py:202: UserWarning: The `local_dir_use_symlinks` argument is deprecated and ignored in `snapshot_download`. Downloading to a local directory does not use symlinks anymore.
  warnings.warn(
Downloading (incomplete total...): 0.00B [00:00, ?B/s]                                                             Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Fetching 10 files: 100%|████████████████████████████████████████████████████████████| 10/10 [01:22<00:00,  8.28s/it]
Download complete: : 3.10GB [01:22, 45.2MB/s]              [done] snapshot_download complete.01:22<00:37, 12.34s/it]
Download complete: : 3.10GB [01:23, 37.2MB/s]


[verify] FAILED: /workspace/Stanford-CS336/assignment5-alignment/.venv/lib/python3.12/site-packages/torch/lib/../../nvidia/cusparse/lib/libcusparse.so.12: undefined symbol: __nvJitLinkComplete_12_4, version libnvJitLink.so.12
(alignment) ➜  assignment5-alignment git:(main) ✗ 
```


```
def verify_local_load(save_dir: Path):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[verify] Loading back from {save_dir}")
        _tok = AutoTokenizer.from_pretrained(save_dir, local_files_only=True)
        _model = AutoModelForCausalLM.from_pretrained(save_dir, local_files_only=True)
        print("[verify] OK: model and tokenizer load locally.")
    except Exception as e:
        print("[verify] FAILED:", e)
        sys.exit(2)
```

```
(alignment) ➜  assignment5-alignment git:(main) ✗ python download_model.py \
  --repo-id Qwen/Qwen2.5-Math-1.5B \
  --save-dir models/Qwen2.5-Math-1.5B \
  --method snapshot --no-symlinks --verify
[download] snapshot_download repo_id=Qwen/Qwen2.5-Math-1.5B -> models/Qwen2.5-Math-1.5B
/workspace/Stanford-CS336/assignment5-alignment/.venv/lib/python3.12/site-packages/huggingface_hub/utils/_validators.py:202: UserWarning: The `local_dir_use_symlinks` argument is deprecated and ignored in `snapshot_download`. Downloading to a local directory does not use symlinks anymore.
  warnings.warn(
Downloading (incomplete total...): 0.00B [00:00, ?B/s]                                                             Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Fetching 10 files: 100%|████████████████████████████████████████████████████████████| 10/10 [01:22<00:00,  8.28s/it]
Download complete: : 3.10GB [01:22, 45.2MB/s]              [done] snapshot_download complete.01:22<00:37, 12.34s/it]
Download complete: : 3.10GB [01:23, 37.2MB/s]
[verify] FAILED: /workspace/Stanford-CS336/assignment5-alignment/.venv/lib/python3.12/site-packages/torch/lib/../../nvidia/cusparse/lib/libcusparse.so.12: undefined symbol: __nvJitLinkComplete_12_4, version libnvJitLink.so.12
```


```
(alignment) ➜  assignment5-alignment git:(main) ✗ python -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.version.cuda}')"
python --version
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/workspace/Stanford-CS336/assignment5-alignment/.venv/lib/python3.12/site-packages/torch/__init__.py", line 367, in <module>
    from torch._C import *  # noqa: F403
    ^^^^^^^^^^^^^^^^^^^^^^
ImportError: /workspace/Stanford-CS336/assignment5-alignment/.venv/lib/python3.12/site-packages/torch/lib/../../nvidia/cusparse/lib/libcusparse.so.12: undefined symbol: __nvJitLinkComplete_12_4, version libnvJitLink.so.12
Python 3.12.12
(alignment) ➜  assignment5-alignment git:(main) ✗ 
```

### 错误分析

**核心错误：**
`ImportError: .../libcusparse.so.12: undefined symbol: __nvJitLinkComplete_12_4, version libnvJitLink.so.12`

**原因诊断：**
这是一个典型的 **PyTorch CUDA 动态库冲突** 问题。
1.  **环境冲突**：你的虚拟环境（`.venv`）中安装的 `torch` 包自带了一套 CUDA 运行时库（通常在 `site-packages/nvidia/` 下）。但是，你的系统环境变量（很可能是 `LD_LIBRARY_PATH`）中包含了系统级的 CUDA 库路径。
2.  **符号不匹配**：当 Python 导入 `torch` 时，动态链接器加载了 `torch` 自带的 `libcusparse.so.12`，但该库试图链接 `libnvJitLink.so.12` 时，错误地加载了**系统版本**的 `libnvJitLink`，而不是 `torch`  bundled 的版本。这两个版本不一致（例如一个是 12.4，另一个是 12.1 或其他），导致找不到符号 `__nvJitLinkComplete_12_4`。
3.  **影响**：这导致 `import torch` 失败，进而导致 `download_model.py` 中的验证步骤（`verify_local_load`）失败。**注意：模型下载本身已经成功（3.10GB 已下载），只是本地加载验证失败了。**

---

### 修复方案

由于这一步主要是**下载和验证模型文件**，并不一定需要 GPU 加速。为了最快速地解决环境问题并完成任务，我建议按以下顺序操作。

#### 方案一：临时规避（最快，无需重装）
尝试在运行命令前清除 `LD_LIBRARY_PATH`，强制 Python 使用 `torch` 自带的库。

```bash
# 临时 unset 该环境变量
unset LD_LIBRARY_PATH

# 再次运行你的下载/验证脚本
python download_model.py \
  --repo-id Qwen/Qwen2.5-Math-1.5B \
  --save-dir models/Qwen2.5-Math-1.5B \
  --method snapshot --no-symlinks --verify
```
*如果这条命令成功运行，说明问题确系环境变量污染。后续运行其他需要 GPU 的代码时，可能需要重新 export 该变量。*

#### 方案二：重装 CPU 版 PyTorch（最稳定，推荐用于此步骤）
如果方案一无效，或者你希望彻底避免 CUDA 库冲突（此步骤不需要 GPU），可以安装 CPU 版本的 PyTorch。

```bash
# 1. 卸载现有的 torch 及相关包
pip uninstall -y torch torchvision torchaudio

# 2. 安装 CPU 版本 (避开 CUDA 库冲突)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. 再次验证
python -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# 4. 如果上面成功，重新运行你的脚本
python download_model.py \
  --repo-id Qwen/Qwen2.5-Math-1.5B \
  --save-dir models/Qwen2.5-Math-1.5B \
  --method snapshot --no-symlinks --verify
```

#### 方案三：重装匹配系统的 CUDA 版 PyTorch（如果你后续必须用 GPU）
如果作业后续步骤**必须**使用 GPU 进行训练或推理，你需要安装与系统驱动匹配的 CUDA 版本。

1.  **查询系统 CUDA 支持版本**：
    ```bash
    nvidia-smi
    ```
    *(请查看右上角的 `CUDA Version`，例如 12.1 或 12.4)*

2.  **重装对应版本的 torch**（假设是 CUDA 12.1，请根据实际调整）：
    ```bash
    pip uninstall -y torch torchvision torchaudio
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

---

### 需要向你确认的信息

为了给出更精确的长期建议，请告诉我：

1.  **后续作业是否需要 GPU？** (如果这只是下载步骤，方案二最稳妥；如果后续要训练，方案三必要)。
2.  **`nvidia-smi` 的输出是什么？** (如果方案一失败且你需要 GPU，我需要知道你的驱动支持的 CUDA 版本)。
3.  **`echo $LD_LIBRARY_PATH` 的输出是什么？** (这能确认是否确实有系统 CUDA 路径污染)。

**建议先尝试【方案一】，如果不行则尝试【方案二】。** 模型文件已经下载好了，只要 `import torch` 能成功，验证就能通过。


# 【排查1】


```
(alignment) ➜  assignment5-alignment git:(main) ✗ ./test.sh          
[skip] models/Qwen2.5-Math-1.5B already exists. Use --force to overwrite.
(alignment) ➜  assignment5-alignment git:(main) ✗ ./test.sh          
Torch: 2.5.1+cu124, CUDA: 12.4
(alignment) ➜  assignment5-alignment git:(main) ✗ echo $LD_LIBRARY_PATH
:/usr/local/cuda-12.2/lib64:/usr/local/cuda-12.2/lib64
```


# 【修复1】

```
# 临时 unset 该环境变量
unset LD_LIBRARY_PATH

uv sync

(alignment) ➜  assignment5-alignment git:(main) ✗ ./test.sh            
Resolved 251 packages in 0.57ms
      Built flash-attn==2.7.4.post1
Prepared 1 package in 34.34s
Installed 1 package in 2ms
 + flash-attn==2.7.4.post1
```

# 【修复2】


```
(alignment) ➜  assignment5-alignment git:(main) ✗ unset LD_LIBRARY_PATH
(alignment) ➜  assignment5-alignment git:(main) ✗ python download_model.py \
  --repo-id Qwen/Qwen2.5-Math-1.5B \
  --save-dir models/Qwen2.5-Math-1.5B \
  --method snapshot --no-symlinks --verify
[verify] Loading back from models/Qwen2.5-Math-1.5B
Loading weights: 100%|███████████████████| 338/338 [00:00<00:00, 9168.17it/s, Materializing param=model.norm.weight]
[verify] OK: model and tokenizer load locally.
[success] Ready to load locally with:
  AutoTokenizer.from_pretrained('models/Qwen2.5-Math-1.5B')
  AutoModelForCausalLM.from_pretrained('models/Qwen2.5-Math-1.5B')
```

