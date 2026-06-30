# TFT GPU Environment Audit

- Python: 3.11.9 (tags/v3.11.9:de54cf5, Apr  2 2024, 10:12:12) [MSC v.1938 64 bit (AMD64)]
- PyTorch: 2.6.0+cu124
- Lightning: 2.6.1
- PyTorch Forecasting: 1.7.0
- CUDA available: True
- GPU model: NVIDIA GeForce RTX 4060
- NVIDIA driver (nvidia-smi): NVIDIA GeForce RTX 4060, 610.62, 8188 MiB, 6433 MiB
- CUDA runtime: 12.4
- cuDNN: 90100
- Total GPU memory (GB): 7.996
- Free GPU memory (GB): 6.939
- Allocated GPU memory (GB): 0.0
- Reserved GPU memory (GB): 0.0
- Mixed precision bf16 supported: True
- CUDA tensor test: PASS

## Printed diagnostics

```
torch.cuda.is_available() -> True
torch.cuda.get_device_name(0) -> NVIDIA GeForce RTX 4060
allocated GPU memory (GB) -> 0.133 (after 4096x4096 matmul test)
reserved GPU memory (GB) -> 0.145 (after 4096x4096 matmul test)
mixed precision supported -> bf16: True, fp16: True
```

## Audit verdict

GPU audit **passed**. CUDA tensor operations complete successfully after driver upgrade to 610.62.
