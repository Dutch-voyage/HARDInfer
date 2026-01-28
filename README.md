<div align="center">
  <img src="assets/icon.png" alt="HARD-KV Icon" width="150">
</div>

# HARD-KV: Head-Adaptive Regularization for Decoding-time KV Compression

This is the code repo for implementing paper **HARD-KV: Head-Adaptive Regularization for Decoding-time KV Compression**, featured by 

- Cascade Cache: a combined structure of 3 tiers of cache (Condense, Sparse and Dense), supporting **Cautious Update** for different KV selection methods.

- Head-wise Block Layouts: supporting block-wise allocation and sparsification of KV cache. More flexible and more accurate. 

- Full Inference Engien Integration: supporting engine-level features continuous batching and CUDA Graph, bringing KV cache in to system-level design. 

For further details, please refer to [architecture.md](architecture.md)

![framework](assets/frame_all.drawio.png)
## Quick Start 

We recommand to set up environments with uv. 

First run 
```
uv sync
```

Then follow [patch.md](patch.md) for applying a critical fix for flashinfer regarding `packed_custom_mask` and `mask_indptr` interfaces. 

Finally run with 
```
uv run python -m eval.test
```

Note: Remember to provide you envrionments in .env like
```
MODEL_PATH=(path to your model weights in hf format)
DATA_PATH=(path to you datasets)
```

and run `export UV_ENV_FILE=.env`

## Full Experiments

We provide simple scripts in ./scripts. The results are main produced by `./eval/baseline` (nanovllm_base) for naive top-k allocation and by `./eval/condense` (nanovllm_HARD) for our implementations. 

Feel free to try with different models and methods, or even contribute custome version of KV selections methods or model supports. 