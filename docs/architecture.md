# Architecture

## Overview

This document provides a comprehensive elboration of the key shift in **nanovllm_HARD** from **nanovllm** (base implementation), featured with **Head-Adaptive Rewrite & Dense-sparse management**. nanovllm_HARD introduces sophisticated KV cache management through head-adaptive selection, utilizing sparse cache (with packed_headwise_mask) and condensed cache (after rewrite operations).

---

## 1. Directory Structure Comparison

### 1.1 Base Implementation: `src/services/nanovllm`

```
src/services/nanovllm/
├── __init__.py
├── config.py                      # Basic configuration
├── llm.py                         # Main LLM interface (inherits from LLMEngine)
├── sampling_params.py             # Sampling parameters configuration
├── engine/
│   ├── llm_engine.py             # Core inference engine with multiprocess model runners
│   ├── scheduler.py              # Request scheduler with prefill/decode phases
│   ├── block_manager.py          # Basic block memory management
│   └── sequence.py               # Sequence data structures
├── model_runner/
│   ├── model_runner.py           # Model execution runner
│   ├── models/
│   │   └── qwen3.py              # Qwen-3 model implementation
│   └── layers/
│       ├── attention.py          # Standard attention mechanism
│       ├── linear.py             # Linear layer implementations
│       ├── layernorm.py          # Layer normalization
│       ├── rotary_embedding.py   # Rotary position embeddings
│       ├── activation.py         # Activation functions
│       ├── embed_head.py         # Embedding and head operations
│       └── sampler.py            # Token sampling logic
└── utils/
    ├── context.py                # Context management
    └── loader.py                 # Model loading utilities
```

### 1.2 HARD Implementation: `src/services/nanovllm_HARD`

```
src/services/nanovllm_HARD/
├── __init__.py
├── config.py                      # Enhanced configuration with HARD parameters
├── llm.py                         # Main LLM interface
├── sampling_params.py             # Extended sampling parameters
├── engine/
│   ├── llm_engine.py             # Enhanced engine with cache compression support
│   ├── scheduler.py              # Enhanced scheduler with block management
│   ├── sequence.py               # Extended sequence with sparse support
│   └── io_struct.py              # Input/output data structures
├── model_runner/
│   ├── model_runner.py           # Enhanced model runner
│   ├── models/
│   │   └── qwen3.py              # Qwen-3 model with HARD optimizations
│   └── layers/                   # Same layer structure but with HARD optimizations
├── utils/
│   ├── context.py                # Enhanced context management
│   ├── loader.py                 # Model loading with HARD optimizations
│   ├── socket.py                 # Socket communication utilities
│   └── logging.py                # Enhanced logging for HARD operations
└── artifacts/nanovllm_HARD/      # HARD-specific implementations
    ├── cache_mngr/               # Cache management with compression strategies
    │   ├── __init__.py
    │   ├── search.py                   # Temperature search algorithm for Logits Calibration
    │   ├── base.py                     # Base cache manager interface
    │   ├── headwise.py                 # Head-wise compression with packed_headwise_mask
    │   ├── rkv_topp_rewrite.py         # RKV with TopP selection
    │   ├── rkv_topp.py                 # RKV with TopP selection + rewrite
    │   ├── snapkv_topp_rewrite.py      # SnapKV with ToPP selection
    │   ├── snapkv_topp.py              # SnapKV ToPP selection + rewrite
    │   ├── vanilla_topp_rewrite.py     # Vanilla with TopP selection
    │   ├── vanilla_topp.py             # Vanilla with TopP selection + rewrite
    │   └── no_compress.py              # Baseline no-compression (for logging use)
    ├── block_mngr/               # Block management for sparse operations
    │   ├── __init__.py
    │   ├── base.py               # Base block manager interface
    │   └── headwise_block_manager.py  # Head-level granularity block manager
    └── attention/                # Optimized attention mechanisms
        ├── __init__.py
        └── hard_attention.py     # HARD-specific attention implementation
```

---

## 2. Architectural Designs

### 2.1 nanovllm: Monolithic Dense Cache Architecture

**Design Principles:**
- **Simplicity**: Single, uniform cache representation for all attention heads
- **Dense Allocation**: Contiguous memory blocks with fixed block size (256 tokens)
- **Layer-Level Management**: Cache operations performed at the layer granularity
- **Standard Attention**: Traditional FlashAttention-style computation

**Key Architectural Components:**

```
┌───────────────────────────────────────────────────────────┐
│                         LLMEngine                         │
│  ┌──────────────────────────────────────────────────────┐ │
│  │              Scheduler (Prefill/Decode)              │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  │ │
│  │  │   Block     │  │  Sequence   │  │   Request    │  │ │
│  │  │  Manager    │  │  Manager    │  │   Queue      │  │ │
│  │  └──────┬──────┘  └──────┬──────┘  └──────────────┘  │ │
│  └─────────┼────────────────┼───────────────────────────┘ │
└────────────┼────────────────┼─────────────────────────────┘
             │                │
             ▼                ▼
    ┌─────────────────┐  ┌─────────────────┐
    │  Model Runner   │  │  KV Cache       │
    │                 │  │  (Dense Blocks) │
    │                 │  │                 │
    └─────────────────┘  └─────────────────┘
             │
             ▼
    ┌──────────────────────────────────────────────┐
    │         Attention Layer (Dense)              │
    │  All heads process same cached data          │
    └──────────────────────────────────────────────┘
```

### 2.2 Hierarchical Sparse-Dense Architecture

**Core featuress:**
- **Head-Adaptive Selection**: Different heads can use different sparse selection results 
- **Cascade Cache System**: Sparse cache + Condensed cache + Dense Cache (not specifically handled) working in tandem
- **Compression Flexibility**: Multiple compression strategies (RKV, SnapKV, ToPP), normalized by temperature search algorithm. 

**Key Architectural Components:**

```
┌───────────────────────────────────────────────────────────────┐
│                    HARD LLMEngine                             │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │              Enhanced Scheduler                          │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │ │
│  │  │ Headwise     │  │ Cache        │  │ Request      │    │ │
│  │  │ Block        │  │ Manager      │  │ Queue        │    │ │
│  │  │ Manager      │  │ (Multiple    │  │              │    │ │
│  │  │              │  │  Strategies) │  │              │    │ │
│  │  └──────┬───────┘  └──────┬───────┘  └──────────────┘    │ │
│  └─────────┼─────────────────┼──────────────────────────────┘ │
└────────────┼─────────────────┼────────────────────────────────┘
             │                 │
             ▼                 ▼
    ┌─────────────────┐  ┌─────────────────────────────────┐
    │  Model Runner   │  │     HARD KV Cache System        │
    │                 │  │  ┌─────────────┐  ┌───────────┐ │
    │                 │  │  │ Sparse      │  │ Condensed │ │
    │                 │  │  │ Cache       │  │ Cache     │ │
    │                 │  │  │ (Headwise   │  │ (Rewrite) │ │
    │                 │  │  │ Mask)       │  │           │ │
    └─────────────────┘  │  └─────────────┘  └───────────┘ │
                         │  ┌────────────────────────────┐ │
                         │  │ packed_headwise_mask       │ │
                         │  │ (uint8 tensor)             │ │
                         │  └────────────────────────────┘ │
                         └─────────────────────────────────┘
             │
             ▼
    ┌────────────────────────────────────────────────────────┐
    │         HARD Attention Layer & organize()              │
    │  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐   │
    │  │ Query Window │  │ Head-wise    │  │ Rewrite     │   │
    │  │ (128 tokens) │  │ Selection    │  │ Operation   │   │
    │  │              │  │              │  │             │   │
    │  └──────────────┘  └──────────────┘  └─────────────┘   │
    └────────────────────────────────────────────────────────┘
```

---

####  Multi-Strategy Cache Manager

```
┌─────────────────────────────────────────────────────────────────┐
│              HARD Cache Manager Factory                         │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────┐   │
│  │  Headwise  │  │  Vanilla   │  │    RKV     │  │  SnapKV  │   │
│  │ Compressor │  │ Compressor │  │ Compressor │  │Compressor│   │
│  └─────┬──────┘  └────────────┘  └────────────┘  └──────────┘   │
│        │                                                        │
│        ▼                                                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │         Dual Cache System                               │    │
│  │  ┌─────────────────┐  ┌─────────────────────────────┐   │    │
│  │  │ Sparse Cache    │  │ Condensed Cache             │   │    │
│  │  │ (Headwise Mask) │  │ (After Rewrite Operation)   │   │    │
│  │  │                 │  │                             │   │    │
│  │  │ - Active heads  │  │ - Top-P selected tokens     │   │    │
│  │  │ - Packed mask   │  │ - Compressed representation │   │    │
│  │  └─────────────────┘  └─────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

For Sparse Cache and Condensed Cache, we use a unify uint8 packed mask: 

`seq.headwise_mask_layer` shped `[num_layers, num_kv_heads, (seq_len + 7) // 8]`.

that works for the kernel-level implemention. 

The rewrite kernel will collect `slot_mappings`(location in the global KV pool) that is read from `seq.block_table` or the seq-to-slot tensor (`seq_to_slot_pool`) (only work for sparse-only occasion, when Cache Manager will maitain a compressed indices for effective locations in the full block-table). 


## 3 Data Structure Evolution

### Sequence Data Structure

```python
# nanovllm_HARD: Enhanced sequence with sparse support
class HARDSequence:
    seq_id: int
    prompt_token_ids: List[int]
    output_token_ids: List[int]
    block_table: Dict[int, Dict[int, List[int]]]  # layer->head->blocks
    headwise_mask_layer: List[torch.Tensor]  # Per-layer head masks
    num_blocks_head: Dict[int, Dict[int, int]]  # Block counters
    query_block_id: Optional[int]  # Recent context
    query_len: int  # Query window size (default: 128)
```

### Head-flattend Attention and Partial Update 

```python
class Attetnion:
    def _partial_update_indices_cudagraph(self,
                                        cu_packed_custom_mask: torch.Tensor,
                                        ):
        self.forward_wrapper._custom_mask_buf.copy_(cu_packed_custom_mask)
    
    def _partial_update_indices(self, 
                                cu_packed_custom_mask: torch.Tensor,
                                ):
        self.forward_wrapper._custom_mask_buf = cu_packed_custom_mask.to(
            self.forward_wrapper.device
        )

    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        layer_id: int
    ):
        # store the q_cache in the local window
        store_q_cache(q, self.q_cache, context.query_slot_mapping)
        
        # context works like one in the nanovllm_base, 
        # where global metadata can be read during cudagraph replay
        self.partial_update_indices(context.packed_headwise_mask[layer_id])

        q = q.view(
                -1,
                self.num_kv_heads,
                self.num_heads // self.num_kv_heads,
                self.head_dim,
            ).view(-1, self.num_heads // self.num_kv_heads, self.head_dim)
        
        k_cache = k_cache.view(-1, 1, self.head_dim)
        v_cache = v_cache.view(-1, 1, self.head_dim)

        o = o.view(
                -1,
                self.num_kv_heads,
                self.num_heads // self.num_kv_heads,
                self.head_dim,
            ).view(-1, self.num_heads, self.head_dim)
            .view(-1, self.num_heads * self.head_dim)

        return o
```

### Configuration Explanation

```python
# Extended configuration for HARD
class Config:
    model: str
    enforce_eager: bool = False # False mean CUDA Graph is enabled
    query_window_size: int = 128 # query local window cache size
    layer_budget: int = 1024
    layer_upper_budget: int = 2048
    
    compress_method: str = "snapkv" # in {"snapkv", "rkv", "vanilla_topp"}
    
    if_compress_kvcache: bool = False # set to False when no Cache Mangager related-functions will be triggered
    if_fake_compress: bool = False # set to False when no rewrite operation (condense cache) is needed
    if_log_compress: bool = False # logging use
    if_log_num_topp: bool = False # logging use
    
    p_attn: float = 0.90 # use in top p selection
    
    attn_reduce_method: str = "raw" # the logging postfix

    steps_between_cache_compressions: int = 128 # steps between compression operations
```


---


## 4. Data Flow Comparison

### 4.1 nanovllm: Linear Dense Flow

```
Request → Scheduler → Block Manager → Model Runner → Attention (Dense) → Output
                    │                               │
                    └── Allocate contiguous blocks ─┘
                              │
                              ▼
                    [Dense KV Cache]
                    All heads × All layers
```

### 4.2 nanovllm_HARD: Hierarchical Adaptive Flow

```
Request → Enhanced Scheduler → Headwise Block Manager → Model Runner
                              │                         │
                              ├── Allocate per-head     │
                              │   blocks                │
                              │                         ▼
                              │              ┌─────────────────────┐
                              │              │ Cache Manager       │
                              │              │ (Select Strategy)   │
                              │              └──────────┬──────────┘
                              │                         │
                              │                         ▼
                              │              ┌─────────────────────┐
                              │              │ Compress KV         │
                              │              │ ├─ Sparse (mask)    │
                              │              │ └─ Condensed (top-p)│
                              │              └──────────┬──────────┘
                              │                         │
                              ▼                         ▼
                    ┌──────────────────────────────────────────┐
                    │      HARD Attention Layer                │
                    │  ┌────────────┐  ┌─────────────────────┐ │
                    │  │ Query      │  │ Head-wise           │ │
                    │  │ Window     │  │ Selection           │ │
                    │  │ (recent)   │  │ (packed_mask)       │ │
                    │  └────────────┘  └─────────────────────┘ │
                    └───────────────────────┬──────────────────┘
                                            │
                                            ▼
                                      Output
```

---


## 5. Memory Architecture Comparison

### 5.1 nanovllm Memory Layout

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU Memory (Dense)                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │               KV Cache Pool                          │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐         │   │
│  │  │Block 0 │ │Block 1 │ │Block 2 │ │Block 3 │ ...     │   │
│  │  │256     │ │256     │ │256     │ │256     │ tokens  │   │
│  │  │tokens  │ │tokens  │ │tokens  │ │tokens  │         │   │
│  │  └────────┘ └────────┘ └────────┘ └────────┘         │   │
│  │                                                      │   │
│  │  All heads in all layers use same blocks             │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 nanovllm_HARD Memory Layout

```
┌─────────────────────────────────────────────────────────────────┐
│                   GPU Memory (Hierarchical)                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                Sparse Cache Pool                           │ │
│  │  Per-head, per-layer allocation (block_size=1)             │ │
│  │  0 for inactive, 1 for active                              │ │
│  │  Layer 0:                                                  │ │
│  │    Head 0: [Block 0, 1][Block 1, 1][Block 2, 1]    ...     │ │
│  │    Head 1: [Block 8, 1][Block 9, 0][Block 11, 0]   ...     │ │
│  │    Head 2: [Block 16, 0][Block 17, 1][Block 18, 1] ...     │ │
│  │    ...                                                     │ │
│  │                                                            │ │
│  │  Layer 1:                                                  │ │
│  │    Head 0: [Block 50][Block 51]...                         │ │
│  │    ...                                                     │ │
│  └────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                Condensed Cache Pool                        │ │
│  │  Compressed tokens after ToPP selection                    │ │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐                       │ │
│  │  │Rewritten│ │Rewritten│ │Rewritten│ ...                   │ │
│  │  │Tokens 0 │ │Tokens 1 │ │Tokens 2 │                       │ │
│  │  └─────────┘ └─────────┘ └─────────┘                       │ │
│  └────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                Query Window (Recent Context)               │ │
│  │  Always dense: 32 most recent tokens                       │ │
│  │  ┌─────────────────────────────────┐                       │ │
│  │  │ [Token-31][Token-30]...[Token-0]│                       │ │
│  │  └─────────────────────────────────┘                       │ │
│  └────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                Metadata Structures                         │ │
│  │  - headwise_mask_layer: uint8 [L, H, N // 8]               │ │
│  │  - num_blocks_head: Dict[int, Dict[int, int]]              │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Extensibility Points

### 6.1 nanovllm: "Single Purpose, Well-Defined"

#### nanovllm Extension Points

| Component | Extension Method | Complexity |
|-----------|------------------|------------|
| Attention | Modify `layers/attention.py` | Moderate |
| Block Management | Extend `block_manager.py` | Morderate |
| Cache Strategy | N/A (not designed for it) | Very High |


The directory structure reflects a straightforward, production-oriented design:
- Clear separation between engine, model_runner, and layers
- Minimal abstraction layers
- Direct implementations without plugin systems

### 6.2 nanovllm_HARD: "Research-Oriented, Plugin-Based"

#### nanovllm_HARD Extension Points

| Component | Extension Method | Complexity |
|-----------|------------------|------------|
| Cache Strategy | Add new compressor in `artifacts/cache_mngr/` | Low |
| Block Management | Extend `artifacts/block_mngr/base.py` | Moderate |
| Attention | Modify `artifacts/attention/` | Moderate |
| Compression Algorithm | Implement `update_kv` interface | Low |

**Design Philosophy Difference**: nanovllm_HARD explicitly anticipates experimentation through its artifact-based architecture, while nanovllm prioritizes simplicity.

The `artifacts/` directory encapsulates experimental components:
- `cache_mngr/`: Pluggable compression strategies
- `block_mngr/`: Specialized block managers
- `attention/`: Custom attention implementations

This separation allows:
- Rapid experimentation without modifying core engine
- A/B testing of different strategies
- Clear isolation of experimental vs. stable code

---
