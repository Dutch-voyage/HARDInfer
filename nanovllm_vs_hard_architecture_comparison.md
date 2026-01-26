# nanovllm vs nanovllm_HARD: Architecture Comparison

## Overview

This document provides a comprehensive comparison of the code structure between **nanovllm** (base implementation) and **nanovllm_HARD** (Head-Adaptive Rewrite & Dense-sparse management). nanovllm_HARD introduces sophisticated KV cache management through head-adaptive selection, utilizing sparse cache (with packed_headwise_mask) and condensed cache (after rewrite operations).

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
└── artifacts/nanovllm_HARD/       # HARD-specific implementations
    ├── cache_mngr/               # Cache management with compression strategies
    │   ├── __init__.py
    │   ├── base.py               # Base cache manager interface
    │   ├── headwise.py           # Head-wise compression with packed_headwise_mask
    │   ├── layerwise.py          # Layer-wise compression
    │   ├── rkv.py                # RKV compression strategy
    │   ├── snapkv.py             # SnapKV with ToPP selection
    │   ├── vanilla_topp.py       # Vanilla ToPP compression
    │   └── no_compress.py        # Baseline no-compression
    ├── block_mngr/               # Block management for sparse operations
    │   ├── __init__.py
    │   ├── base.py               # Base block manager interface
    │   └── headwise_block_manager.py  # Head-level granularity block manager
    └── attention/                # Optimized attention mechanisms
        ├── __init__.py
        └── hard_attention.py     # HARD-specific attention implementation
```

---

## 2. Architectural Design Philosophy

### 2.1 nanovllm: Monolithic Dense Cache Architecture

**Design Principles:**
- **Simplicity**: Single, uniform cache representation for all attention heads
- **Dense Allocation**: Contiguous memory blocks with fixed block size (256 tokens)
- **Layer-Level Management**: Cache operations performed at the layer granularity
- **Standard Attention**: Traditional FlashAttention-style computation

**Key Architectural Components:**

```
┌─────────────────────────────────────────────────────────────┐
│                         LLMEngine                           │
│  ┌───────────────────────────────────────────────────────┐ │
│  │              Scheduler (Prefill/Decode)               │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  │ │
│  │  │   Block     │  │  Sequence   │  │   Request    │  │ │
│  │  │  Manager    │  │  Manager    │  │   Queue      │  │ │
│  │  └──────┬──────┘  └──────┬──────┘  └──────────────┘  │ │
│  └─────────┼─────────────────┼───────────────────────────┘ │
└────────────┼─────────────────┼─────────────────────────────┘
             │                 │
             ▼                 ▼
    ┌─────────────────┐  ┌─────────────────┐
    │  Model Runner   │  │  KV Cache       │
    │  (Tensor        │  │  (Dense Blocks) │
    │   Parallelism)  │  │                 │
    └─────────────────┘  └─────────────────┘
             │
             ▼
    ┌──────────────────────────────────────────────┐
    │         Attention Layer (Dense)              │
    │  All heads process same cached data          │
    └──────────────────────────────────────────────┘
```

### 2.2 nanovllm_HARD: Hierarchical Sparse-Dense Architecture

**Design Principles:**
- **Head-Adaptive Selection**: Different heads can use different cache representations
- **Dual Cache System**: Sparse cache + Condensed cache working in tandem
- **Fine-Grained Management**: Block-level granularity (block_size=1) for head-level control
- **Compression Flexibility**: Multiple compression strategies (RKV, SnapKV, ToPP)

**Key Architectural Components:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    HARD LLMEngine                               │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │              Enhanced Scheduler                            │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │ │
│  │  │ Headwise     │  │ Cache        │  │ Request      │    │ │
│  │  │ Block        │  │ Manager      │  │ Queue        │    │ │
│  │  │ Manager      │  │ (Multiple    │  │              │    │ │
│  │  │              │  │  Strategies) │  │              │    │ │
│  │  └──────┬───────┘  └──────┬───────┘  └──────────────┘    │ │
│  └─────────┼─────────────────┼───────────────────────────────┘ │
└────────────┼─────────────────┼─────────────────────────────────┘
             │                 │
             ▼                 ▼
    ┌─────────────────┐  ┌─────────────────────────────────┐
    │  Model Runner   │  │     HARD KV Cache System        │
    │  (Enhanced      │  │  ┌─────────────┐  ┌───────────┐ │
    │   Tensor        │  │  │ Sparse      │  │ Condensed │ │
    │   Parallelism)  │  │  │ Cache       │  │ Cache     │ │
    │                 │  │  │ (Headwise   │  │ (Rewrite) │ │
    │                 │  │  │ Mask)       │  │           │ │
    └─────────────────┘  │  └─────────────┘  └───────────┘ │
                         │  ┌─────────────────────────────┐ │
                         │  │ packed_headwise_mask        │ │
                         │  │ (uint8 tensor)              │ │
                         │  └─────────────────────────────┘ │
                         └─────────────────────────────────┘
             │
             ▼
    ┌────────────────────────────────────────────────────────┐
    │         HARD Attention Layer                            │
    │  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │
    │  │ Query Window │  │ Head-wise    │  │ Rewrite     │  │
    │  │ (128 tokens) │  │ Selection    │  │ Operation   │  │
    │  │              │  │              │  │             │  │
    │  └──────────────┘  └──────────────┘  └─────────────┘  │
    └────────────────────────────────────────────────────────┘
```

---

## 3. Core Architectural Differences

### 3.1 Block Management Strategy

| Aspect | nanovllm | nanovllm_HARD |
|--------|----------|---------------|
| **Block Size** | 256 tokens (coarse-grained) | 1 token (fine-grained, head-level) |
| **Allocation Unit** | Sequence-level blocks | (Layer, Head)-level blocks |
| **Block Table** | Single mapping per sequence | `layer_head_to_table`: nested dict |
| **Memory Model** | Continuous dense blocks | Sparse, distributed allocation |
| **Manager Location** | `engine/block_manager.py` | `artifacts/nanovllm_HARD/block_mngr/` |

**Key Implementation Difference:**

```python
# nanovllm: Simple block table
class Sequence:
    block_ids: List[int]  # Single list of block IDs

# nanovllm_HARD: Hierarchical block table
class HARDSequence:
    block_table: Dict[int, Dict[int, List[int]]]  # layer -> head -> blocks
    headwise_mask_layer_transpose: List[torch.Tensor]  # Active heads per layer
    num_blocks_head: Dict[int, Dict[int, int]]  # layer -> head -> count
    query_block_id: Optional[int]  # Recent context block
```

### 3.2 Cache Management Architecture

#### nanovllm: Single Cache Manager

```
┌─────────────────────────────────────────────────────────┐
│              Block Manager                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │         KV Cache (Dense)                        │   │
│  │  [Layer0][Layer1]...[LayerN]                   │   │
│  │   All heads share same cache                   │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

#### nanovllm_HARD: Multi-Strategy Cache Manager

```
┌─────────────────────────────────────────────────────────────────┐
│              HARD Cache Manager Factory                         │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────┐ │
│  │  Headwise  │  │ Layerwise  │  │    RKV     │  │  SnapKV  │ │
│  │ Compressor │  │ Compressor │  │ Compressor │  │Compressor│ │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └────┬─────┘ │
│        │               │               │              │        │
│        ▼               ▼               ▼              ▼        │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │         Dual Cache System                               │  │
│  │  ┌─────────────────┐  ┌─────────────────────────────┐  │  │
│  │  │ Sparse Cache    │  │ Condensed Cache             │  │  │
│  │  │ (Headwise Mask) │  │ (After Rewrite Operation)   │  │  │
│  │  │                 │  │                             │  │  │
│  │  │ - Active heads  │  │ - Top-P selected tokens     │  │  │
│  │  │ - Packed mask   │  │ - Compressed representation │  │  │
│  │  └─────────────────┘  └─────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Cache Manager Hierarchy in `artifacts/nanovllm_HARD/cache_mngr/`:**

1. **BaseCacheManager**: Abstract interface defining:
   - `compress_kv()` - Compression interface
   - `get_statistics()` - Metrics collection

2. **Strategy Implementations**:
   - `HeadwiseCompressor`: Uses `packed_headwise_mask` for per-head selection
   - `LayerwiseCompressor`: Layer-level compression budget
   - `RKVCompressor`: Re-Keying Vector strategy
   - `SnapKVCompressor`: SnapKV with ToPP token selection
   - `VanillaToPPCompressor`: Direct Top-P probability thresholding
   - `NoCompress`: Baseline (no compression)

### 3.3 Data Structure Evolution

#### Sequence Data Structure

```python
# nanovllm: Simple sequence representation
class Sequence:
    seq_id: int
    prompt_token_ids: List[int]
    output_token_ids: List[int]
    block_ids: List[int]  # Simple block list

# nanovllm_HARD: Enhanced sequence with sparse support
class HARDSequence:
    seq_id: int
    prompt_token_ids: List[int]
    output_token_ids: List[int]
    block_table: Dict[int, Dict[int, List[int]]]  # layer->head->blocks
    headwise_mask_layer_transpose: List[torch.Tensor]  # Per-layer head masks
    num_blocks_head: Dict[int, Dict[int, int]]  # Block counters
    query_block_id: Optional[int]  # Recent context
    query_len: int  # Query window size (default: 128)
```

#### Attention Kernel Integration

```python
# nanovllm: Standard attention
class Attention:
    def forward(
        hidden_states,
        past_key_values,  # Standard KV cache
        attention_mask
    ):
        # FlashAttention with dense cache
        return flash_attn(
            q, k, v,
            past_key_values=past_key_values
        )

# nanovllm_HARD: Head-adaptive attention
class HARDAttention:
    def forward(
        hidden_states,
        past_key_values,  # Dual cache (sparse + condensed)
        packed_headwise_mask,  # uint8 tensor [num_layers, num_heads]
        query_block_id  # Recent context
    ):
        # Split processing between sparse and condensed
        sparse_output = flash_attn_sparse(
            q, k, v,
            head_mask=packed_headwise_mask
        )
        condensed_output = flash_attn_condensed(
            q, k, v,
            compressed_cache=past_key_values.condensed
        )
        return merge_outputs(sparse_output, condensed_output)
```

---

## 4. Key Design Patterns in nanovllm_HARD

### 4.1 Strategy Pattern for Cache Compression

**Location**: `artifacts/nanovllm_HARD/cache_mngr/`

The cache compression system uses the Strategy pattern to allow runtime selection of compression algorithms:

```
BaseCacheManager (Abstract)
    ├── HeadwiseCompressor (Default: uses packed_headwise_mask)
    ├── LayerwiseCompressor (Layer-level budget)
    ├── RKVCompressor (Re-Keying Vector)
    ├── SnapKVCompressor (SnapKV algorithm)
    ├── VanillaToPPCompressor (Top-P thresholding)
    └── NoCompress (Baseline)
```

**Design Benefit**: Enables easy experimentation with different compression strategies without modifying core engine logic.

### 4.2 Two-Level Block Management

**Location**: `artifacts/nanovllm_HARD/block_mngr/headwise_block_manager.py`

Unlike the monolithic block manager in nanovllm, nanovllm_HARD separates concerns:

```
BaseBlockManager (Abstract)
    └── HeadwiseBlockManager
        ├── allocate(layer_id, head_id, num_blocks)
        ├── free(layer_id, head_id, block_ids)
        └── get_allocated_blocks(layer_id, head_id)
```

**Design Benefit**: Enables head-level granularity in memory allocation, critical for sparse attention patterns.

### 4.3 Dual Cache Representation

**Sparse Cache**:
- Represented by `packed_headwise_mask`: `uint8` tensor of shape `[num_layers, num_heads]`
- Each bit/pixel indicates whether a head uses sparse or dense representation
- Directly integrated with FlashInfer's optimized kernels

**Condensed Cache**:
- Created via rewrite operations (e.g., ToPP selection)
- Stores only important tokens in contiguous memory
- Managed through `gather_selected_kv()` utility functions

**Design Benefit**: Balances memory efficiency (sparse) with computation efficiency (condensed).

---

## 5. Data Flow Comparison

### 5.1 nanovllm: Linear Dense Flow

```
Request → Scheduler → Block Manager → Model Runner → Attention (Dense) → Output
                    │                               │
                    └── Allocate contiguous blocks ─┘
                              │
                              ▼
                    [Dense KV Cache]
                    All heads × All layers
```

### 5.2 nanovllm_HARD: Hierarchical Adaptive Flow

```
Request → Enhanced Scheduler → Headwise Block Manager → Model Runner
                              │                         │
                              ├── Allocate per-head    │
                              │   blocks               │
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

## 6. Component Interaction Diagram

### 6.1 nanovllm Component Graph

```
                    ┌──────────────┐
                    │     LLM      │
                    │   (API)      │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  LLMEngine   │◄─────────────┐
                    └──────┬───────┘              │
                           │                      │
            ┌──────────────┼──────────────┐      │
            │              │              │      │
            ▼              ▼              ▼      │
     ┌────────────┐ ┌───────────┐ ┌──────────┐  │
     │ Scheduler  │ │Block Mgr  │ │ Model    │  │
     └─────┬──────┘ └─────┬─────┘ │ Runner   │  │
           │              │       └─────┬────┘  │
           │              │              │       │
           └──────────────┴──────────────┘       │
                           │                     │
                           ▼                     │
                    ┌──────────────┐             │
                    │ Attention    │─────────────┘
                    │ (Dense)      │
                    └──────────────┘
```

### 6.2 nanovllm_HARD Component Graph

```
                    ┌──────────────┐
                    │  HARD LLM    │
                    │   (API)      │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │HARD LLMEngine│◄─────────────────────┐
                    └──────┬───────┘                     │
                           │                             │
            ┌──────────────┼──────────────┐             │
            │              │              │             │
            ▼              ▼              ▼             │
     ┌───────────┐ ┌──────────────┐ ┌──────────┐      │
     │Enhanced   │ │Headwise      │ │Enhanced  │      │
     │Scheduler  │ │Block Mgr     │ │Model     │      │
     └─────┬─────┘ └──────┬───────┘ │Runner    │      │
           │              │          └─────┬────┘      │
           │              │                │            │
           │              ▼                │            │
           │     ┌──────────────────┐      │            │
           │     │Cache Manager     │      │            │
           │     │Factory           │      │            │
           │     └────────┬─────────┘      │            │
           │              │                │            │
           │    ┌─────────┴─────────┐      │            │
           │    ▼                   ▼      │            │
           │ ┌─────────┐       ┌─────────┐ │            │
           │ │Sparse   │       │Condensed│ │            │
           │ │Cache    │       │Cache    │ │            │
           │ │Manager  │       │Manager  │ │            │
           │ └────┬────┘       └────┬────┘ │            │
           └──────┼────────────────┼──────┘            │
                  │                │                   │
                  └────────────────┴───────────────────┘
                                    │
                                    ▼
                          ┌──────────────────┐
                          │HARD Attention    │
                          │├─ Query Window   │
                          │├─ Head Selection │
                          │└─ Rewrite Ops    │
                          └──────────────────┘
```

---

## 7. Memory Architecture Comparison

### 7.1 nanovllm Memory Layout

```
┌──────────────────────────────────────────────────────────────┐
│                    GPU Memory (Dense)                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │               KV Cache Pool                           │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐        │   │
│  │  │Block 0 │ │Block 1 │ │Block 2 │ │Block 3 │ ...   │   │
│  │  │256     │ │256     │ │256     │ │256     │ tokens│   │
│  │  │tokens  │ │tokens  │ │tokens  │ │tokens  │       │   │
│  │  └────────┘ └────────┘ └────────┘ └────────┘        │   │
│  │                                                        │   │
│  │  All heads in all layers use same blocks             │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

### 7.2 nanovllm_HARD Memory Layout

```
┌──────────────────────────────────────────────────────────────────┐
│                   GPU Memory (Hierarchical)                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                Sparse Cache Pool                            │ │
│  │  Per-head, per-layer allocation (block_size=1)             │ │
│  │                                                             │ │
│  │  Layer 0:                                                  │ │
│  │    Head 0: [Block 0][Block 1][Block 2]...                 │ │
│  │    Head 1: [Block 10][Block 11]...                        │ │
│  │    Head 2: [None] (inactive)                             │ │
│  │    ...                                                    │ │
│  │                                                             │ │
│  │  Layer 1:                                                  │ │
│  │    Head 0: [Block 50][Block 51]...                        │ │
│  │    ...                                                    │ │
│  └────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                Condensed Cache Pool                        │ │
│  │  Compressed tokens after ToPP selection                   │ │
│  │  ┌────────┐ ┌────────┐ ┌────────┐                         │ │
│  │  │Rewritten│ │Rewritten│ │Rewritten│ ...                  │ │
│  │  │Tokens 0 │ │Tokens 1 │ │Tokens 2 │                       │ │
│  │  └────────┘ └────────┘ └────────┘                         │ │
│  └────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                Query Window (Recent Context)               │ │
│  │  Always dense: 128 most recent tokens                     │ │
│  │  ┌──────────────────────────────────┐                     │ │
│  │  │ [Token-128][Token-127]...[Token-1]│                    │ │
│  │  └──────────────────────────────────┘                     │ │
│  └────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                Metadata Structures                         │ │
│  │  - packed_headwise_mask: uint8 [L, H]                     │ │
│  │  - layer_head_to_table: Dict[int, Dict[int, List[int]]]  │ │
│  │  - num_blocks_head: Dict[int, Dict[int, int]]            │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## 8. Attention Computation Flow

### 8.1 nanovllm: Uniform Attention

```python
# Pseudo-code for nanovllm attention
def compute_attention(query, past_kv_cache, attention_mask):
    """
    All heads process the same KV cache
    """
    for layer in layers:
        for head in heads:
            # Same cache for all heads
            output[layer][head] = flash_attention(
                query[layer][head],
                past_kv_cache[layer],  # Dense cache
                attention_mask
            )
    return output
```

### 8.2 nanovllm_HARD: Adaptive Attention

```python
# Pseudo-code for nanovllm_HARD attention
def compute_hard_attention(query, past_kv_cache, packed_headwise_mask, query_block_id):
    """
    Heads adaptively use sparse or condensed cache based on mask
    """
    for layer in layers:
        for head in heads:
            is_sparse = packed_headwise_mask[layer][head]

            if is_sparse:
                # Path 1: Sparse cache (head-adaptive)
                kv_sparse = get_sparse_cache(layer, head, past_kv_cache)
                output_sparse = flash_attention_sparse(
                    query[layer][head],
                    kv_sparse,
                    head_mask=packed_headwise_mask[layer]
                )
            else:
                # Path 2: Condensed cache (after rewrite)
                kv_condensed = get_condensed_cache(layer, head, past_kv_cache)
                output_condensed = flash_attention_condensed(
                    query[layer][head],
                    kv_condensed
                )

            # Path 3: Query window (always dense)
            kv_recent = get_query_window_cache(query_block_id)
            output_recent = flash_attention(
                query[layer][head],
                kv_recent
            )

            # Merge outputs
            output[layer][head] = merge(
                output_sparse if is_sparse else output_condensed,
                output_recent
            )

    return output
```

---

## 9. Configuration System Comparison

### 9.1 nanovllm Configuration

```python
# Key configuration parameters
@dataclass
class CacheConfig:
    block_size: int = 256  # Coarse-grained
    num_gpu_blocks: int
    num_cpu_blocks: int = 0
```

### 9.2 nanovllm_HARD Configuration

```python
# Extended configuration for HARD
@dataclass
class HARDConfig:
    # Block management
    block_size: int = 1  # Fine-grained (head-level)
    num_gpu_blocks: int

    # Cache compression
    cache_compress_strategy: str = "headwise"  # or "layerwise", "rkv", "snapkv", etc.
    compress_ratio: float = 0.5

    # Query window
    query_len: int = 128  # Recent context window

    # Head-adaptive settings
    use_head_adaptive: bool = True
    head_selection_threshold: float = 0.1

    # Rewrite operations
    enable_rewrite: bool = True
    topp_threshold: float = 0.9
```

---

## 10. Extensibility Points

### 10.1 nanovllm Extension Points

| Component | Extension Method | Complexity |
|-----------|------------------|------------|
| Attention | Modify `layers/attention.py` | Moderate |
| Block Management | Extend `block_manager.py` | High |
| Cache Strategy | N/A (not designed for it) | Very High |
| Model Integration | Add new model in `models/` | Low |

### 10.2 nanovllm_HARD Extension Points

| Component | Extension Method | Complexity |
|-----------|------------------|------------|
| Cache Strategy | Add new compressor in `artifacts/cache_mngr/` | Low |
| Block Management | Extend `artifacts/block_mngr/base.py` | Low |
| Attention | Modify `artifacts/attention/` | Moderate |
| Model Integration | Add new model in `models/` | Low |
| Compression Algorithm | Implement `BaseCacheManager` interface | Low |

**Design Philosophy Difference**: nanovllm_HARD explicitly anticipates experimentation through its artifact-based architecture, while nanovllm prioritizes simplicity.

---

## 11. Summary: Key Architectural Shifts

### From nanovllm to nanovllm_HARD

| Dimension | nanovllm | nanovllm_HARD | Rationale |
|-----------|----------|---------------|-----------|
| **Granularity** | Layer-level | Head-level | Exploit head heterogeneity |
| **Cache Model** | Single dense cache | Dual cache (sparse + condensed) | Balance memory and compute |
| **Block Size** | 256 tokens | 1 token | Enable head-level allocation |
| **Compression** | None | Multiple strategies | Flexibility for research |
| **Attention** | Uniform | Head-adaptive | Quality-efficiency tradeoff |
| **Memory Layout** | Contiguous | Sparse + query window | Reduce memory footprint |
| **Extensibility** | Monolithic | Plugin-based (artifacts/) | Rapid experimentation |

### Design Trade-offs

**nanovllm Advantages:**
- Simpler codebase, easier to understand
- Lower computational overhead
- Standard attention patterns
- Faster for small models/batches

**nanovllm_HARD Advantages:**
- Significant memory reduction (through sparsity)
- Head-adaptive quality control
- Flexible compression strategies
- Better for large-batch, long-context scenarios
- Designed for research experimentation

### When to Use Each

- **nanovllm**: Production deployments with standard models, limited memory pressure, simple requirements
- **nanovllm_HARD**: Research experimentation, memory-constrained environments, large-batch inference, long-context scenarios

---

## 12. Code Organization Philosophy

### nanovllm: "Single Purpose, Well-Defined"

The directory structure reflects a straightforward, production-oriented design:
- Clear separation between engine, model_runner, and layers
- Minimal abstraction layers
- Direct implementations without plugin systems

### nanovllm_HARD: "Research-Oriented, Plugin-Based"

The `artifacts/` directory encapsulates experimental components:
- `cache_mngr/`: Pluggable compression strategies
- `block_mngr/`: Specialized block managers
- `attention/`: Custom attention implementations

This separation allows:
- Rapid experimentation without modifying core engine
- A/B testing of different strategies
- Clear isolation of experimental vs. stable code

---

## Conclusion

The transition from nanovllm to nanovllm_HARD represents a fundamental architectural shift from **uniform dense cache management** to **hierarchical, head-adaptive sparse-dense cache management**. This is achieved through:

1. **Fine-grained block management** (block_size=1 for head-level control)
2. **Dual cache system** (sparse via packed_headwise_mask + condensed via rewrite)
3. **Plugin-based compression strategies** (extensible through artifacts/)
4. **Query window optimization** (recent context always dense)
5. **Enhanced data structures** (layer_head_to_table, headwise_mask_layer_transpose)

This architecture enables significant memory savings while maintaining inference quality through intelligent head selection and cache compression strategies.
