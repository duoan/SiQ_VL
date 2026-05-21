"""Test cuTile DSL on Blackwell GPU."""

import math
import time

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

INV_LOG_2 = 1.0 / math.log(2)


@ct.kernel()
def fmha_kernel(
    Q, K, V, Out,
    qk_scale: float,
    TILE_D: ConstInt,
    H: ConstInt,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    CAUSAL: ConstBool,
):
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)

    batch_idx = bid_y // H
    head_idx = bid_y % H

    qk_scale = qk_scale * INV_LOG_2

    offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
    offs_m = offs_m[:, None]

    offs_n_tile = ct.arange(TILE_N, dtype=ct.int32)
    offs_n_tile = offs_n_tile[None, :]

    m_i = ct.full((TILE_M, 1), -math.inf, dtype=ct.float32)
    l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)

    q = ct.load(Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D)).reshape((TILE_M, TILE_D))

    k_seqlen = K.shape[2]
    m_end = (bid_x + 1) * TILE_M

    if CAUSAL:
        Tc = ct.cdiv(min(m_end, k_seqlen), TILE_N)
    else:
        Tc = ct.cdiv(k_seqlen, TILE_N)

    for j in range(0, Tc):
        k = ct.load(K, index=(batch_idx, head_idx, 0, j), shape=(1, 1, TILE_D, TILE_N), order=(0, 1, 3, 2)).reshape((TILE_D, TILE_N))
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q, k, qk)

        if CAUSAL:
            offs_n = j * TILE_N + offs_n_tile
            mask = offs_m >= offs_n
            mask = ct.where(mask, 0.0, -math.inf)
            qk += mask

        qk_max = ct.max(qk, axis=-1, keepdims=True)
        qk_max_scaled = qk_max * qk_scale
        m_ij = max(m_i, qk_max_scaled)

        qk = qk * qk_scale
        qk = qk - m_ij
        p = ct.exp2(qk, flush_to_zero=True)

        l_ij = ct.sum(p, axis=-1, keepdims=True)
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha

        v = ct.load(V, index=(batch_idx, head_idx, j, 0), shape=(1, 1, TILE_N, TILE_D)).reshape((TILE_N, TILE_D))
        p = p.astype(Q.dtype)
        acc = ct.mma(p, v, acc)
        m_i = m_ij

    acc = ct.truediv(acc, l_i, flush_to_zero=True)
    acc = acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)


@ct.kernel()
def fmha_kernel_optimized(
    Q, K, V, Out,
    qk_scale: float,
    TILE_D: ConstInt,
    H: ConstInt,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    CAUSAL: ConstBool,
    NUM_M_BLOCKS: ConstInt,
):
    """Optimized Flash Attention with K-loop split and ProgramId remapping."""
    bid_y = ct.bid(1)
    # ProgramId remapping: reverse order for causal (better load balancing)
    if CAUSAL:
        bid_x = NUM_M_BLOCKS - 1 - ct.bid(0)
    else:
        bid_x = ct.bid(0)

    batch_idx = bid_y // H
    head_idx = bid_y % H

    qk_scale = qk_scale * INV_LOG_2

    offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
    offs_m = offs_m[:, None]
    offs_n_tile = ct.arange(TILE_N, dtype=ct.int32)
    offs_n_tile = offs_n_tile[None, :]

    m_i = ct.full((TILE_M, 1), -math.inf, dtype=ct.float32)
    l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)

    q = ct.load(Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D)).reshape((TILE_M, TILE_D))

    k_seqlen = K.shape[2]
    m_end = (bid_x + 1) * TILE_M

    if CAUSAL:
        Tc = ct.cdiv(min(m_end, k_seqlen), TILE_N)
        # K-loop split: calculate where masking starts
        mask_start = bid_x * TILE_M // TILE_N
        mask_start = min(mask_start, k_seqlen // TILE_N)
    else:
        Tc = ct.cdiv(k_seqlen, TILE_N)
        mask_start = Tc

    # Phase 1: Unmasked iterations (no causal mask needed)
    for j in range(0, mask_start):
        k = ct.load(K, index=(batch_idx, head_idx, 0, j), shape=(1, 1, TILE_D, TILE_N), order=(0, 1, 3, 2), latency=2).reshape((TILE_D, TILE_N))
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q, k, qk)

        qk_max = ct.max(qk, axis=-1, keepdims=True)
        qk_max_scaled = qk_max * qk_scale
        m_ij = max(m_i, qk_max_scaled)
        qk = qk * qk_scale - m_ij
        p = ct.exp2(qk, flush_to_zero=True)

        l_ij = ct.sum(p, axis=-1, keepdims=True)
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha

        v = ct.load(V, index=(batch_idx, head_idx, j, 0), shape=(1, 1, TILE_N, TILE_D), latency=4).reshape((TILE_N, TILE_D))
        p = p.astype(Q.dtype)
        acc = ct.mma(p, v, acc)
        m_i = m_ij

    # Phase 2: Masked iterations (apply causal mask)
    for j in range(mask_start, Tc):
        k = ct.load(K, index=(batch_idx, head_idx, 0, j), shape=(1, 1, TILE_D, TILE_N), order=(0, 1, 3, 2), latency=2).reshape((TILE_D, TILE_N))
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q, k, qk)

        if CAUSAL:
            offs_n = j * TILE_N + offs_n_tile
            mask = offs_m >= offs_n
            mask = ct.where(mask, 0.0, -math.inf)
            qk += mask

        qk_max = ct.max(qk, axis=-1, keepdims=True)
        qk_max_scaled = qk_max * qk_scale
        m_ij = max(m_i, qk_max_scaled)
        qk = qk * qk_scale - m_ij
        p = ct.exp2(qk, flush_to_zero=True)

        l_ij = ct.sum(p, axis=-1, keepdims=True)
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha

        v = ct.load(V, index=(batch_idx, head_idx, j, 0), shape=(1, 1, TILE_N, TILE_D), latency=4).reshape((TILE_N, TILE_D))
        p = p.astype(Q.dtype)
        acc = ct.mma(p, v, acc)
        m_i = m_ij

    acc = ct.truediv(acc, l_i, flush_to_zero=True)
    acc = acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)


def cutile_flash_attention(q, k, v, is_causal=True):
    """Launch cuTile Flash Attention kernel (baseline 64x64)."""
    batch_size, num_heads, seq_len, head_dim = q.shape

    sm_scale = 1.0 / math.sqrt(head_dim)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    o = torch.empty_like(q)

    TILE_M, TILE_N = 64, 64

    grid_x = math.ceil(seq_len / TILE_M)
    grid_y = batch_size * num_heads
    grid = (grid_x, grid_y, 1)

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        fmha_kernel,
        (q, k, v, o, sm_scale, head_dim, num_heads, TILE_M, TILE_N, is_causal),
    )
    return o


def cutile_flash_attention_opt(q, k, v, is_causal=True, tile_m=64, tile_n=64):
    """Launch optimized cuTile Flash Attention (K-loop split + remapping)."""
    batch_size, num_heads, seq_len, head_dim = q.shape

    sm_scale = 1.0 / math.sqrt(head_dim)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    o = torch.empty_like(q)

    TILE_M, TILE_N = tile_m, tile_n
    grid_x = math.ceil(seq_len / TILE_M)
    grid_y = batch_size * num_heads
    grid = (grid_x, grid_y, 1)

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        fmha_kernel_optimized,
        (q, k, v, o, sm_scale, head_dim, num_heads, TILE_M, TILE_N, is_causal, grid_x),
    )
    return o


def benchmark(fn, *args, warmup=5, repeats=20, **kwargs):
    """Benchmark a function."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return sum(times) / len(times)


def main():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"cuTile: {ct.__version__}")
    print()

    # Test configs matching our model: Qwen2.5-1.5B has head_dim=64, 12 heads
    configs = [
        (4, 12, 512, 64),    # short seq
        (4, 12, 1024, 64),   # medium seq
        (4, 12, 1536, 64),   # our pack_max_length
        (4, 12, 2048, 64),   # long seq
    ]

    print("=== Correctness Check ===")
    B, H, N, D = 2, 4, 128, 64
    q = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")

    out_cutile = cutile_flash_attention(q, k, v, is_causal=True)
    out_sdpa = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

    diff = (out_cutile.float() - out_sdpa.float()).abs().max().item()
    print(f"  Max diff vs SDPA: {diff:.4f} (bf16 tolerance ~0.03)")
    assert diff < 0.1, f"Correctness check failed: {diff}"
    print("  PASSED!\n")

    print("=== Performance Benchmark ===")
    print(f"{'Config':<25} {'Baseline':<12} {'Optimized':<12} {'SDPA':<12} {'Opt/SDPA':<10}")
    print("-" * 72)

    for B, H, N, D in configs:
        q = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")

        t_base = benchmark(cutile_flash_attention, q, k, v, is_causal=True)
        t_opt = benchmark(cutile_flash_attention_opt, q, k, v, is_causal=True, tile_m=64, tile_n=64)
        t_sdpa = benchmark(
            torch.nn.functional.scaled_dot_product_attention, q, k, v, is_causal=True
        )

        # Also verify optimized version correctness
        out_opt = cutile_flash_attention_opt(q, k, v, is_causal=True)
        out_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        diff = (out_opt.float() - out_ref.float()).abs().max().item()

        config_str = f"B={B},H={H},N={N},D={D}"
        print(f"{config_str:<25} {t_base:<12.3f} {t_opt:<12.3f} {t_sdpa:<12.3f} {t_opt/t_sdpa:<10.2f}x (diff={diff:.4f})")

    # Try larger tiles for long sequences
    print("\n=== Larger Tile Sizes (N=2048) ===")
    B, H, N, D = 4, 12, 2048, 64
    q = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")

    t_sdpa = benchmark(torch.nn.functional.scaled_dot_product_attention, q, k, v, is_causal=True)
    print(f"  SDPA:       {t_sdpa:.3f} ms")
    for tm, tn in [(64, 64), (128, 64), (128, 128)]:
        try:
            t = benchmark(cutile_flash_attention_opt, q, k, v, is_causal=True, tile_m=tm, tile_n=tn)
            out = cutile_flash_attention_opt(q, k, v, is_causal=True, tile_m=tm, tile_n=tn)
            diff = (out.float() - out_ref.float()).abs().max().item()
            print(f"  cuTile {tm}x{tn}: {t:.3f} ms ({t/t_sdpa:.2f}x SDPA) diff={diff:.4f}")
        except Exception as e:
            print(f"  cuTile {tm}x{tn}: FAILED ({type(e).__name__})")

    print("\nDone!")


if __name__ == "__main__":
    main()
