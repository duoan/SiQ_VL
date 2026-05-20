"""
Custom Flash Attention kernel for Blackwell (sm_120) using Triton.

Implements FlashAttention-2 algorithm with:
- Tiled computation: never materializes full S×S attention matrix
- Online softmax: running max/sum for numerical stability
- Causal masking: built into the kernel
- Non-power-of-2 head_dim support (pads internally)
- Variable-length support for sample packing via cu_seqlens
"""

import math

import torch
import triton
import triton.language as tl


def _next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()


@triton.jit
def _flash_attn_fwd(
    Q, K, V, Out, L,
    stride_qz, stride_qm, stride_qk,
    stride_kz, stride_kn, stride_kk,
    stride_vz, stride_vn, stride_vk,
    stride_oz, stride_om, stride_ok,
    stride_lz,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SM_SCALE: tl.constexpr,
):
    """
    Flash attention forward. Grid: (num_m_blocks, B*H).
    Q/K/V/Out are indexed as [z, seq, dim] where z = batch*heads (flattened).
    """
    start_m = tl.program_id(0)
    off_z = tl.program_id(1)

    Q += off_z * stride_qz
    K += off_z * stride_kz
    V += off_z * stride_vz
    Out += off_z * stride_oz
    L += off_z * stride_lz

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    # Load Q tile (BLOCK_M, HEAD_DIM)
    q = tl.load(
        Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
        mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < N_CTX * 0 + HEAD_DIM),
        other=0.0,
    )

    # Online softmax accumulators
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Upper bound for KV iteration
    hi = tl.minimum(N_CTX, (start_m + 1) * BLOCK_M) if IS_CAUSAL else N_CTX

    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        curr_n = start_n + offs_n

        # Load K tile (BLOCK_N, HEAD_DIM)
        k = tl.load(
            K + curr_n[:, None] * stride_kn + offs_d[None, :] * stride_kk,
            mask=(curr_n[:, None] < N_CTX) & (offs_d[None, :] < N_CTX * 0 + HEAD_DIM),
            other=0.0,
        )

        # QK^T (BLOCK_M, BLOCK_N)
        qk = tl.dot(q, tl.trans(k)) * SM_SCALE

        # Masks
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= curr_n[None, :], qk, float("-inf"))
        qk = tl.where(curr_n[None, :] < N_CTX, qk, float("-inf"))

        # Online softmax
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        # Load V and accumulate
        v = tl.load(
            V + curr_n[:, None] * stride_vn + offs_d[None, :] * stride_vk,
            mask=(curr_n[:, None] < N_CTX) & (offs_d[None, :] < N_CTX * 0 + HEAD_DIM),
            other=0.0,
        )
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # Normalize
    acc = acc / l_i[:, None]

    # Store LSE
    lse = m_i + tl.log(l_i)
    tl.store(L + offs_m, lse, mask=offs_m < N_CTX)

    # Store output
    tl.store(
        Out + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok,
        acc.to(Out.dtype.element_ty),
        mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < N_CTX * 0 + HEAD_DIM),
    )


@triton.jit
def _flash_attn_bwd_preprocess(
    Out, DO, Delta,
    stride_oz, stride_om, stride_ok,
    stride_dz,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    off_z = tl.program_id(1)
    start_m = tl.program_id(0)

    Out += off_z * stride_oz
    DO += off_z * stride_oz
    Delta += off_z * stride_dz

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    o = tl.load(
        Out + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok,
        mask=(offs_m[:, None] < N_CTX),
        other=0.0,
    ).to(tl.float32)
    do = tl.load(
        DO + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok,
        mask=(offs_m[:, None] < N_CTX),
        other=0.0,
    ).to(tl.float32)

    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + offs_m, delta, mask=offs_m < N_CTX)


@triton.jit
def _flash_attn_bwd(
    Q, K, V, DO, DQ, DK, DV, L, Delta,
    stride_qz, stride_qm, stride_qk,
    stride_kz, stride_kn, stride_kk,
    stride_vz, stride_vn, stride_vk,
    stride_dqz, stride_dqm, stride_dqk,
    stride_lz, stride_dz,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SM_SCALE: tl.constexpr,
):
    start_n = tl.program_id(0)
    off_z = tl.program_id(1)

    Q += off_z * stride_qz
    K += off_z * stride_kz
    V += off_z * stride_vz
    DO += off_z * stride_qz  # same layout
    DQ += off_z * stride_dqz
    DK += off_z * stride_kz
    DV += off_z * stride_vz
    L += off_z * stride_lz
    Delta += off_z * stride_dz

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_m = tl.arange(0, BLOCK_M)

    # Load K, V for this column block
    k = tl.load(
        K + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk,
        mask=(offs_n[:, None] < N_CTX),
        other=0.0,
    )
    v = tl.load(
        V + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk,
        mask=(offs_n[:, None] < N_CTX),
        other=0.0,
    )

    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    lo = start_n * BLOCK_N if IS_CAUSAL else 0

    for start_m in range(lo, N_CTX, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        curr_m = start_m + offs_m

        q = tl.load(Q + curr_m[:, None] * stride_qm + offs_d[None, :] * stride_qk, mask=curr_m[:, None] < N_CTX, other=0.0)
        do = tl.load(DO + curr_m[:, None] * stride_qm + offs_d[None, :] * stride_qk, mask=curr_m[:, None] < N_CTX, other=0.0)
        lse = tl.load(L + curr_m, mask=curr_m < N_CTX, other=0.0)
        delta = tl.load(Delta + curr_m, mask=curr_m < N_CTX, other=0.0)

        # Recompute P
        qk = tl.dot(q, tl.trans(k)) * SM_SCALE
        if IS_CAUSAL:
            qk = tl.where(curr_m[:, None] >= offs_n[None, :], qk, float("-inf"))
        qk = tl.where(offs_n[None, :] < N_CTX, qk, float("-inf"))
        p = tl.exp(qk - lse[:, None])

        # dV += P^T @ dO
        dv += tl.dot(tl.trans(p.to(do.dtype)), do)

        # dS = P * (dO @ V^T - Delta)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None]) * SM_SCALE

        # dQ (atomic add across column blocks)
        dq_contrib = tl.dot(ds.to(k.dtype), k)
        tl.atomic_add(DQ + curr_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqk, dq_contrib, mask=curr_m[:, None] < N_CTX)

        # dK
        dk += tl.dot(tl.trans(ds.to(q.dtype)), q)

    tl.store(DK + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk, dk.to(DK.dtype.element_ty), mask=offs_n[:, None] < N_CTX)
    tl.store(DV + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk, dv.to(DV.dtype.element_ty), mask=offs_n[:, None] < N_CTX)


class FlashAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal=True, sm_scale=None):
        """q, k, v: (B, H, N, D)"""
        B, H, N, D = q.shape
        if sm_scale is None:
            sm_scale = D ** -0.5

        # Pad head_dim to next power of 2 for Triton
        D_padded = _next_power_of_2(D)
        if D_padded != D:
            q = torch.nn.functional.pad(q, (0, D_padded - D))
            k = torch.nn.functional.pad(k, (0, D_padded - D))
            v = torch.nn.functional.pad(v, (0, D_padded - D))

        # Reshape to (B*H, N, D_padded) for simpler kernel indexing
        q_flat = q.reshape(B * H, N, D_padded)
        k_flat = k.reshape(B * H, N, D_padded)
        v_flat = v.reshape(B * H, N, D_padded)

        out_flat = torch.empty_like(q_flat)
        L = torch.empty(B * H, N, device=q.device, dtype=torch.float32)

        BLOCK_M = 128 if D_padded <= 64 else 64
        BLOCK_N = 64 if D_padded <= 64 else 32
        grid = (triton.cdiv(N, BLOCK_M), B * H)

        _flash_attn_fwd[grid](
            q_flat, k_flat, v_flat, out_flat, L,
            q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
            k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
            v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
            out_flat.stride(0), out_flat.stride(1), out_flat.stride(2),
            L.stride(0),
            N_CTX=N,
            HEAD_DIM=D_padded,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            IS_CAUSAL=causal,
            SM_SCALE=sm_scale,
        )

        out = out_flat.reshape(B, H, N, D_padded)
        if D_padded != D:
            out = out[:, :, :, :D]

        ctx.save_for_backward(q_flat, k_flat, v_flat, out_flat, L)
        ctx.causal = causal
        ctx.sm_scale = sm_scale
        ctx.shape = (B, H, N, D, D_padded)
        ctx.BLOCK_M = BLOCK_M
        ctx.BLOCK_N = BLOCK_N
        return out

    @staticmethod
    def backward(ctx, do):
        q_flat, k_flat, v_flat, out_flat, L = ctx.saved_tensors
        B, H, N, D, D_padded = ctx.shape
        causal = ctx.causal
        sm_scale = ctx.sm_scale
        BLOCK_M = ctx.BLOCK_M
        BLOCK_N = ctx.BLOCK_N

        # Pad do if needed
        if D_padded != D:
            do = torch.nn.functional.pad(do, (0, D_padded - D))
        do_flat = do.reshape(B * H, N, D_padded).contiguous()

        # Precompute Delta
        Delta = torch.empty(B * H, N, device=do.device, dtype=torch.float32)
        grid_pre = (triton.cdiv(N, BLOCK_M), B * H)
        _flash_attn_bwd_preprocess[grid_pre](
            out_flat, do_flat, Delta,
            out_flat.stride(0), out_flat.stride(1), out_flat.stride(2),
            Delta.stride(0),
            N_CTX=N, HEAD_DIM=D_padded, BLOCK_M=BLOCK_M,
        )

        dq_flat = torch.zeros_like(q_flat)
        dk_flat = torch.empty_like(k_flat)
        dv_flat = torch.empty_like(v_flat)

        grid_bwd = (triton.cdiv(N, BLOCK_N), B * H)
        _flash_attn_bwd[grid_bwd](
            q_flat, k_flat, v_flat, do_flat, dq_flat, dk_flat, dv_flat, L, Delta,
            q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
            k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
            v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
            dq_flat.stride(0), dq_flat.stride(1), dq_flat.stride(2),
            L.stride(0), Delta.stride(0),
            N_CTX=N, HEAD_DIM=D_padded, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            IS_CAUSAL=causal, SM_SCALE=sm_scale,
        )

        dq = dq_flat.reshape(B, H, N, D_padded)[:, :, :, :D]
        dk = dk_flat.reshape(B, H, N, D_padded)[:, :, :, :D]
        dv = dv_flat.reshape(B, H, N, D_padded)[:, :, :, :D]
        return dq, dk, dv, None, None


def flash_attention(q, k, v, causal=True, sm_scale=None):
    """
    Flash Attention for Blackwell GPUs.

    Args:
        q, k, v: (B, H, N, D) tensors in bf16/fp16
        causal: whether to apply causal masking
        sm_scale: softmax scale (default: 1/sqrt(D))

    Returns:
        out: (B, H, N, D)
    """
    return FlashAttentionFunc.apply(q, k, v, causal, sm_scale)


def flash_attention_varlen(q, k, v, cu_seqlens, max_seqlen, causal=True, sm_scale=None):
    """
    Flash Attention with variable-length sequences (for sample packing).

    Args:
        q, k, v: (1, H, total_len, D) — packed sequences
        cu_seqlens: (num_seqs + 1,) int32 — cumulative sequence lengths
        max_seqlen: maximum individual sequence length
        causal: apply causal masking within each sequence
        sm_scale: softmax scale

    Returns:
        out: (1, H, total_len, D)
    """
    B, H, total_len, D = q.shape
    assert B == 1, "varlen expects batch_size=1 (packed)"

    num_seqs = cu_seqlens.shape[0] - 1
    out = torch.zeros_like(q)

    if sm_scale is None:
        sm_scale = D ** -0.5

    for i in range(num_seqs):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        q_i = q[:, :, start:end, :].contiguous()
        k_i = k[:, :, start:end, :].contiguous()
        v_i = v[:, :, start:end, :].contiguous()
        out[:, :, start:end, :] = flash_attention(q_i, k_i, v_i, causal=causal, sm_scale=sm_scale)

    return out
