"""
Givens (particle-number-conserving) rotations in the Fock basis.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from .config import to_device, JAX_DEVICE


def build_givens_pairs(n: int, basis: list, idx_map: dict):
    """
    Pre-compute CSR-like index arrays for the Givens scan.

    Returns
    -------
    srcs_flat_gpu, dsts_flat_gpu : jax arrays (GPU, int32)
    row_ptr_np                   : np.ndarray (CPU, int32)
    """
    srcs_ragged, dsts_ragged = [], []
    for i in range(n):
        for j in range(i + 1, n):
            srcs, dsts = [], []
            for row, bits in enumerate(basis):
                if ((bits >> i) & 1) and not ((bits >> j) & 1):
                    flipped = bits ^ (1 << i) ^ (1 << j)
                    if flipped in idx_map:
                        srcs.append(row)
                        dsts.append(idx_map[flipped])
            srcs_ragged.append(np.array(srcs, dtype=np.int32))
            dsts_ragged.append(np.array(dsts, dtype=np.int32))

    counts  = np.array([len(s) for s in srcs_ragged], dtype=np.int32)
    row_ptr = np.zeros(len(counts) + 1, dtype=np.int32)
    row_ptr[1:] = np.cumsum(counts)

    srcs_cat = (
        np.concatenate(srcs_ragged) if srcs_ragged
        else np.array([], dtype=np.int32)
    )
    dsts_cat = (
        np.concatenate(dsts_ragged) if dsts_ragged
        else np.array([], dtype=np.int32)
    )

    srcs_flat_gpu = to_device(jnp.array(srcs_cat, dtype=jnp.int32))
    dsts_flat_gpu = to_device(jnp.array(dsts_cat, dtype=jnp.int32))

    nnz    = len(srcs_cat)
    n_pair = n * (n - 1) // 2
    print(
        f"[GivensPairs N={n}]  n_pair={n_pair}  total_nnz={nnz}  "
        f"GPU mem≈{nnz*2*4/1e6:.1f} MB  device={JAX_DEVICE}"
    )
    return srcs_flat_gpu, dsts_flat_gpu, row_ptr


def givens_scan_csr(
    psi,
    thetas,
    srcs_flat_gpu,
    dsts_flat_gpu,
    row_ptr_np: np.ndarray,
    imag: bool = False,
):
    """
    Apply one layer of Givens rotations to ``psi`` in-place (functionally).

    Parameters
    ----------
    imag : bool
        If ``True``, use imaginary (XY-type) rotations instead of real ones.
    """
    n_pair = row_ptr_np.shape[0] - 1
    for k in range(n_pair):
        start = int(row_ptr_np[k])
        end   = int(row_ptr_np[k + 1])
        if start == end:
            continue
        srcs_k = srcs_flat_gpu[start:end]
        dsts_k = dsts_flat_gpu[start:end]
        c = jnp.cos(thetas[k])
        s = jnp.sin(thetas[k])
        p_s, p_d = psi[srcs_k], psi[dsts_k]
        if imag:
            new_s =  c * p_s - 1j * s * p_d
            new_d = -1j * s * p_s + c * p_d
        else:
            new_s = c * p_s - s * p_d
            new_d = s * p_s + c * p_d
        psi = psi.at[srcs_k].set(new_s).at[dsts_k].set(new_d)
    return psi
