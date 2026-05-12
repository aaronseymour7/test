"""
Jastrow factor: phase function and application.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from .config import to_device, JASTROW_CHUNKED, JASTROW_CHUNK


def build_jastrow_indices(n: int, basis: np.ndarray):
    """
    Pre-compute pair indices and upload the basis to the GPU.

    Returns
    -------
    pair_i_np, pair_j_np : np.ndarray  (CPU, int32)
    basis_bits_gpu       : jax array   (GPU, int32)
    """
    pairs      = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pair_i_np  = np.array([p[0] for p in pairs], dtype=np.int32)
    pair_j_np  = np.array([p[1] for p in pairs], dtype=np.int32)
    basis_bits_gpu = to_device(jnp.array(basis, dtype=jnp.int32))
    return pair_i_np, pair_j_np, basis_bits_gpu


def make_jastrow_phase_fn(pair_i_np, pair_j_np, basis_bits_gpu):
    """
    Build a JIT-compiled function that maps theta_J → per-basis-state phases.

    Uses chunked evaluation when ``JASTROW_CHUNKED=True`` to avoid
    materialising the full (n_pair × dim) occupation matrix.
    """
    pi_gpu = to_device(jnp.array(pair_i_np, dtype=jnp.int32))
    pj_gpu = to_device(jnp.array(pair_j_np, dtype=jnp.int32))

    if not JASTROW_CHUNKED:
        def _occ_product(i, j):
            bi = (basis_bits_gpu >> i.astype(jnp.int32)) & 1
            bj = (basis_bits_gpu >> j.astype(jnp.int32)) & 1
            return (bi * bj).astype(jnp.float64)

        _occ_mat_fn = jax.vmap(_occ_product)

        @jax.jit
        def jastrow_phase(theta_J):
            return jnp.dot(theta_J, _occ_mat_fn(pi_gpu, pj_gpu))

    else:
        n_pair = pair_i_np.shape[0]
        chunk  = JASTROW_CHUNK

        @jax.jit
        def jastrow_phase(theta_J):
            phase = jnp.zeros(basis_bits_gpu.shape[0], dtype=jnp.float64)
            for start in range(0, n_pair, chunk):
                end  = min(start + chunk, n_pair)
                pi_c = pi_gpu[start:end]
                pj_c = pj_gpu[start:end]
                th_c = theta_J[start:end]

                def _occ_c(i, j):
                    bi = (basis_bits_gpu >> i.astype(jnp.int32)) & 1
                    bj = (basis_bits_gpu >> j.astype(jnp.int32)) & 1
                    return (bi * bj).astype(jnp.float64)

                phase = phase + jnp.dot(th_c, jax.vmap(_occ_c)(pi_c, pj_c))
            return phase

    return jastrow_phase


def apply_jastrow(psi, theta_J, jastrow_phase_fn):
    """Multiply ``psi`` by exp(i · J(θ))."""
    return psi * jnp.exp(1j * jastrow_phase_fn(theta_J))
