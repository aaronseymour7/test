"""
Hamiltonian construction — both sparse CPU (for exact diagonalisation)
and JAX GPU versions.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from scipy.sparse import lil_matrix, csr_matrix

from .config import J_COUPLING, PBC, to_device


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_n_up(n: int) -> int:
    """Half-filling: ceil(n/2) up-spins."""
    return (n + 1) // 2 if n % 2 == 1 else n // 2


def build_basis(n: int, n_up: int) -> np.ndarray:
    """All bit-strings with exactly *n_up* ones."""
    return np.array(
        [b for b in range(1 << n) if bin(b).count("1") == n_up],
        dtype=np.int64,
    )


# ---------------------------------------------------------------------------
# Sparse CPU Hamiltonian (Lanczos / exact diag)
# ---------------------------------------------------------------------------

def build_hamiltonian(
    n: int,
    n_up: int,
    j: float = J_COUPLING,
    pbc: bool = PBC,
) -> tuple[csr_matrix, np.ndarray, dict]:
    """
    Returns
    -------
    H : csr_matrix
    basis : np.ndarray of shape (dim,)
    idx_map : dict[int, int]  bit-string → row index
    """
    basis   = build_basis(n, n_up)
    idx_map = {int(b): i for i, b in enumerate(basis)}
    H       = lil_matrix((len(basis), len(basis)), dtype=np.float64)
    edges   = (
        [(i, (i + 1) % n) for i in range(n)]
        if pbc
        else [(i, i + 1) for i in range(n - 1)]
    )
    for si, sj in edges:
        for row, bits in enumerate(basis):
            zi = 0.5 if (bits >> si) & 1 else -0.5
            zj = 0.5 if (bits >> sj) & 1 else -0.5
            H[row, row] += j * zi * zj
            if ((bits >> si) & 1) != ((bits >> sj) & 1):
                fl  = bits ^ (1 << si) ^ (1 << sj)
                col = idx_map.get(int(fl), -1)
                if col >= 0:
                    H[row, col] += 0.5 * j
    return csr_matrix(H), basis, idx_map


# ---------------------------------------------------------------------------
# JAX GPU Hamiltonian
# ---------------------------------------------------------------------------

def build_jax_hamiltonian(
    n: int,
    n_up: int,
    j: float = J_COUPLING,
    pbc: bool = PBC,
) -> tuple:
    """
    Returns (h_rows, h_cols, h_vals) as JAX arrays on the preferred device.
    """
    basis_list = [b for b in range(1 << n) if bin(b).count("1") == n_up]
    idx_map    = {b: i for i, b in enumerate(basis_list)}
    rows, cols, vals = [], [], []
    edges = (
        [(i, (i + 1) % n) for i in range(n)]
        if pbc
        else [(i, i + 1) for i in range(n - 1)]
    )
    for i, js in edges:
        for row, bits in enumerate(basis_list):
            zi = 0.5 if (bits >> i) & 1 else -0.5
            zj = 0.5 if (bits >> js) & 1 else -0.5
            rows.append(row); cols.append(row); vals.append(j * zi * zj)
            if ((bits >> i) & 1) != ((bits >> js) & 1):
                fl = bits ^ (1 << i) ^ (1 << js)
                if fl in idx_map:
                    rows.append(row)
                    cols.append(idx_map[fl])
                    vals.append(0.5 * j)
    h_rows = to_device(jnp.array(rows, dtype=jnp.int32))
    h_cols = to_device(jnp.array(cols, dtype=jnp.int32))
    h_vals = to_device(jnp.array(vals, dtype=jnp.float64))
    return h_rows, h_cols, h_vals


def make_apply_H(h_rows, h_cols, h_vals, dim: int):
    """Return a JIT-compiled H|ψ⟩ function."""
    @jax.jit
    def apply_H(psi):
        return (
            jnp.zeros(dim, dtype=psi.dtype)
            .at[h_rows].add(h_vals * psi[h_cols])
        )
    return apply_H


# ---------------------------------------------------------------------------
# Reference state
# ---------------------------------------------------------------------------

def neel_state(
    n: int,
    n_up: int,
    basis: list,
    idx_map: dict,
):
    """Néel state |↑↓↑↓…⟩ as a JAX complex vector."""
    neel_bits = sum(1 << i for i in range(n) if i % 2 == 0)
    psi = jnp.zeros(len(basis), dtype=jnp.complex128)
    psi = psi.at[idx_map[neel_bits]].set(1.0)
    return to_device(psi)
