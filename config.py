"""
Global hyperparameters and backend selection.

Override any of these before calling ``run()`` or pass them explicitly
to the functions that accept keyword arguments.
"""
from __future__ import annotations

import os

import jax

# ---------------------------------------------------------------------------
# JAX setup
# ---------------------------------------------------------------------------
jax.config.update("jax_enable_x64", True)

_GPU_DEVICES = jax.devices("gpu")
_CPU_DEVICE  = jax.devices("cpu")[0]
JAX_DEVICE   = _GPU_DEVICES[0] if _GPU_DEVICES else _CPU_DEVICE


def to_device(x):
    """Move a JAX array to the preferred device (GPU if available)."""
    return jax.device_put(x, JAX_DEVICE)


# ---------------------------------------------------------------------------
# PennyLane backend
# ---------------------------------------------------------------------------
PENNYLANE_BACKEND: str = os.environ.get("PENNYLANE_BACKEND", "lightning.qubit")
MPS_BOND_DIM:      int = int(os.environ.get("MPS_BOND_DIM", "64"))


def make_pennylane_device(n_wires: int):
    """Try backends in preference order; return (device, backend_name)."""
    import pennylane as qml

    for backend in [PENNYLANE_BACKEND, "lightning.qubit", "default.qubit"]:
        try:
            if backend == "lightning.tensor":
                dev = qml.device(
                    backend, wires=n_wires,
                    method="mps", max_bond_dim=MPS_BOND_DIM,
                )
            else:
                dev = qml.device(backend, wires=n_wires)
            print(f"[PennyLane device]  N={n_wires}  backend={backend}")
            return dev, backend
        except Exception as exc:
            print(f"[PennyLane]  {backend} unavailable ({exc}), trying next…")
    raise RuntimeError("No PennyLane backend could be initialised.")


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------
J_COUPLING: float = 1.0
PBC:        bool  = True


# ---------------------------------------------------------------------------
# VMC / RBM
# ---------------------------------------------------------------------------
ALPHA:       int = 3
VMC_SAMPLES: int = 1024
VMC_STEPS:   int = 600
SEED:        int = 23

# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------
K_MAX:          int   = 1
E_TOL:          float = 5e-3
N_RESTARTS:     int   = 3
N_COLD_RESTARTS: int  = 2

LBFGS_MAXITER: int = 800
LBFGS_MAXFUN:  int = 50_000

# ---------------------------------------------------------------------------
# Ansatz variants
# ---------------------------------------------------------------------------
VARIANTS: list[str] = ["re", "im", "g"]

# ---------------------------------------------------------------------------
# Jastrow GPU chunking
# ---------------------------------------------------------------------------
JASTROW_CHUNKED: bool = True
JASTROW_CHUNK:   int  = 32
