# KT-Kernel: High-performance kernel operations for KTransformers
# SPDX-License-Identifier: Apache-2.0

"""
KT-Kernel provides high-performance kernel operations for KTransformers,
including CPU-optimized MoE inference with AMX, AVX, and KML support.

The package automatically detects your CPU capabilities and loads the optimal
kernel variant (AMX, AVX512, or AVX2) at runtime.

Example usage:
    >>> from kt_kernel import KTMoEWrapper
    >>> wrapper = KTMoEWrapper(
    ...     layer_idx=0,
    ...     num_experts=8,
    ...     num_experts_per_tok=2,
    ...     hidden_size=4096,
    ...     moe_intermediate_size=14336,
    ...     num_gpu_experts=2,
    ...     cpuinfer_threads=32,
    ...     threadpool_count=2,
    ...     weight_path="/path/to/weights",
    ...     chunked_prefill_size=512,
    ...     method="AMXINT4"
    ... )

    Check which CPU variant is loaded:
    >>> import kt_kernel
    >>> print(kt_kernel.__cpu_variant__)  # 'amx', 'avx512', or 'avx2'

Environment Variables:
    KT_KERNEL_CPU_VARIANT: Override automatic detection ('amx', 'avx512', 'avx2')
    KT_KERNEL_DEBUG: Enable debug output ('1' to enable)
"""

from __future__ import annotations
import sys

# Try to load C++ extensions, but don't fail if they're not available
# This allows using pure Python components (like models) without C++ build
_kt_kernel_ext = None
__cpu_variant__ = "none"
_cpp_extensions_available = False

try:
    from ._cpu_detect import initialize as _initialize_cpu
    _kt_kernel_ext, __cpu_variant__ = _initialize_cpu()
    sys.modules["kt_kernel_ext"] = _kt_kernel_ext
    kt_kernel_ext = _kt_kernel_ext
    _cpp_extensions_available = True
except (ImportError, ModuleNotFoundError, OSError) as e:
    # C++ extensions not available - that's OK for pure Python usage
    kt_kernel_ext = None
    import warnings
    warnings.warn(
        f"kt-kernel C++ extensions not available ({e}). "
        "KTMoEWrapper will not work, but models.xoron can still be used.",
        ImportWarning
    )

# Import main API only if C++ extensions are available
KTMoEWrapper = None
generate_gpu_experts_masks = None

if _cpp_extensions_available:
    try:
        from .experts import KTMoEWrapper
        from .experts_base import generate_gpu_experts_masks
    except ImportError:
        pass

# Read version from package metadata (preferred) or fallback to project root
try:
    from importlib.metadata import version, PackageNotFoundError

    try:
        __version__ = version("kt-kernel")
    except PackageNotFoundError:
        import os
        _root_version_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "version.py")
        if os.path.exists(_root_version_file):
            _version_ns = {}
            with open(_root_version_file, "r", encoding="utf-8") as f:
                exec(f.read(), _version_ns)
            __version__ = _version_ns.get("__version__", "0.4.3")
        else:
            __version__ = "0.4.3"
except ImportError:
    try:
        from pkg_resources import get_distribution, DistributionNotFound
        try:
            __version__ = get_distribution("kt-kernel").version
        except DistributionNotFound:
            __version__ = "0.4.3"
    except ImportError:
        __version__ = "0.4.3"

__all__ = ["KTMoEWrapper", "generate_gpu_experts_masks", "kt_kernel_ext", "__cpu_variant__", "__version__"]
