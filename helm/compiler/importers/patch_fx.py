"""
Monkey-patch for ``torch.fx._symbolic_trace._patch_function``.

Bypasses ``ValueError: code: co_varnames is too small`` that occurs on
Python 3.10+ when FX tries to trace *into* HuggingFace decoder layers
whose ``forward()`` signatures have many keyword arguments.

The patch is applied automatically on import so every FX tracing path
(``DecodeTracer``, ``dev_pipeline.py``, ``compile_graph``, …) is covered.

Calling ``apply_fx_patch()`` multiple times is safe (idempotent).
"""

import torch.fx._symbolic_trace

_APPLIED = False


def apply_fx_patch():
    """
    Replace ``torch.fx._symbolic_trace._patch_function`` with a version
    that gracefully handles the ``co_varnames is too small`` error.

    Idempotent — safe to call multiple times.
    """
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True

    original_patch_function = torch.fx._symbolic_trace._patch_function

    def patched_patch_function(fn, nargs):
        co = fn.__code__
        # If the function already accepts enough positional args, skip
        # the CodeType rewrite entirely.
        if co.co_argcount >= nargs:
            return fn

        # Otherwise, try the original implementation first.
        try:
            return original_patch_function(fn, nargs)
        except ValueError as e:
            if "varnames is too small" in str(e):
                # HF forward() methods with many **kwargs hit this path.
                # Returning the original function is safe because FX
                # symbolic tracing largely ignores kwargs unpacking.
                return fn
            raise

    torch.fx._symbolic_trace._patch_function = patched_patch_function


# Auto-apply on import so every FX tracing path is covered.
apply_fx_patch()
