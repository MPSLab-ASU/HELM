"""Packaging regression tests.

These guard the two real defects fixed during the release-readiness pass:

1. The runtime used to depend on an unpackaged ``experiments/`` tree, so an
   installed wheel could not run. The pipeline implementation now lives in
   ``helm._pipeline`` and ``helm/runtime/inference.py`` must not import from
   ``experiments``.
2. The native AVX2 kernel source (``helm/kernels/fp16_gemv.cpp``) was declared
   as package-data but a regression could drop it from the built package. It
   must resolve relative to the installed ``helm.kernels`` package so the check
   passes both in-tree and when installed from a wheel.

All checks are CPU-only, need no network, and perform no build step.
"""

import ast
import importlib.resources
from pathlib import Path


def test_pipeline_module_is_importable_from_helm_package():
    # Defect 1: the pipeline moved into the installed package. Importing it via
    # the ``helm`` namespace must succeed without the source ``experiments/``
    # tree being present.
    import helm._pipeline

    assert helm._pipeline is not None


def test_inference_runtime_does_not_import_from_experiments():
    # Defect 1: statically assert helm/runtime/inference.py has no import that
    # reaches into the unpackaged ``experiments`` tree.
    import helm.runtime.inference as inference

    source = Path(inference.__file__).read_text()
    tree = ast.parse(source)

    offending = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "experiments" or alias.name.startswith("experiments."):
                    offending.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "experiments" or module.startswith("experiments."):
                offending.append(module)

    assert not offending, f"inference.py imports from experiments: {offending}"


def test_native_kernel_cpp_source_ships_with_package():
    # Defect 2: the AVX2 kernel .cpp source must ship inside the installed
    # helm.kernels package. importlib.resources resolves it both in-tree and
    # from an installed wheel.
    resource = importlib.resources.files("helm.kernels").joinpath("fp16_gemv.cpp")

    assert resource.is_file(), f"missing kernel source: {resource}"
    assert resource.read_text().strip(), "kernel source is empty"
