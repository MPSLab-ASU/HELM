"""Keep native security pins and installation entry points aligned."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def test_native_manifest_excludes_research_packages():
    text = (ROOT / "pyproject.toml").read_text()
    assert '"setuptools>=84.0.0"' in text
    for package in ("vllm", "deepspeed", "lm-eval", "sqlitedict", "nltk"):
        assert not re.search(r'"' + package + r'(?:[=><!~"\[])', text)


def test_cpu_and_container_torch_pins_match_native_manifest():
    text = (ROOT / "pyproject.toml").read_text()
    for name in ("torch", "torchvision"):
        pin = re.search(r'"(' + name + r'==[^" ]+)"', text).group(1)
        assert pin in (ROOT / "Dockerfile").read_text()
        assert pin in (ROOT / ".github/workflows/ci.yml").read_text()
