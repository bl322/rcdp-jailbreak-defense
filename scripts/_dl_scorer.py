"""Download stronger MNLI / zero-shot scorer backbones through the HF mirror.

huggingface.co is unreachable from this box; hf-mirror.com works.
Set HF_ENDPOINT before importing huggingface_hub.
"""
import os
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from huggingface_hub import snapshot_download

MODELS = sys.argv[1:] or ["MoritzLaurer/deberta-v3-large-zeroshot-v2.0"]

for mid in MODELS:
    try:
        p = snapshot_download(
            mid,
            allow_patterns=[
                "config.json", "tokenizer.json", "tokenizer_config.json",
                "special_tokens_map.json", "vocab.json", "merges.txt",
                "*.safetensors", "*.bin", "*.model", "*.txt",
            ],
        )
        print("DOWNLOADED", mid, "->", p, flush=True)
    except Exception as e:
        print("FAILED", mid, type(e).__name__, str(e)[:300], flush=True)
print("DL DONE", flush=True)
