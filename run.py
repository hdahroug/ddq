import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import engine as E
from tokenizer import Qwen2Tokenizer

_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_model_dir() -> str:
    env = os.environ.get("QWEN_MODEL_DIR")
    if env:
        return env
    for name in ("Qwen2.5-0.5B", "qwen2-model"):
        for d in (os.path.join(_DIR, name), os.path.join(os.path.dirname(_DIR), name)):
            if os.path.exists(os.path.join(d, "model.safetensors")):
                return d
    return os.path.join(os.path.dirname(_DIR), "Qwen2.5-0.5B")


MODEL_DIR = _find_model_dir()


def main() -> None:
    if not os.path.exists(os.path.join(MODEL_DIR, "model.safetensors")):
        sys.exit(
            f"weights not found under {MODEL_DIR}.\n"
            f"Download them with:\n"
            f"  huggingface-cli download Qwen/Qwen2.5-0.5B --local-dir Qwen2.5-0.5B "
            f"--include config.json model.safetensors tokenizer.json\n"
            f"(or set QWEN_MODEL_DIR to an existing model dir)"
        )

    tok = Qwen2Tokenizer.from_pretrained(MODEL_DIR)
    eng = E.Engine.load(MODEL_DIR)

    args = sys.argv[1:]
    prompt = " ".join(args[:-1]) or "The capital of France is"
    max_new = int(args[-1]) if args and args[-1].isdigit() else 30

    ids = torch.tensor([tok.encode(prompt)], device="cuda")
    out = eng.generate(ids, max_new_tokens=max_new, eos_token_id=tok.eos_token_id)
    print(f"[prompt] {prompt}")
    print(tok.decode(out, skip_special_tokens=True))


if __name__ == "__main__":
    main()