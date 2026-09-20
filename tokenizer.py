"""Self-contained Qwen2 tokenizer for the standalone engine.

Loads the model's fast-tokenizer definition (`tokenizer.json`) directly through
the `tokenizers` library, producing ids that match HF AutoTokenizer exactly,
without importing `transformers`.

Dependencies: `tokenizers` only. Data required next to the weights: tokenizer.json.

Usage:
    tok = Qwen2Tokenizer.from_pretrained(model_dir)
    ids = tok.encode("The capital of France is")
    text = tok.decode(out_ids, skip_special_tokens=True)
"""

import os


class Qwen2Tokenizer:
    def __init__(self, tokenizer):
        self._t = tokenizer

    @classmethod
    def from_pretrained(cls, model_dir: str, tokenizer_file: str = "tokenizer.json"):
        from tokenizers import Tokenizer

        path = os.path.join(model_dir, tokenizer_file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing {path}: the model dir must ship tokenizer.json")
        return cls(Tokenizer.from_file(path))

    @property
    def vocab_size(self) -> int:
        return self._t.get_vocab_size()

    @property
    def eos_token_id(self):
        return self._t.token_to_id("<|endoftext|>")

    @property
    def pad_token_id(self):
        return self._t.token_to_id("<|endoftext|>")

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return self._t.encode(text).ids

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return self._t.decode(list(ids), skip_special_tokens=skip_special_tokens)