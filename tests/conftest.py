
from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

SEED = 1234

def _seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

@pytest.fixture(scope="session")
def tiny_tokenizer():
    try:
        tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained("gpt2")
    tok.add_special_tokens(
        {
            "pad_token": "<pad>",
            "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
        }
    )
    return tok

@pytest.fixture()
def tiny_model_factory(tiny_tokenizer):

    def _make(hidden_size: int = 32, num_hidden_layers: int = 2, num_attention_heads: int = 4):
        _seed_everything()
        config = Qwen2Config(
            vocab_size=len(tiny_tokenizer),
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 2,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=max(1, num_attention_heads // 2),
            max_position_embeddings=256,
            pad_token_id=tiny_tokenizer.pad_token_id,
            bos_token_id=tiny_tokenizer.bos_token_id,
            eos_token_id=tiny_tokenizer.eos_token_id,
        )
        model = Qwen2ForCausalLM(config)
        model.eval()
        return model

    return _make

@pytest.fixture()
def tiny_model(tiny_model_factory):
    return tiny_model_factory()
