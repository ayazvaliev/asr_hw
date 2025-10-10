import os
import sentencepiece as spm
import tempfile
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace

from src.tokenizer.tokenizer_utils import text_stream

from typing import Sequence


class BPETokenizer:
    EMPTY_TOK = "-"
    SILENCE_TOK = "|"

    def __init__(self, model_path: str | Path):
        self.tokenizer: Tokenizer = Tokenizer.from_file(model_path)

    @staticmethod
    def train(
        data_dir: str | Path,
        init_vocab_path: str | Path,
        vocab_size: int,
        save_path: str,
        **trainer_kwargs,
    ) -> "BPETokenizer":
        with open(init_vocab_path, "r") as vocab_f:
            vocab = [w.strip() for w in vocab_f]
        tokenizer = Tokenizer(BPE(unk_token=BPETokenizer.SILENCE_TOK, fuse_unk=True))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            initial_alphabet=vocab,
            special_tokens=[BPETokenizer.EMPTY_TOK, BPETokenizer.SILENCE_TOK],
            **trainer_kwargs,
        )
        tokenizer.train_from_iterator(text_stream(data_dir), trainer=trainer)
        tokenizer.pre_tokenizer = None
        tokenizer.save(save_path)

        return BPETokenizer(save_path)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: Sequence[int], merge_tokens=True) -> str:
        if merge_tokens:
            return "".join(self.tokenizer.decode(ids, skip_special_tokens=False).split())
        else:
            return self.tokenizer.decode(ids, skip_special_tokens=False)

    def get_vocab(self) -> dict[int, str]:
        return self.tokenizer.get_vocab(with_added_tokens=True)

    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def id_to_token(self, id: int) -> str:
        token = self.tokenizer.id_to_token(id)
        return token if token is not None else BPETokenizer.SILENCE_TOK

    def token_to_id(self, token: str) -> int:
        id = self.token_to_id(token)
        return id if id is not None else self.token_to_id(BPETokenizer.SILENCE_TOK)
