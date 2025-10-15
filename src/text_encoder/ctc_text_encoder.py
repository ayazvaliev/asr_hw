import os
from string import ascii_lowercase
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.tokenizer.tokenizer_utils import normalize_text
from torchaudio.models.decoder import ctc_decoder
from pathlib import Path
from typing import Sequence

import torch
from logging import Logger

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    def __init__(
        self,
        lm: str,
        beam_size: int,
        lm_weight: int,
        word_score: int,
        beam_threshold: int,
        logger: Logger,
        lexicon: str | None = None,
        words_path: str | None = None,
        tokenizer: BPETokenizer | None = None,
    ):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """
        self.logger = logger
        if tokenizer is None:
            self.vocab = [BPETokenizer.EMPTY_TOK, BPETokenizer.SILENCE_TOK] + list(ascii_lowercase)
            ind2char = dict(enumerate(self.vocab))
            self.ind2char = lambda x: ind2char[x]
            char2ind = {v: k for k, v in ind2char.items()}
            self.decode_ = lambda xs, merge_tokens: ("" if merge_tokens else " ").join(
                ind2char[x] for x in xs
            )
            self.encode_ = lambda xs: [char2ind.get(x, 1) for x in xs] + [char2ind[BPETokenizer.SILENCE_TOK]]
        else:
            self.vocab = list(tokenizer.get_vocab().items())
            self.vocab.sort(key=lambda tok_id: tok_id[-1])
            self.vocab = [tok for tok, id in self.vocab]

            self.ind2char = tokenizer.id_to_token
            self.decode_ = lambda xs, merge_tokens: tokenizer.decode(xs, merge_tokens=merge_tokens)
            self.encode_ = lambda xs: tokenizer.encode(xs + BPETokenizer.SILENCE_TOK)

        self.silence_tok = self.ind2char(1)
        self.empty_tok = self.ind2char(0)

        if lexicon is not None and not os.path.exists(lexicon):
            assert words_path is not None, "Path to words list is not defined for lexicon forming"
            self._prepare_lexicon(words_path, lexicon)

        self.ctc_decoder = ctc_decoder(
            tokens=self.vocab,
            lexicon=None if tokenizer is None else lexicon,
            lm=lm,
            beam_size=beam_size,
            lm_weight=lm_weight,
            word_score=word_score,
            beam_threshold=beam_threshold,
            blank_token=self.empty_tok,
            sil_token=self.silence_tok,
        )

    def _prepare_lexicon(self, words_path: str, lexicon_path: str) -> None:
        words_path = Path(words_path)
        lexicon_path = Path(lexicon_path)

        try:
            with open(lexicon_path, "w") as lexicon:
                with open(words_path, "r") as words:
                    for word in words:
                        word = word.rstrip()
                        token_seq = self.decode_(self.encode_(word), False).strip()
                        lexicon.write(word + "\t" + token_seq + "\n")
            self.logger.info(f"Lexicon txt file saved {lexicon_path.absolute().resolve()}")
        except Exception as e:
            self.logger.error(f"Error occured during lexicon generation: {e}")

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char(item)

    def encode(self, text) -> torch.Tensor:
        return torch.tensor(self.encode_(text), dtype=torch.long)

    def decode(self, inds: Sequence[int]) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        decoded = self.decode_(inds, True)
        if len(decoded) == 0:
            return decoded
        filtered = [decoded[0]]
        for i in range(1, len(decoded)):
            if decoded[i] == filtered[-1]:
                continue
            filtered.append(decoded[i])
        filtered = (
            "".join(filtered).replace(self.empty_tok, "").replace(self.silence_tok, " ").strip()
        )
        return filtered

    def ctc_decode(
        self, emissions: torch.Tensor, lengths: torch.IntTensor | None = None
    ) -> list[str]:
        emissions = emissions.transpose(0, 1).contiguous()
        if torch.any(torch.isnan(emissions)):
            self.logger.warning("NaNs in emissions during validation AGAIN AHAHAHHAHAHHA")
        predictions = self.ctc_decoder(emissions.cpu(), lengths.cpu())
        decoded_strs = []
        for hypo_list in predictions:
            hypo = hypo_list[0]
            decoded_strs.append(self.decode(hypo.tokens.tolist()))
        return decoded_strs
