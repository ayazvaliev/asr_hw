import os
from string import ascii_lowercase
from src.tokenizer.bpe_tokenizer import BPETokenizer
from torchaudio.models.decoder import ctc_decoder
from pathlib import Path
from typing import Sequence

from src.text_encoder.custom_ctc_decoder import CustomCTCDecoder

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
        logger: Logger,
        lexicon: str | None = None,
        words_path: str | None = None,
        tokenizer: BPETokenizer | None = None,
        use_custom_decode: bool = False
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

        self.lexicon = lexicon
        self.lm = lm

        if not use_custom_decode:
            if lexicon is not None and not os.path.exists(lexicon):
                assert words_path is not None, "Path to words list is not defined for lexicon forming"
                self._prepare_lexicon(words_path, lexicon)
            self.ctc_decoder = ctc_decoder(
                tokens=self.vocab,
                lexicon=self.lexicon,
                lm=self.lm,
                beam_size=beam_size,
                lm_weight=lm_weight,
                word_score=word_score,
                blank_token=self.empty_tok,
                sil_token=self.silence_tok,
            )
            self.ctc_decode = self._ctc_decode_pytorch
        else:
            self.custom_ctc_decoder = CustomCTCDecoder(
                tokens=self.vocab,
                beam_size=beam_size,
                blank_token=self.empty_tok
            )
            self.ctc_decode = self._ctc_decode_custom

    def reinitialize_decoder(self, word_score: float, lm_weight: float, beam_size: int):
        self.ctc_decoder = ctc_decoder(
            tokens=self.vocab,
            lexicon=self.lexicon,
            lm=self.lm,
            beam_size=beam_size,
            lm_weight=lm_weight,
            word_score=word_score,
            blank_token=self.empty_tok,
            sil_token=self.silence_tok
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
            msg = f"Lexicon txt file saved {lexicon_path.absolute().resolve()}"
            if self.logger is not None:
                self.logger.info(msg)
            else:
                print(msg)
        except Exception as e:
            msg = f"Error occured during lexicon generation: {e}"
            if self.logger is not None:
                self.logger.error(msg)
            else:
                print(msg)

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char(item)

    def encode(self, text) -> torch.Tensor:
        return torch.tensor(self.encode_(text), dtype=torch.long)

    def _filter_text_pred(self, text: str) -> str:
        if len(text) == 0:
            return text
        filtered = [text[0]]
        for i in range(1, len(text)):
            if text[i] == filtered[-1]:
                continue
            filtered.append(text[i])
        filtered = (
            "".join(filtered).replace(self.empty_tok, "").replace(self.silence_tok, " ").strip()
        )
        return filtered

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
        return self._filter_text_pred(decoded)

    def _ctc_decode_pytorch(
        self, emissions: torch.Tensor, lengths: torch.IntTensor | None = None
    ) -> list[str]:
        emissions = emissions.transpose(0, 1).contiguous()  # (N, T, C)
        predictions = self.ctc_decoder(emissions.cpu(), lengths.cpu())
        predictions = [" ".join(prediction[0].words).strip() for prediction in predictions]
        return predictions

    def _ctc_decode_custom(self, emissions: torch.Tensor, lengths: torch.IntTensor):
        emissions = emissions.transpose(0, 1).contiguous()
        predictions = self.custom_ctc_decoder(emissions.cpu(), lengths.cpu())
        decoded_strs = []
        for hypo_list in predictions:
            hypo = hypo_list[0].replace(self.empty_tok, "").replace(self.silence_tok, " ").strip()
            decoded_strs.append(hypo)
        return decoded_strs
