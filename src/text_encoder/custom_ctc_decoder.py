import torch
from math import inf
from dataclasses import dataclass


@dataclass
class BeamEntry:
    pb: float
    pnb: float

    def __iter__(self):
        return iter((self.pb, self.pnb))


class CustomCTCDecoder:
    def __init__(self, tokens: list[str], beam_size: int, blank_token: int):
        self.tokens = tokens
        self.beam_size = beam_size
        self.blank_token = blank_token

    def __call__(self, emissions: torch.Tensor, lengths: torch.IntTensor) -> list[str]:
        # emissions (N, T, C)
        emissions = emissions.exp()
        decoded = []
        for emission, length in zip(emissions, lengths):
            decoded.append(self._beam_search(emission[:length]))

        return decoded

    def _beam_search(self, emission: torch.Tensor) -> str:
        beam_prefixes = {"": BeamEntry(pb=1, pnb=0)}
        for prob in emission:
            for prefix in beam_prefixes:
                pb, pnb = beam_prefixes[prefix].pb, beam_prefixes[prefix].pnb
                for i, token in enumerate(self.tokens):
                    if token == self.blank_token:
                        beam_prefixes[prefix].pb += (pb + pnb) * prob[i]
                    elif token == prefix[-1]:
                        beam_prefixes[prefix].pnb += pnb * prob[i]
                        new_prefix = prefix + token
                        if new_prefix in beam_prefixes:
                            beam_prefixes[new_prefix].pnb += pb * prob[i]
                        else:
                            beam_prefixes[new_prefix] = BeamEntry(pb=0, pnb=pb * prob[i])
                    else:
                        new_prefix = new_prefix + token
                        if new_prefix in beam_prefixes:
                            beam_prefixes[new_prefix].pnb += (pb + pnb) * prob[i]
                        else:
                            beam_prefixes[new_prefix] = BeamEntry(pb=0, pnb=(pb + pnb) * prob[i])
            sorted_prefixes = sorted([(k, pb, pnb) for k, (pb, pnb) in beam_prefixes.items()], key=lambda k, pb, pnb: pb + pnb, reverse=True)
            sorted_prefixes = sorted_prefixes[:self.beam_size]
            beam_prefixes = {prefix: BeamEntry(pb, pnb) for prefix, pb, pnb in sorted_prefixes}

        sorted_prefixes = sorted([(k, pb, pnb) for k, (pb, pnb) in beam_prefixes.items()], key=lambda k, pb, pnb: pb + pnb, reverse=True)
        if len(sorted_prefixes) != 0:
            return [sorted_prefixes[0][0]]
        else:
            return []
