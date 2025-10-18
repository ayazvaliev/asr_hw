import torch
from math import inf
from dataclasses import dataclass
from copy import deepcopy


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
        decoded = []
        for emission, length in zip(emissions, lengths):
            decoded.append(self._beam_search(emission[:length]))

        return decoded

    @staticmethod
    def _log_sum_exp(a: torch.Tensor, b: torch.Tensor):
        res = torch.logsumexp(torch.concat([a.unsqueeze(0), b.unsqueeze(0)]), dim=0).squeeze(0)
        return res

    def _beam_search(self, emission: torch.Tensor) -> str:
        beam_prefixes = {"": BeamEntry(pb=torch.tensor(0.0), pnb=-torch.tensor(torch.inf))}
        for prob in emission:
            new_beam_prefixes = {}
            for prefix in beam_prefixes:
                pb, pnb = beam_prefixes[prefix].pb, beam_prefixes[prefix].pnb
                p_total = CustomCTCDecoder._log_sum_exp(pb, pnb)
                for i, token in enumerate(self.tokens):
                    same_prefix_b = p_total + prob[i]
                    if token == self.blank_token:
                        if prefix in new_beam_prefixes:
                            new_beam_prefixes[prefix].pb = CustomCTCDecoder._log_sum_exp(
                                new_beam_prefixes[prefix].pb, same_prefix_b
                            )
                        else:
                            new_beam_prefixes[prefix] = BeamEntry(
                                pb=same_prefix_b, pnb=-torch.tensor(torch.inf)
                            )
                    elif len(prefix) > 0 and token == prefix[-1]:
                        same_prefix_nb = pnb + prob[i]
                        if prefix in new_beam_prefixes:
                            new_beam_prefixes[prefix].pnb = CustomCTCDecoder._log_sum_exp(
                                new_beam_prefixes[prefix].pnb, same_prefix_nb
                            )
                        else:
                            new_beam_prefixes[prefix] = BeamEntry(
                                pb=-torch.tensor(torch.inf), pnb=same_prefix_nb
                            )

                        new_prefix = prefix + token
                        new_prefix_pnb = pb + prob[i]
                        if new_prefix in new_beam_prefixes:
                            new_beam_prefixes[new_prefix].pnb = CustomCTCDecoder._log_sum_exp(
                                new_beam_prefixes[new_prefix].pnb, new_prefix_pnb
                            )
                        else:
                            new_beam_prefixes[new_prefix] = BeamEntry(
                                pb=-torch.tensor(torch.inf), pnb=new_prefix_pnb
                            )
                    else:
                        new_prefix = prefix + token
                        new_prefix_pnb = p_total + prob[i]
                        if new_prefix in new_beam_prefixes:
                            new_beam_prefixes[new_prefix].pnb = CustomCTCDecoder._log_sum_exp(
                                new_beam_prefixes[new_prefix].pnb, new_prefix_pnb
                            )
                        else:
                            new_beam_prefixes[new_prefix] = BeamEntry(
                                pb=-torch.tensor(torch.inf), pnb=new_prefix_pnb
                            )

            items = []
            for prefix, ent in new_beam_prefixes.items():
                p = CustomCTCDecoder._log_sum_exp(ent.pb, ent.pnb)
                items.append((p, prefix, ent))
            items.sort(key=lambda x: x[0], reverse=True)
            items = items[: self.beam_size]
            beam_prefixes = {prefix: BeamEntry(ent.pb, ent.pnb) for _, prefix, ent in items}

        final_items = []
        for pfx, ent in beam_prefixes.items():
            p = CustomCTCDecoder._log_sum_exp(ent.pb, ent.pnb)
            final_items.append((p, pfx))
        final_items.sort(key=lambda x: x[0], reverse=True)
        best_prefix = final_items[0][1]

        return [best_prefix]
