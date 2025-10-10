import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    batch = {
        "text_encoded": pad_sequence(
            [elem["text_encoded"] for elem in dataset_items], batch_first=True
        ),
        "text_encoded_length": torch.tensor(
            [len(elem["text_encoded"]) for elem in dataset_items], dtype=torch.int32
        ),
        "audio_len": torch.tensor([elem["audio_len"] for elem in dataset_items], dtype=torch.int32),
        "audio": pad_sequence([elem["audio"] for elem in dataset_items], batch_first=True),
    }
    batch["audio"] = batch["audio"].unsqueeze(1)
    excluded_keys = set(batch.keys())
    for k in dataset_items[0].keys():
        if k not in excluded_keys:
            batch[k] = [elem[k] for elem in dataset_items]

    return batch
