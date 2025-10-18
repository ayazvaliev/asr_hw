import torchaudio
from pathlib import Path
from urllib.parse import urlparse
import gdown

from src.datasets.base_dataset import BaseDataset


def is_valid_url(s: str) -> bool:
    try:
        result = urlparse(s)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


class CustomDirAudioDataset(BaseDataset):
    def __init__(self, data_dir, *args, **kwargs):
        data = []
        if is_valid_url(data_dir):
            files = gdown.download_folder(data_dir, use_cookies=False, quiet=False)
            data_dir = Path(files[0]).parent.parent
        else:
            data_dir = Path(data_dir)

        audio_dir = data_dir / "audio"
        transcription_dir = data_dir / "transcriptions"

        for path in Path(audio_dir).iterdir():
            entry = {}
            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                entry["path"] = str(path)
                if transcription_dir and Path(transcription_dir).exists():
                    transc_path = Path(transcription_dir) / (path.stem + ".txt")
                    if transc_path.exists():
                        with transc_path.open() as f:
                            entry["text"] = f.read().strip()
            if len(entry) > 0:
                t_info = torchaudio.info(entry["path"])
                length = t_info.num_frames / t_info.sample_rate
                entry.update({"audio_len": length})
                data.append(entry)
        super().__init__(data, *args, **kwargs)
