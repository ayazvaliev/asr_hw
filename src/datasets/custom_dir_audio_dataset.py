import torchaudio
from pathlib import Path
from urllib.parse import urlparse
import gdown
import zipfile

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
            zip_path = "data.zip"
            gdown.download(data_dir, output=zip_path, use_cookies=False, quiet=False, fuzzy=True)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                dirs = [name for name in zip_ref.namelist() if name.endswith('/')]
                assert len(dirs) == 1, "Unsupported file structure for dataset"
                zip_ref.extractall(".")
                data_dir = Path(dirs)
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
