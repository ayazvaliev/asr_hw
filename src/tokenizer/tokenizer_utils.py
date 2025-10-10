import re
from pathlib import Path
from typing import Callable, Generator


def normalize_text(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z ]", "", text)
    return text.strip()


def base_line_parser(line: str) -> str:
    text = " ".join(line.split()[1:])
    return normalize_text(text)


def text_stream(
    data_dir: str | Path, line_parser: Callable[[str], str] = base_line_parser
) -> Generator[str, None, None]:
    for text_path in Path(data_dir).rglob("*.trans.txt"):
        try:
            with open(text_path, "r") as text_data:
                for line in text_data:
                    line = line_parser(line)
                    if not line:
                        continue
                    yield line
        except Exception as e:
            print(f"Unable to read from {text_path.absolute().resolve()}: {e}")
