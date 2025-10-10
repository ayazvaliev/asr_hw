# Based on seminar materials
# Don't forget to support cases when target_text == ''


import editdistance


def calc_cer(target_text: str, predicted_text: str) -> float | None:
    if len(target_text.split()) == 0:
        return None
    return editdistance.eval(target_text.split(), predicted_text.split()) / len(target_text.split())


def calc_wer(target_text: str, predicted_text: str) -> float:
    if len(target_text) == 0:
        return None
    return editdistance.eval(target_text, predicted_text) / len(target_text)
