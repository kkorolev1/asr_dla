import editdistance
# Don't forget to support cases when target_text == ''

def calc_cer(target_text, predicted_text) -> float:
    # TODO: your code here
    if len(target_text) == 0:
        return 1 if len(predicted_text) > 0 else 0
    return editdistance.eval(target_text, predicted_text) / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    # TODO: your code here
    if len(target_text) == 0:
        return 1 if len(predicted_text) > 0 else 0
    target_text = target_text.split(' ')
    predicted_text = predicted_text.split(' ')
    return editdistance.eval(target_text, predicted_text) / len(target_text)