import torch

def is_female_polish_name(word: str) -> bool:
    return word[:-1].endswith('a') and word[:-1] != 'kuba'

def is_selected_gender(word: str, gender: str) -> bool:
    return gender not in ('f', 'm') or is_female_polish_name(word) and gender == 'f' or not is_female_polish_name(word) and gender == 'm'

def format_as_name(word: str) -> str:
    word = word.replace('.', '')
    return word[0].upper() + word[1:]

def prepare_words() -> list:
    return [w.lower() for w in open('polish_names.txt', 'r', encoding="utf8").read().splitlines()]

def create_probability_matrix(N: torch.tensor, dimension: int) -> torch.Tensor:
    P = N.float()
    P /= P.sum(dimension, keepdim=True)
    P = torch.nan_to_num(P)
    return P

def prepare_mappings(words: list) -> tuple:
    chars = sorted(list(set(''.join(words))))
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s,i in stoi.items()}
    return stoi, itos

def trigram(word_length=None, word_count=100, gender=None):
    results = []
    words = prepare_words()

    stoi, itos = prepare_mappings(words)
    l = len(stoi)

    N = torch.zeros((l, l, l), dtype=torch.int32)
    for w in words:
        chs = ['.', '.'] + list(w) + ['.', '.']
        for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            ix3 = stoi[ch3]
            N[ix1, ix2, ix3] += 1

    P = create_probability_matrix(N, 2)
    g = torch.Generator()
    words_left = word_count
    while words_left > 0:
        out = []
        indexes = []
        while True:
            p = P[indexes[-2] if len(indexes) >= 2 else 0, indexes[-1] if len(indexes) >= 1 else 0]
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            out.append(itos[ix])
            indexes.append(ix)
            if ix == 0:
                break
        word = ''.join(out)
        if (word_length is None or len(word) == word_length + 1) and is_selected_gender(word, gender):
            words_left -= 1
            results.append(format_as_name(word))
    return results