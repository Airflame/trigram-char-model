import torch

def prepare_words() -> list:
    return [w.lower() for w in open('polish_names.txt', 'r', encoding="utf8").read().splitlines()]

def create_probability_matrix(N: torch.tensor) -> torch.Tensor:
    P = N.float()
    P /= P.sum(2, keepdim=True)
    P = torch.nan_to_num(P)
    return P

def prepare_mappings(words: list) -> tuple:
    chars = sorted(list(set(''.join(words))))
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s,i in stoi.items()}
    return stoi, itos

def trigram(word_length=None):
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

    P = create_probability_matrix(N)
    g = torch.Generator().manual_seed(2147483641)
    words_left = 100
    while words_left > 0:
        out = []
        indexes = []
        ix = 0
        for i in range(2):
            p = P[0, ix]
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            out.append(itos[ix])
            indexes.append(ix)
        while True:
            p = P[indexes[-2], indexes[-1]]
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            out.append(itos[ix])
            indexes.append(ix)
            if ix == 0:
                break
        word = ''.join(out)
        if word_length is None or len(word) == word_length:
            words_left -= 1
            print(word)

if __name__ == "__main__":
    trigram(word_length=8)