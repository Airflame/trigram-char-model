import torch
import matplotlib.pyplot as plt

def prepare_words() -> list:
    return [w.lower() for w in open('polish_names.txt', 'r', encoding="utf8").read().splitlines()]

def create_probability_matrix(N: torch.tensor) -> torch.Tensor:
    P = N.float()
    P /= P.sum(1, keepdim=True)
    P = torch.nan_to_num(P)
    return P

def prepare_mappings(words: list) -> tuple:
    chars = sorted(list(set(''.join(words))))
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s,i in stoi.items()}
    return stoi, itos

def bigram():
    words = prepare_words()

    stoi, itos = prepare_mappings(words)
    l = len(stoi)

    N = torch.zeros((l, l), dtype=torch.int32)
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            N[ix1, ix2] += 1

    P = create_probability_matrix(N)
    g = torch.Generator().manual_seed(2147483647)
    for i in range(100):
        out = []
        ix = 0
        while True:
            p = P[ix]
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            out.append(itos[ix])
            if ix == 0:
                break
        print(''.join(out))

def trigram(show_plot=False):
    words = prepare_words()

    stoi, itos = prepare_mappings(words)
    l = len(stoi)

    N = torch.zeros((l*l, l), dtype=torch.int32)
    for w in words:
        chs = ['.', '.'] + list(w) + ['.', '.']
        for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            ix3 = stoi[ch3]
            N[ix1*l + ix2, ix3] += 1

    if show_plot:
        plt.figure(figsize=(16, 16*16))
        plt.imshow(N, cmap='Blues')
        for i in range(l*l):
            for j in range(l):
                chstr = itos[i // l] + itos[i % l] + itos[j]
                plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
                plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")
        plt.axis('off')
        plt.show()

    P = create_probability_matrix(N)
    g = torch.Generator().manual_seed(2147483643)
    for i in range(100):
        out = []
        indexes = []
        ix = 0
        for j in range(2):
            p = P[ix]
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            out.append(itos[ix])
            indexes.append(ix)
        while True:
            p = P[indexes[-2]*l + indexes[-1]]
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            out.append(itos[ix])
            indexes.append(ix)
            if ix == 0:
                break
        print(''.join(out))

if __name__ == "__main__":
    trigram()