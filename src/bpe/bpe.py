import json
import regex as re
from functools import lru_cache


CACHE = {}


@lru_cache()
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


@lru_cache()
def unicode_to_bytes():
    return {v: k for k, v in bytes_to_unicode().items()}


@lru_cache()
def bpe_ranks():
    with open("gpt2_model/vocab.bpe", "r") as f:
        bpe_data = f.read()
    merges = [tuple(merge_str.split()) for merge_str in bpe_data.split("\n")[1:-1]]
    return dict(zip(merges, range(len(merges))))


def get_tokens(text: str) -> list[str]:
    """Get tokens from text."""
    pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    tokens = pattern.findall(text)
    return tokens


def get_pairs(word: str) -> list[tuple[str, str]]:
    pairs = set()
    prev_word = word[0]
    for char in word[1:]:
        pairs.add((prev_word, char))
        prev_word = char
    return pairs


def bpe(token: str) -> str:
    if token in CACHE:
        return CACHE[token]
    word = tuple(token)
    pairs = get_pairs(word)
    if not pairs:
        return token
    while True:
        bigram = min(pairs, key=lambda pair: bpe_ranks().get(pair, float("inf")))
        if bigram not in bpe_ranks():
            break
        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
                new_word.extend(word[i:j])
                i = j
            except ValueError:
                new_word.extend(word[i:])
                break
            if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word[i])
        new_word = tuple(new_word)
        word = new_word
        if len(word) == 1:
            break
        else:
            pairs = get_pairs(word)
    new_token = " ".join(word)
    CACHE[token] = new_token
    return new_token


def tokenizer_encode(text: str) -> list[int]:
    tokens = get_tokens(text)
    vocabs = []
    for token in tokens:
        token = "".join(bytes_to_unicode()[b] for b in token.encode("utf-8"))
        vocabs.extend(bpe(token))
    with open("gpt2_model/encoder.json", "r") as f:
        encoder = json.load(f)
    return [encoder[id] for id in vocabs]


def tokenizer_decode(tokens: list[int]) -> str:
    with open("gpt2_model/encoder.json", "r") as f:
        encoder = json.load(f)
    decoder = {v: k for k, v in encoder.items()}
    text = "".join(decoder[id] for id in tokens)
    byte_array = ([unicode_to_bytes()[c] for c in text])
    text = bytearray(byte_array).decode("utf-8")
    return text


if __name__ == "__main__":
    text = "Hello, world!"
    ids = tokenizer_encode(text)
    text = tokenizer_decode(ids)
    assert text == "Hello, world!", "Tokenization and de-tokenization failed"