from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import spacy
import torch

BATCH_SIZE = 128
device = "cuda" if torch.cuda.is_available() else "cpu"
spacy_de = spacy.load('de')
spacy_en = spacy.load('en')




def de_tokenize(text):
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]


def en_tokenize(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def prepare_data():
    src = Field(tokenize=en_tokenize,
                init_token='<SOS>',
                eos_token='<EOS>',
                lower=True)
    tgt = Field(tokenize=de_tokenize,
                init_token='<SOS>',
                eos_token='<EOS>',
                lower=True)
    train, valid, test = Multi30k.splits(exts=('.de', '.en'), fields=(src, tgt))
    print(f"{len(train)} training samples\n{len(valid)} validation samples\n{len(test)} testing samples")
    src.build_vocab(train, min_freq=2)
    tgt.build_vocab(train, min_freq=2)
    return BucketIterator.splits(
        (train, valid, test),
        batch_size=BATCH_SIZE,
        device=device
    ), src, tgt

if __name__ == '__main__':
    train_data, valid_data, test_data = prepare_data()
    print(train_data.batch_size)
