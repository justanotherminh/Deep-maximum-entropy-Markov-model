def load_data(path, lowercase=True):
    sents = []
    tags = []
    with open(path, 'r') as f:
        for line in f.read().splitlines():
            sent = ["<START>"]
            tag = ["<START>"]
            for pair in line.split('####')[1].split(' '):
                tn, tg = pair.rsplit('=', 1)
                if lowercase:
                    sent.append(tn.lower())
                else:
                    sent.append(tn)
                tag.append(tg)
            sents.append(sent)
            tags.append(tag)
    return sents, tags


def count_tokens(sents):
    tokens = {}
    for sent in sents:
        for tok in sent:
            if tok not in tokens:
                tokens[tok] = 0
            tokens[tok] += 1
    return tokens


def remove_hyperlink(sents):
    for sent in sents:
        for i in range(len(sent)):
            if 'http' in sent[i]:
                sent[i] = "<HTTP>"


def remove_outofvocab(sents, vocab):
    for sent in sents:
        for i in range(len(sent)):
            if sent[i] not in vocab:
                sent[i] = "<UNK>"


def remove_rare_words(sents, min_count=1):
    token_count = count_tokens(sents)
    for sent in sents:
        for i in range(len(sent)):
            if token_count[sent[i]] <= min_count:
                sent[i] = "<RARE>"


def get_vocab_idx(train):
    tokens = set()
    for sent in train:
        tokens.update(sent)
    tokens = sorted(list(tokens))
    return dict(zip(tokens, range(1, len(tokens)+1)))


def convert_to_idx(sents, word2idx):
    for sent in sents:
        for i in range(len(sent)):
            sent[i] = word2idx[sent[i]]


def pad_sents(sents, pad_idx=0):
    padded_sents = []
    maxlen = max([len(sent) for sent in sents])
    for sent in sents:
        padded_sent = sent.copy()
        padded_sent.extend([pad_idx]*(maxlen-len(sent)))
        padded_sents.append(padded_sent)
    return padded_sents


def data_iterator(sents, tags, batch_size):
    for i in range(len(sents)//batch_size):
        yield sents[i*batch_size:(i+1)*batch_size], tags[i*batch_size:(i+1)*batch_size]
