import torch
import torch.optim as optim
import torch.nn.functional as F

from sklearn.utils import shuffle
from utils import load_data, remove_hyperlink, remove_outofvocab, get_vocab_idx, convert_to_idx, pad_sents, data_iterator
from model import EncodeSentence, FeedForward

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("Using {}".format(device))
word_dim = 150
tag_dim = 5
use_lstm = False
train_iter = 35
use_word2vec = False

train_sents, train_tags = load_data('data/twitter1_train.txt')
test_sents, test_tags = load_data('data/twitter1_test.txt')

remove_hyperlink(train_sents)
remove_hyperlink(test_sents)

vocab2idx = get_vocab_idx(train_sents)
vocab2idx["<PAD>"] = 0  # Added just for the sake of completion
vocab2idx["<UNK>"] = len(vocab2idx)  # Out-of-vocab words
tag2idx = {"<START>": 0, "O": 1, "T-NEG": 2, "T-NEU": 3, "T-POS": 4}
idx2tag = ["<START>", "O", "T-NEG", "T-NEU", "T-POS"]

if use_word2vec:
    import os
    if os.path.exists("word2vec.pt"):
        word_emb = torch.load("word2vec.pt")
    else:
        from gensim.models import Word2Vec
        from gensim.models import KeyedVectors
        from gensim.test.utils import datapath
        print("Loading Word2Vec vectors")
        wv_from_bin = KeyedVectors.load_word2vec_format(datapath("/homes/cs577/hw2/w2v.bin"), binary=True)
        print("Finished loading vectors")

        word_emb = 0.1 * torch.randn(len(vocab2idx), 300)
        for word, idx in vocab2idx.items():
            if word in wv_from_bin.wv.vocab:
                word_emb[idx] = torch.from_numpy(wv_from_bin[word])
        torch.save(word_emb, "word2vec.pt")
    word_dim = word_emb.size(1)

remove_outofvocab(test_sents, vocab2idx)
convert_to_idx(train_sents, vocab2idx)
convert_to_idx(test_sents, vocab2idx)
convert_to_idx(train_tags, tag2idx)
convert_to_idx(test_tags, tag2idx)

if use_word2vec:
    encoder = EncodeSentence(len(vocab2idx), word_dim, wordEmbedding=word_emb, use_lstm=use_lstm).to(device)
else:
    encoder = EncodeSentence(len(vocab2idx), word_dim, use_lstm=use_lstm).to(device)
model = FeedForward(len(tag2idx), tag_dim, word_dim).to(device)
optimizer = optim.Adam(list(model.parameters())+list(encoder.parameters()))

# Training
for it in range(train_iter):
    print("Iteration: {}".format(it))
    train_sents, train_tags = shuffle(train_sents, train_tags)
    for data, label in data_iterator(train_sents, train_tags, batch_size=100):
        optimizer.zero_grad()
        train_data = torch.tensor(pad_sents(data), device=device)
        train_data = encoder(train_data)
        train_label = torch.tensor(pad_sents(label, 1), device=device)
        maxlen = train_data.size(1)
        loss_sum = torch.tensor([0.], device=device)
        for i in range(1, maxlen):
            output = model(train_data[:, i], train_label[:, i-1])
            loss_sum += F.nll_loss(output, train_label[:, i])
        loss_sum.backward()
        optimizer.step()
    print("Loss: {:.4f}".format(loss_sum.item()/maxlen))

# Testing
test_data = torch.tensor(pad_sents(test_sents), device=device)
test_data = encoder(test_data)
maxlen = test_data.size(1)
table = torch.empty(test_data.size(0), maxlen-1, len(tag2idx), len(tag2idx))  # (batch_size, sent_length, p(current_tag), p(prev_tag))
for i in range(1, maxlen):
    for label in range(len(tag2idx)):
        table[:, i-1, :, label] = model(test_data[:, i], torch.full((test_data.size(0),), label, dtype=torch.long, device=device))
# Viterbi and evaluation
TP = 0
FN = 0
FP = 0
# with open('output.txt', 'w') as f:
for sent_idx in range(test_data.size(0)):
    sent = table[sent_idx]
    # sent.shape == (maxlen-1, len(tag2idx), len(tag2idx))
    dptable = torch.empty((maxlen-1, len(tag2idx)))
    bktrack = torch.empty((maxlen-1, len(tag2idx)), dtype=torch.long)
    dptable[0] = sent[0, :, 1]
    bktrack[0] = torch.ones(len(tag2idx))
    for i in range(1, maxlen-1):
        for j in range(len(tag2idx)):
            dptable[i, j] = (dptable[i-1] + sent[i, j, :]).max()
            bktrack[i, j] = (dptable[i-1] + sent[i, j, :]).argmax()
    sent_length = len(test_tags[sent_idx])-1  # Length without <START>
    best_seq = [None] * sent_length
    best_seq[sent_length-1] = dptable[sent_length-1].argmax().item()
    for i in range(sent_length-2, -1, -1):
        best_seq[i] = bktrack[i+1, best_seq[i+1]].item()
    # f.write(' '.join([idx2tag[tag] for tag in best_seq]))
    # f.write('\n')
    # Evaluation
    for p, t in zip(best_seq, test_tags[sent_idx][1:]):
        if p < 1 or t < 1:
            print("Tag {} is invalid".format(idx2tag[min(p, t)]))
            exit(0)
        if p > 1 and t > 1:
            if p == t:
                TP += 1
            else:
                FN += 1
                FP += 1
        elif p == 1 or t == 1:
            if p > t:
                FP += 1
            elif p < t:
                FN += 1
precision = TP/(TP+FP)
recall = TP/(TP+FN)
microF1 = 2*precision*recall/(precision+recall)
print("Precision {:.4f}".format(precision))
print("Recall: {:.4f}".format(recall))
print("F1: {:.4f}".format(microF1))
