import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import load_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dropout_rate = 0.25
# device = torch.device("cpu")


class EncodeSentence(nn.Module):

    def __init__(self, n_words, word_dim, wordEmbedding=None, use_lstm=False):
        super(EncodeSentence, self).__init__()
        self.use_lstm = use_lstm
        if wordEmbedding:
            self.word_emb = nn.Embedding.from_pretrained(wordEmbedding)
        else:
            self.word_emb = nn.Embedding(n_words, word_dim, padding_idx=0)
        if use_lstm:
            self.bilstm = nn.LSTM(word_dim, word_dim//2, dropout=dropout_rate, batch_first=True, bidirectional=True)

    def forward(self, sent):
        sent_embed = self.word_emb(sent)
        if self.use_lstm:
            sent_embed, _ = self.bilstm(sent_embed)
        return sent_embed


class FeedForward(nn.Module):

    def __init__(self, n_tags, tag_dim, word_dim, tagEmbedding=None, hidden_dim=150):
        super(FeedForward, self).__init__()
        self.n_tags = n_tags
        if tagEmbedding:
            self.tag_emb = nn.Embedding.from_pretrained(tagEmbedding)
        else:
            self.tag_emb = nn.Embedding(n_tags, tag_dim, padding_idx=0)
        self.fc1 = nn.Linear(word_dim+2*tag_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, word_embed, prev_tag):
        batch_size = word_embed.size(0)
        tag_embed = self.tag_emb(prev_tag)  # batch_size, tag_dim
        X = torch.cat((word_embed, tag_embed), dim=1)  # batch_size, word_dim+tag_dim
        X = torch.stack([X]*self.n_tags)  # n_tags, batch_size, word_dim+tag_dim
        all_embed = self.tag_emb(torch.arange(0, self.n_tags, 1, device=device))  # n_tags, tag_dim
        all_embed = torch.stack([all_embed]*batch_size, dim=1)  # n_tags, batch_size, tag_dim
        X = torch.cat((X, all_embed), dim=-1)  # n_tags, batch_size, word_dim+2*tag_dim
        X = X.reshape(-1, X.size(-1))  # n_tags*batch_size, word_dim+2*tag_dim
        X = self.dropout(F.leaky_relu(self.fc1(X)))  # n_tags*batch_size, hidden_dim
        X = self.fc2(X).squeeze().reshape(self.n_tags, batch_size).t()  # batch_size, n_tags
        return F.log_softmax(X, dim=1)
