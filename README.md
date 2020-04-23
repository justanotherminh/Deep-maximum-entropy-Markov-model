# Deep maximum entropy Markov model
A PyTorch implementation of the DMEMM model for NER tagging

A Maximum entropy Markov model (MEMM) conditions the probability of a tag on the previous tag and the current input.
A DMEMM uses a neural network to approximate this probability distribution.
The input can be raw word vectors or contextualized vectors produced by a BiLSTM.
