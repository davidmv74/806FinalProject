import os
import sys
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
import tqdm
import datetime
import pdb
import gzip
import numpy as np
import random

def get_model():
    return DAN()

class DAN(nn.Module):

    def __init__(self):
        super(DAN, self).__init__()

        embed_dim = 200
        hidden_dim = embed_dim

        self.W_input = nn.Linear(embed_dim, hidden_dim)
        #self.W_hidden = nn.Linear(hidden_dim, out_dim)

    def forward(self, review_features):
        hidden = F.tanh(self.W_input(review_features)) # 300 -> hidden_dim
        #out = F.log_softmax(self.W_hidden(hidden))
        return hidden

'''
def train_model(train_data, dev_data, model, lr, wd):

    optimizer = torch.optim.Adam(model.parameters(), lr = 10**-1, weight_decay=10**-3)
    torch.manual_seed(1)
    model.train()

    for epoch in range(1, 51):

        print("-------------\nEpoch {}:\n".format(epoch))

        result = run_epoch(train_data, True, model, optimizer, 173)

        print('Train NLL loss: {:.6f}'.format( result[0]))
        print('Train Precision: {:.6f}'.format( result[1]))

        print()

        #val_result = run_epoch(dev_data, False, model, optimizer, 19)
        #print('Val NLL loss: {:.6f}'.format( val_result[0]))
        print('Val Precision: {:.6f}'.format( val_result[1]))
'''

def getQuestionEmbedding(qId, idToQuestions, embeddings):
    title, body  = idToQuestions[qId]
    bodyEmbedding = np.zeros(200)
    titleEmbedding = np.zeros(200)
    for word in body.split(" "):
        if word in embeddings:
            bodyEmbedding += embeddings[word]
    bodyEmbedding /= len(body)

    for word in title.split(" "):
        if word in embeddings:
            titleEmbedding += embeddings[word]
    titleEmbedding /= len(title)

    return (titleEmbedding+bodyEmbedding)/2

def run_batch(size, idToQuestions, embeddings):
    with open('train_random.txt', 'r') as f:
        embeddingsList = []
        i = 0
        while i < size:
            line = f.readline()
            splitLine = line.split("\t")
            negatives = splitLine[2][:-1].split(" ")
            random.shuffle(negatives)
            totalLines = []
            positives = splitLine[1].split(" ")
            for j in range(len(positives)):
                i += 1
                if i > 20:
                    break
                totalLines = [splitLine[0]]+[positives[j]]+negatives[:20]
                print(len(totalLines))
                for qId in totalLines:
                    embeddingsList.append(getQuestionEmbedding(qId, idToQuestions, embeddings))
        model = get_model()
        testytest = model.forward(Variable(torch.FloatTensor(embeddingsList)))
        print(testytest)

idToQuestions = {}
titleAndBodies = []
with gzip.open('text_tokenized.txt.gz', 'rb') as f:
    for file_content in f.readlines():
        qId, qTitle, qBody = file_content.split("\t")
        #print(qBody)
        idToQuestions[qId] = (qTitle, qBody)
        titleAndBodies.append(qTitle)
        titleAndBodies.append(qBody)
embeddings = {}
with gzip.open('vectors_pruned.200.txt.gz', 'rb') as f:
    for file_content in f:
        word, embedding = file_content.split(" ")[0], file_content.split(" ")[1:-1]
        embeddings[word] = [float(emb) for emb in embedding]

run_batch(20, idToQuestions, embeddings)

#print(embeddings)
