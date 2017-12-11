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
from evaluation import Evaluation

# DAN MODEL
class DAN(nn.Module):

    def __init__(self):
        super(DAN, self).__init__()

        embed_dim = 200     # 200 Initial States
        hidden_dim = embed_dim      # Also 200 hidden states, we could change this as hyper parameter (they use around 250 on paper)

        self.W_input = nn.Linear(embed_dim, hidden_dim)

    def forward(self, review_features):
        hidden = F.tanh(self.W_input(review_features)) # 200 -> hidden_dim
        return hidden

# Get embeddings corresponding to a question's title/body by averaging word embeddings
def getQuestionEmbedding(qId, idToQuestions, embeddings):
    title, body  = idToQuestions[qId]
    bodyEmbedding = np.zeros(200)
    for word in body.split(" "):
        if word in embeddings:
            bodyEmbedding += embeddings[word]
    bodyEmbedding /= len(body)

    titleEmbedding = np.zeros(200)
    for word in title.split(" "):
        if word in embeddings:
            titleEmbedding += embeddings[word]
    titleEmbedding /= len(title)

    return (titleEmbedding+bodyEmbedding)/2

# Runs one batch of samples (passed in as lines) on the model
def run_batch(lines, idToQuestions, embeddings, model):
    embeddingsList = []
    for line in lines:
        splitLine = line.split("\t")
        negatives = splitLine[2][:-1].split(" ")
        random.shuffle(negatives)
        totalLines = []
        positives = splitLine[1].split(" ")
        for j in range(len(positives)):
            totalLines = [splitLine[0]]+[positives[j]]+negatives[:20]
            for qId in totalLines:
                embeddingsList.append(getQuestionEmbedding(qId, idToQuestions, embeddings))
    encodings = model.forward(Variable(torch.FloatTensor(embeddingsList)))
    return encodings

# Trains a model by calling run_batch on the lines passed,
def train_model(lines, idToQuestions, embeddings, model):
    learning_rate = 10**-3
    optimizer = torch.optim.Adam(model.parameters(), lr = 10**-3, weight_decay=10**-2)
    batchSize = len(lines)
    encodings = run_batch(lines, idToQuestions, embeddings, model)
    y = [0]*batchSize
    x = []
    criterion = nn.MultiMarginLoss()
    for i in range(batchSize):
        sampleEncodings = encodings[i*22:(i+1)*22]
        refQ = sampleEncodings[0]
        distance_vector = [ ]
        input1 = refQ
        for j in range(1,22):
            input2 = sampleEncodings[j]
            cos = nn.CosineSimilarity(dim=0, eps=1e-6)
            dist = cos(input1, input2)
            distance_vector.append(dist)
        x.append(torch.cat(distance_vector))
    x = torch.cat(x)
    y = Variable(torch.LongTensor(y))
    x = x.view(y.data.size(0), -1)
    loss = criterion(x, y)
    loss.backward()
    optimizer.step()
    for p in model.parameters():
        p.data.add_(-learning_rate, p.grad.data)
    return loss.data[0]

#test on dev
def testing(filee, numSamples, model, idToQuestions, embeddings):
    with open(filee, 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)
        lines = lines[:numSamples]
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        allRanks = []
        for line in lines:
            splitLines = line.split("\t")
            refQ = model.forward(Variable(torch.FloatTensor(getQuestionEmbedding(splitLines[0], idToQuestions, embeddings))))
            candidatesCosine = []
            for i in range(20):
                candidateId = splitLines[2].split(" ")[i]
                candidateEncoding = model.forward(Variable(torch.FloatTensor(getQuestionEmbedding(candidateId, idToQuestions, embeddings))))
                candidatesCosine.append((candidateId,cos(refQ, candidateEncoding).data[0]))
            sortedCosines = sorted(candidatesCosine, key = lambda x: x[1], reverse=True)
            sortedRanks = [splitLines[2].split(" ").index(cand[0])  for cand in sortedCosines]
            zeroOrOne = [1 if cand[0] in splitLines[1] else 0 for cand in sortedCosines]
            allRanks.append(zeroOrOne)
        evaluation = Evaluation(allRanks)
        print("MAP", evaluation.MAP())
        print("MRR", evaluation.MRR())
        print("P@1", evaluation.Precision(1))


# Get all the questions associated to an id
idToQuestions = {}
with gzip.open('text_tokenized.txt.gz', 'rb') as f:
    for file_content in f.readlines():
        qId, qTitle, qBody = file_content.split("\t")
        idToQuestions[qId] = (qTitle, qBody)

# get all word embeddings
embeddings = {}
with gzip.open('vectors_pruned.200.txt.gz', 'rb') as f:
    for file_content in f:
        word, embedding = file_content.split(" ")[0], file_content.split(" ")[1:-1]
        embeddings[word] = [float(emb) for emb in embedding]

# train
with open('train_random.txt', 'r') as f:
    lines = f.readlines()
    length = len(lines)
    batch_size = int(length/53)
    model = DAN()
    epochs = 5
    batches = 7
    for j in range(epochs):
        print("EPOCH:", j)
        loss = 0
        random.shuffle(lines)
        for i in range(batches):
            print("BATCH:",i)
            liness = lines[i*batch_size:(i+1)*batch_size]
            loss += train_model(liness, idToQuestions, embeddings, model)
        loss /= batches
        print("Loss for epoch", i, loss)
        testing('dev.txt', 1000, model, idToQuestions, embeddings)
