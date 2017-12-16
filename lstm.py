import os
import sys
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
import datetime
import pdb
import gzip
import numpy as np
import random
from evaluation import Evaluation
import parser

inp = 200
h = 120
rnn = nn.LSTM(input_size=inp, hidden_size=h, num_layers=1, dropout=0.2)

def train_model(train_data, model, lr, wd, epochs, batch_size, num_tests):
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=wd)

    for epoch in range(1, epochs+1):
        print("-------------\nEpoch {}:\n".format(epoch))
        result = run_epoch(train_data, model, batch_size, optimizer)
        print('Train MML loss: {:.6f}'.format(result))
        print(" ")
        test_data = parser.get_testing_vectors(idQuestionsDict, embeddings, num_tests)
        testing(model, test_data)

def run_epoch(data, model, size, optimizer):
    #Train model for one pass of train data, and return loss, acccuracy
    '''length = len(data)
    cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    y = Variable(torch.LongTensor([1]+[0]*20))
    criterion = nn.MultiMarginLoss()
    losses = []
    for i in tqdm(range(length)):
        embList = data[i]
        count = 0
        #print len(embList)
        refQ, candidates = None , []
        optimizer.zero_grad()
        for embedding in embList:
            titleEmbedding = embedding[0]
            bodyEmbedding = embedding[1]
            titleHidden = []
            bodyHidden = []
            for wordEmbedding in titleEmbedding:
                wordEmbedding = torch.FloatTensor(wordEmbedding).view(1,1,inp)
                wordHidden = rnn.forward(Variable(wordEmbedding))[0]
                titleHidden.append(wordHidden)
            for wordEmbedding in bodyEmbedding:
                wordEmbedding = torch.FloatTensor(wordEmbedding).view(1,1,inp)
                wordHidden = rnn.forward(Variable(wordEmbedding))[0]
                bodyHidden.append(wordHidden)
            #print titleHidden
            titleTensor = torch.stack(titleHidden)
            bodyTensor = torch.stack(bodyHidden)
            titleTensorPooled = torch.mean(titleTensor, dim=0)
            bodyTensorPooled = torch.mean(bodyTensor, dim=0)
            avgTensor = (titleTensorPooled + bodyTensorPooled)/2.0
            if count == 0:
                refQ = avgTensor
            else:
                candidates.append(avgTensor)
            count += 1
        cosSim = []
        for cand in candidates:
            cosSim.append(cos(refQ, cand))
        cosSim = torch.stack(cosSim).view(21,)
        loss = criterion(cosSim, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.data[0])
    avg_loss = np.mean(losses)
    return avg_loss'''
    model.train()
    losses = []
    numBatches = len(data)/size
    for i in tqdm(range(numBatches)):
        batch = data[i*size:(i+1)*size]
        optimizer.zero_grad()
        #print "batch", batch, np.array(batch).shape
        #encodings = model.forward(Variable(torch.FloatTensor(batch)))[0]
        #print "encodings", encodings
        y = [0]*size
        x = []
        criterion = nn.MultiMarginLoss()
        for j in range(size):
            sampleBatch = batch[j]
            #print "sampleBatch", sampleBatch[0]
            refQ = torch.from_numpy(sampleBatch[0]).view(1,1,inp).float()
            refQ = model.forward(Variable(refQ))[0]
            #print "refQ", refQ
            #refQ = torch.mean(refQ, dim=0).view(h,1)
            distance_vector = [ ]
            input1 = refQ
            for k in range(1,22):
                input2 = torch.from_numpy(np.array(sampleBatch[k])).view(1,1,inp).float()
                input2 = model.forward(Variable(input2))[0]
                input2 = torch.mean(input2, dim=0).view(h,1)
                cos = nn.CosineSimilarity(dim=0, eps=1e-6)
                dist = cos(input1, input2)
                #print "dist", dist
                distance_vector.append(dist)
            x.append(torch.cat(distance_vector))
        x = torch.cat(x)
        y = Variable(torch.LongTensor(y))
        x = x.view(y.data.size(0), -1)
        #print x, y
        loss = criterion(x, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.data[0])
    # Calculate epoch level scores
    avg_loss = np.mean(losses)
    return avg_loss

#test on dev
def testing(model, test_data):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    allRanks = []
    for dev_sample in test_data:
        refQ = dev_sample['refq'].data.numpy().astype(float)
        #print "sample", refQ
        refQ = model.forward(Variable(torch.FloatTensor(\
                            [refQ])))[0]
        #print "refQ", refQ[0]
        refQ = torch.mean(refQ[0], dim=0).view(h,1)
        #print "refQ mean", refQ
        candidates = dev_sample['candidates']
        candidateIds = [x[0] for x in candidates]
        candidatesCosine = []
        for i in range(len(candidates)):
            cand = candidates[i][1].data.numpy().astype(float)
            candidateEncoding = model.forward(Variable(torch.FloatTensor([cand])))[0]
            candidateEncoding = torch.mean(candidateEncoding[0], dim=0).view(h,1)
            #print "candidateEncoding", candidateEncoding
            candidatesCosine.append((candidates[i][0],cos(refQ, candidateEncoding).data[0]))
        sortedCosines = sorted(candidatesCosine, key = lambda x: x[1], reverse=True)
        sortedRanks = [candidateIds.index(cand[0])  for cand in sortedCosines]
        zeroOrOne = [1 if cand[0] in dev_sample['positives'] else 0 for cand in sortedCosines]
        allRanks.append(zeroOrOne)
    evaluation = Evaluation(allRanks)
    print("MAP", evaluation.MAP())
    print("MRR", evaluation.MRR())
    print("P@1", evaluation.Precision(1))
    print("P@5", evaluation.Precision(5))

#get all word embeddings
print("Getting Word Embeddings...")
embeddings = parser.get_embeddings('vectors_pruned.200.txt.gz')
idQuestionsDict = parser.get_questions_id("text_tokenized.txt.gz")

# train
print("Getting ALL Train Samples...")
label_train_data = parser.get_training_vectors(idQuestionsDict, embeddings)
model = rnn
train_model(label_train_data, model, 0.001, 0, 1, 40, 1000)
