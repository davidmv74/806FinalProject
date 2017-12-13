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

# DOMAIN ADAPTATION DAN
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        embed_dim = 200     # 200 Initial States
        hidden_dim = 2      # The domain it belongs to

        self.W_input = nn.Linear(embed_dim, 2)
        self.W_hidden = nn.Sigmoid()

    def forward(self, review_features):
        hidden = F.tanh(self.W_input(review_features)) # 200 -> hidden_dim
        output = self.W_input(hidden)
        return output

def train_adversarial_model(label_train_data, domain_train_data, label_model, domain_model, lr, wd, epochs, batch_size, num_tests):
    domain_optimizer = torch.optim.Adam(domain_model.parameters(), lr = lr, weight_decay=wd)
    label_optimizer = torch.optim.Adam(label_model.parameters(), lr = -lr, weight_decay=wd)
    for epoch in range(1, epochs+1):
        print("-------------\nEpoch {}:\n".format(epoch))
        result = run_adversarial_epoch(label_train_data, domain_train_data, label_model, domain_model, domain_optimizer, label_optimizer, batch_size)
        print('Train MML loss: {:.6f}'.format(result))
        print(" ")
        #dev_data = parser.get_development_vectors(idToQuestionsUbuntu, embeddings, num_tests)
        #testing(model, dev_data)

def train_model(train_data, model, lr, wd, epochs, batch_size, num_tests):
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=wd)

    for epoch in range(1, epochs+1):
        print("-------------\nEpoch {}:\n".format(epoch))
        result = run_epoch(train_data, model, optimizer, batch_size)
        print('Train MML loss: {:.6f}'.format(result))
        print(" ")
        dev_data = parser.get_development_vectors(idToQuestionsUbuntu, embeddings, num_tests)
        testing(model, dev_data)

def run_epoch(data, model, optimizer, size):
    '''
    Train model for one pass of train data, and return loss, acccuracy
    '''
    model.train()
    losses = []
    numBatches = len(data)/size
    for i in tqdm(range(numBatches)):
        batch = data[i*size:(i+1)*size]
        optimizer.zero_grad()
        encodings = model.forward(Variable(torch.FloatTensor(batch)))
        y = [0]*size
        x = []
        criterion = nn.MultiMarginLoss()
        for i in range(size):
            sampleEncodings = encodings[i]
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
        losses.append(loss.data[0])
    # Calculate epoch level scores
    avg_loss = np.mean(losses)
    return avg_loss

def run_adversarial_epoch(label_train_data, domain_train_data, label_model, domain_model, domain_optimizer, label_optimizer, size):
    '''
    Train model for one pass of train data, and return loss, acccuracy
    '''
    domain_model.train()
    label_model.train()
    losses = []
    numBatches = len(label_train_data)/size

    for i in tqdm(range(numBatches)):
        label_batch = label_train_data[i*size:(i+1)*size]
        domain_batch = domain_train_data[i*size:(i+1)*size]
        domain_inputs = [x[0] for x in domain_batch]
        domain_y = [x[1] for x in domain_batch]
        domain_optimizer.zero_grad()
        label_optimizer.zero_grad()
        print(type(domain_batch), domain_batch)
        encodings = label_model.forward(Variable(torch.FloatTensor(label_batch)))
        pairInput = label_model.forward(Variable(torch.FloatTensor(domain_inputs)))
        domains = domain_model.forward(pairInput)
        y = [0]*size
        x = []
        criterion = nn.MultiMarginLoss()
        for i in range(size):
            sampleEncodings = encodings[i]
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
        label_loss = criterion(x, y)
        domain_loss = criterion(domains, y.long())
        totalLoss = label_loss-(10**-7)*domain_loss
        totalLoss.backward()
        label_optimizer.step()
        domain_optimizer.step()
        losses.append(totalLoss.data[0])
    # Calculate epoch level scores
    avg_loss = np.mean(losses)
    return avg_loss

# Trains a model by calling run_batch on the lines passed,
def train_model_domain(lines, pairs, idToQuestions, embeddings, model, domainModel):
    lambdaa = 10**-7
    batchSize = len(lines)
    encodings = run_batch(lines, idToQuestions, embeddings, model)
    pairInput = [x[0] for x in pairs]
    domainInput = model.forward(Variable(torch.FloatTensor(pairInput)))
    pairY = [x[1] for x in pairs]
    print(domainInput)
    domains = domainModel.forward(domainInput) #pass output from encoder
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
    domainLoss = criterion(domains, y.long())
    totalLoss = loss - lambdaa*domainLoss
    totalLoss.backward()
    return loss.data[0]

#test on dev
def testing(model, dev_data):
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        allRanks = []
        for dev_sample in dev_data:
            refQ = model.forward(dev_sample['refq'])
            candidates = dev_sample['candidates']
            candidateIds = [x[0] for x in candidates]
            candidatesCosine = []
            for i in range(len(candidates)):
                candidateEncoding = model.forward(candidates[i][1])
                candidatesCosine.append((candidates[i][0],cos(refQ, candidateEncoding).data[0]))
            sortedCosines = sorted(candidatesCosine, key = lambda x: x[1], reverse=True)
            sortedRanks = [candidateIds.index(cand[0])  for cand in sortedCosines]
            zeroOrOne = [1 if cand[0] in dev_sample['positives'] else 0 for cand in sortedCosines]
            allRanks.append(zeroOrOne)
        evaluation = Evaluation(allRanks)
        print("MAP", evaluation.MAP())
        print("MRR", evaluation.MRR())
        print("P@1", evaluation.Precision(1))

# Get all the questions associated to an id from each data set
print("Getting Ubuntu Dataset...")
idToQuestionsUbuntu = parser.get_questions_id('text_tokenized.txt.gz')
print("Getting Android Dataset...")
idToQuestionsAndroid = parser.get_questions_id('corpus.tsv.gz')

# get all word embeddings
print("Getting Word Embeddings...")
embeddings = parser.get_embeddings('vectors_pruned.200.txt.gz')

# train
print("Getting 2000 Train Samples...")
label_train_data = parser.get_training_vectors(idToQuestionsUbuntu, embeddings)
labelModel = DAN()
domainModel = Net()
print("Getting Domain Classification Pairs...")
domain_train_data = parser.get_domain_train_vectors(idToQuestionsUbuntu, idToQuestionsAndroid, embeddings)
print("Training Model...")
#train_model(label_train_data, labelModel, 0.001, 0, 3, 40, 1000)
train_adversarial_model(label_train_data, domain_train_data, labelModel, domainModel, 0.001, 0, 3, 40, 1000)


'''
with open('train_random.txt', 'r') as f:
    lines = f.readlines()
    length = len(lines)
    batch_size = int(length/53)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 0
    batches = 0
    for j in range(epochs):
        print("EPOCH:", j)
        loss = 0
        random.shuffle(lines)
        for i in range(batches):
            print("BATCH:",i)
            optimizer.zero_grad()
            liness = lines[i*batch_size:(i+1)*batch_size]
            loss += train_model(liness, idToQuestions, embeddings, model, optimizer)
            optimizer.step()
        loss /= batches
        print("Loss for epoch", j, loss)
        testing('dev.txt', 1000, model, idToQuestions, embeddings)


# train domain classifier as well
with open('train_random.txt', 'r') as f:
    lines = f.readlines()
    length = len(lines)
    zeroPairs = [(getQuestionEmbedding(x, idToQuestions, embeddings),0) for x in list(idToQuestions)[:length/2]]
    onePairs = [(getQuestionEmbedding(x, idToQuestionsAndroid, embeddings),1) for x in list(idToQuestionsAndroid)[:length/2]]
    pairs = zeroPairs+onePairs
    batch_size = int(length/53)
    model = DAN()
    domainModel = RNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    optimizer_domain = torch.optim.Adam(domainModel.parameters(), lr=-0.05)
    epochs = 1
    batches = 3
    for j in range(epochs):
        print("EPOCH:", j)
        loss = 0
        random.shuffle(lines)
        random.shuffle(pairs)
        for i in range(batches):
            print("BATCH:",i)
            optimizer.zero_grad()
            optimizer_domain.zero_grad()
            liness = lines[i*batch_size:(i+1)*batch_size]
            pairss = pairs[i*batch_size:(i+1)*batch_size]
            loss += train_model_domain(liness, pairss, idToQuestions, embeddings, model, domainModel)
            optimizer.step()
            optimizer_domain.step()
        loss /= batches
        print("Loss for epoch", j, loss)
        testing_domain('dev.txt', 1000, model, idToQuestions, embeddings)
'''
