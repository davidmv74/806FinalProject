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


input_size = 50
output_size = 180
kernel_size = 3
tan = nn.Tanh()
conv_layer = nn.Conv1d(input_size, output_size, kernel_size)
cnn = nn.Sequential(conv_layer, tan)

def train_model(train_data, model, lr, wd, epochs, batch_size, num_tests):
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=wd)

    for epoch in range(1, epochs+1):
        print("-------------\nEpoch {}:\n".format(epoch))
        result = run_epoch(train_data, model, optimizer, batch_size)
        print('Train MML loss: {:.6f}'.format(result))
        print(" ")
        test_data = parser.get_testing_vectors(idQuestionsDict, embeddings, num_tests)
        testing(model, test_data)

def run_epoch(data, model, optimizer, size):
    #Train model for one pass of train data, and return loss, acccuracy
    model.train()
    losses = []
    numBatches = len(data)/size
    for i in tqdm(range(numBatches)):
        batch = data[i*size:(i+1)*size]
        optimizer.zero_grad()
        #encodings = model.forward(Variable(torch.FloatTensor(batch)))[0]
        y = [0]*size
        x = []
        criterion = nn.MultiMarginLoss()
        for j in range(size):
            sampleBatch = batch[j]
            sampleEncodings = model.forward(Variable(\
                              torch.FloatTensor(sampleBatch).view(22,input_size,4)))
            #print sampleEncodings
            #print "sampleEnc", sampleEncodings
            #refQ = sampleBatch[0]
            #print refQ
            #refQEncoding = model.forward(Variable(\
            #                    torch.FloatTensor(refQ).view(1,input_size,1)))
            distance_vector = [ ]
            input1 = sampleEncodings[0]
            for k in range(1,22):
                input2 = sampleEncodings[k]
                input2Encoding = model.forward(Variable(\
                        torch.FloatTensor(sampleBatch).view(22,input_size,4)))
                cos = nn.CosineSimilarity(dim=0, eps=1e-6)
                dist = cos(input1, input2)
                #print "dist", dist
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

#test on dev
def testing(model, test_data):
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        allRanks = []
        for dev_sample in test_data:
            refQ = dev_sample['refq'].data.numpy().astype(float)
            refQ = model.forward(Variable(torch.FloatTensor(\
                                [refQ])).view(1,input_size,4))
            refQ = torch.mean(refQ[:,:output_size,:], dim=1).view(2,1)
            #print "refQ", refQ
            candidates = dev_sample['candidates']
            candidateIds = [x[0] for x in candidates]
            candidatesCosine = []
            for i in range(len(candidates)):
                #print "cand_before_float", candidates[i][1]
                cand = candidates[i][1].data.numpy().astype(float)
                #print "cand_after_float", cand
                candidateEncoding = model.forward(Variable(\
                                    torch.FloatTensor([cand]).view(1,input_size,4)))
                candidateEncoding = torch.mean(candidateEncoding[:,:output_size,:], dim=1).view(2,1)
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
model = cnn
train_model(label_train_data, model, 0.001, 0, 2, 40, 1000)


'''
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
    print len(embeddingsList), len(embeddingsList[0])
    encodings = model.forward(Variable(torch.FloatTensor(embeddingsList)))[0]
    #print "encodings", encodings
    return encodings

# Trains a model by calling run_batch on the lines passed,
def train_model(lines, idToQuestions, embeddings, model):
    encodings = run_batch(lines, idToQuestions, embeddings, model)
    y = [0]*batchSize
    x = []
    criterion = nn.MultiMarginLoss()
    for i in range(batchSize):
        sampleEncodings = encodings[i*22:(i+1)*22]
        refQ = sampleEncodings[0]
        #print 0, refQ
        #print 1, sampleEncodings[1]
        distance_vector = [ ]
        input1 = refQ
        for j in range(1,22):
            input2 = sampleEncodings[j]
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            #print "input1", input1
            #print "input2", input2
            dist = cos(input1, input2)
            #print "dist", dist
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
            refQ = model.forward(Variable(torch.FloatTensor(\
                [getQuestionEmbedding(splitLines[0], idToQuestions, embeddings)])))[0]
            refQ = torch.mean(refQ[:,:200,:], dim=1).view(120,1)
            candidatesCosine = []
            candidateIds = splitLines[2].split(" ")
            for i in range(20):
                candidateId = candidateIds[i]
                candidateEncoding = model.forward(Variable(torch.FloatTensor(\
                    [getQuestionEmbedding(candidateId, idToQuestions, embeddings)])))[0]
                candidateEncoding = torch.mean(candidateEncoding[:,:200,:], dim=1).view(120,1)
                #print "refQ", refQ
                #print "candQ", candidateEncoding
                #print "cos", cos(refQ, candidateEncoding)
                candidatesCosine.append((candidateId,cos(refQ, candidateEncoding).data[0]))
            #print candidatesCosine
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
    random.shuffle(lines)
    lines = lines[:3000]
    length = len(lines)
    batch_size = int(length/53)
    model = rnn
    #model = DAN()
    epochs = 2
    batches = 1
    for j in range(epochs):
        print("EPOCH:", j)
        loss = 0
        random.shuffle(lines)
        for i in range(batches):
            print("BATCH:",i)
            liness = lines[i*batch_size:(i+1)*batch_size]
            loss += train_model(liness, idToQuestions, embeddings, model)
        loss /= batches
        print("Loss for epoch", j, loss)
        testing('dev.txt', 1000, model, idToQuestions, embeddings)
'''
