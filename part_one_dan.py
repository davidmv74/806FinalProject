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
import meter
from sklearn.feature_extraction.text import CountVectorizer
import part_two

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

def train_model(idToQuestions, embeddings, train_data, model, lr, wd, epochs, batch_size, transfer= False):
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=wd)

    for epoch in range(1, epochs+1):
        print("-------------\nEpoch {}:\n".format(epoch))
        result = run_epoch(train_data, model, optimizer, batch_size)
        print('Train MML loss: {:.6f}'.format(result))
        print(" ")
        if not transfer:
            dev_data = parser.get_development_vectors('dev.txt', idToQuestions, embeddings)
            test_data = parser.get_development_vectors('test.txt', idToQuestions, embeddings)
            print("DEV")
            testing(model, dev_data)
            print("TEST")
            testing(model, test_data)
        else:
            dev_data = parser.get_android_samples('test.neg.txt', 'test.pos.txt', embeddings, idToQuestions, 200)
            part_two.testing_android_eval(model, dev_data)
            part_two.testing_android_auc(model, dev_data)

def run_epoch(data, model, optimizer, size):
    '''
    Train model for one pass of train data, and return loss
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
            for k in range(1,22):
                input2 = sampleEncodings[k]
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

#test function
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
        print("P@1", evaluation.Precision(5))

if __name__ == "__main__":
    # Get all the questions associated to an id from each data set
    print("Getting Ubuntu Dataset...")
    idToQuestionsUbuntu = parser.get_questions_id('text_tokenized.txt.gz')
    print("Getting Android Dataset...")
    idToQuestionsAndroid = parser.get_questions_id('corpus.tsv.gz')

    print("Getting Words in datasets...")
    vectorizer = CountVectorizer(ngram_range=(1,1), token_pattern=r"\b\w+\b")
    files = gzip.open('text_tokenized.txt.gz', 'rb').readlines()+ gzip.open('corpus.tsv.gz', 'rb').readlines()
    vocabulary = vectorizer.fit_transform(files)
    vocabulary = vectorizer.get_feature_names()
    vocabulary = dict(zip(vocabulary, range(len(vocabulary))))

    labelModel = DAN()

    # get all word embeddings
    print("Getting Word Embeddings...")
    embeddings = parser.get_embeddings('vectors_pruned.200.txt.gz', vocabulary)

    # train
    print("Getting 2000 Train Samples...")
    label_train_data = parser.get_training_vectors(idToQuestionsUbuntu, embeddings)

    print("Training Model...")
    train_model(idToQuestionsUbuntu, embeddings, label_train_data, labelModel, 0.001, 0, 5, 80)
