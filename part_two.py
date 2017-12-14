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
import part_one


# DAN MODEL
class DAN(nn.Module):

    def __init__(self):
        super(DAN, self).__init__()

        embed_dim = 299     # 200 Initial States
        hidden_dim = embed_dim      # Also 200 hidden states, we could change this as hyper parameter (they use around 250 on paper)

        self.W_input = nn.Linear(embed_dim, hidden_dim)

    def forward(self, review_features):
        hidden = F.tanh(self.W_input(review_features)) # 200 -> hidden_dim
        return hidden

# DOMAIN ADAPTATION DAN
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        embed_dim = 299     # 200 Initial States
        hidden_dim = 2      # The domain it belongs to

        self.W_input = nn.Linear(embed_dim, 2)
        self.W_hidden = nn.Sigmoid()

    def forward(self, review_features):
        hidden = F.tanh(self.W_input(review_features)) # 200 -> hidden_dim
        output = self.W_hidden(hidden)
        return output


def train_adversarial_model(idToQuestions, embeddings,label_train_data, domain_train_data, label_model, domain_model, lr, wd, epochs, batch_size, num_tests):
    domain_optimizer = torch.optim.Adam(domain_model.parameters(), lr = lr, weight_decay=wd)
    label_optimizer = torch.optim.Adam(label_model.parameters(), lr = -lr, weight_decay=wd)
    for epoch in range(1, epochs+1):
        print("-------------\nEpoch {}:\n".format(epoch))
        result = run_adversarial_epoch(label_train_data, domain_train_data, label_model, domain_model, domain_optimizer, label_optimizer, batch_size)
        print('Train MML loss: {:.6f}'.format(result))
        print(" ")

        dev_data = parser.get_android_samples('dev.neg.txt', 'dev.pos.txt', embeddings, idToQuestions, num_tests)
        testing_android(label_model, dev_data)


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
        totalLoss = label_loss-(10**-3)*domain_loss
        totalLoss.backward()
        label_optimizer.step()
        domain_optimizer.step()
        print(domain_loss.data[0])
        print(label_loss.data[0])
        losses.append(totalLoss.data[0])
    # Calculate epoch level scores
    avg_loss = np.mean(losses)
    return avg_loss

def testing_android(model, test_data):
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    auc = meter.AUCMeter()
    for dev_sample in test_data:
        target = [x[1] for x in dev_sample[1:]]
        embeddings = [x[0] for x in dev_sample]
        refQ = model.forward(Variable(torch.FloatTensor(embeddings[0])))
        candidates = embeddings[1:]
        candidatesCosine = []
        for i in range(len(candidates)):
            candidateEncoding = model.forward(Variable(torch.FloatTensor(candidates[i])))
            candidatesCosine.append(cos(refQ, candidateEncoding).data[0])
        auc.add(np.array(candidatesCosine), np.array(target))
    print(auc.value(0.05))

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

    # get all word embeddings
    print("Getting Glove Embeddings...")
    embeddings = parser.get_embeddings('glove.840B.300d.txt', vocabulary)

    # train
    print("Getting 2000 Train Samples...")
    label_train_data = parser.get_training_vectors(idToQuestionsUbuntu, embeddings)

    labelModel = DAN()
    domainModel = Net()
    print("Getting Domain Classification Pairs...")
    domain_train_data = parser.get_domain_train_vectors(idToQuestionsUbuntu, idToQuestionsAndroid, embeddings, len(label_train_data))
    print("Getting Development Samples...")
    dev_data = parser.get_android_samples('dev.neg.txt', 'dev.pos.txt', embeddings, idToQuestionsAndroid, 1000)
    print("HERE")
    # Part 2 Milestone 1.a)
    #testing_android(labelModel, dev_data)

    # Part 2 Milestone 1.b)
    #part_one.train_model(idToQuestionsAndroid, embeddings, label_train_data, labelModel, 0.0001, 0, 1, 40, 1000, True)

    # Part 2 Milestine 2
    train_adversarial_model(idToQuestionsAndroid, embeddings, label_train_data, domain_train_data, labelModel, domainModel, 0.001, 0, 3, 40, 1000)
