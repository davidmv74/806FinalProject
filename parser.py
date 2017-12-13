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

def get_training_vectors(idToQuestions, embeddings):
    training_data = []
    with open('train_random.txt', 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)
        lines = lines[:200]
        for line in tqdm(lines):
            splitLine = line.split("\t")
            negatives = splitLine[2][:-1].split(" ")
            totalLines = []
            positives = splitLine[1].split(" ")
            for j in range(len(positives)):
                embeddingsList = []
                random.shuffle(negatives)
                totalLines = [splitLine[0]]+[positives[j]]+negatives[:20]
                for qId in totalLines:
                    embeddingsList.append(get_question_embedding(qId, idToQuestions, embeddings))
                training_data.append(embeddingsList)
    return training_data

def get_development_vectors(idToQuestions, embeddings, size):
    dev_data = []
    with open('dev.txt', 'rb') as f:
        lines = f.readlines()
        random.shuffle(lines)
        lines = lines[:size]
        for line in tqdm(lines):
            lineEmbeddings = {}
            splitLines = line.split("\t")
            refQ = Variable(torch.FloatTensor(get_question_embedding(splitLines[0], idToQuestions, embeddings)))
            lineEmbeddings['refq'] = refQ
            candidateEmbeddings = []
            lineEmbeddings['positives'] = splitLines[1]
            for i in range(20):
                candidateId = splitLines[2].split(" ")[i]
                candidateEmbeddings.append((candidateId,Variable(torch.FloatTensor(get_question_embedding(candidateId, idToQuestions, embeddings)))))
            lineEmbeddings['candidates'] =candidateEmbeddings
            dev_data.append(lineEmbeddings)
    return dev_data

def get_domain_train_vectors(setOneQ, setTwoQ, embeddings):
    setOneQList = list(setOneQ.keys())
    setTwoQList = list(setTwoQ.keys())
    random.shuffle(setOneQList)
    random.shuffle(setTwoQList)
    setOneQ = [(get_question_embedding(x,setOneQ, embeddings) , 0) for x in setOneQList[:500]]
    setTWoQ = [(get_question_embedding(x,setTwoQ, embeddings) , 0) for x in setTwoQList[:500]]
    pairs = list(setOneQ) + list(setTwoQ)
    random.shuffle(pairs)
    pairs = pairs[:200]
    print(pairs)
    return pairs

def get_questions_id(filee):
    idToQuestions = {}
    with gzip.open(filee, 'rb') as f:
        for file_content in f.readlines():
            qId, qTitle, qBody = file_content.split("\t")
            idToQuestions[qId] = (qTitle, qBody)
    return idToQuestions

def get_embeddings(filee):
    embeddings = {}
    with gzip.open(filee, 'rb') as f:
        for file_content in f:
            word, embedding = file_content.split(" ")[0], file_content.split(" ")[1:-1]
            embeddings[word] = [float(emb) for emb in embedding]
    return embeddings

def get_question_embedding(qId, idToQuestions, embeddings):
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
