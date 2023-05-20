# -*- encoding:utf-8 -*-

import os
import random
import sys
import pickle
import pandas as pd
import numpy as np
import torch
from transformers import BertModel, BertTokenizer
from sklearn.neighbors import NearestNeighbors


class KnowledgeGraph(object):

    def __init__(self, spo_files, lookdown=False):
        self.lookdown = lookdown
        self.KGS = {'dsc': './kg/dsc.spo', 'CnDbpedia': './kg/CnDbpedia.spo', 'Medical': './kg/Medical.spo', 'webhealth': './kg/webhealth.spo'}
        self.spo_file_paths = [self.KGS.get(f, f) for f in spo_files]
        self.lookup_table = self._create_lookup_table()
        self.lookdown_table = self._create_lookdown_table()
        self.segment_vocab = list(self.lookup_table.keys())
        self.segment_vocab2 = list(self.lookdown_table.keys())
        self.vocab = self.segment_vocab.extend(self.segment_vocab2)
        self.tokenizer = pkuseg.pkuseg(model_name="default", postag=False, user_dict=self.vocab)

    def _create_lookup_table(self):
        lookup_table = {}
        for spo_path in self.spo_file_paths:
            print("[KnowledgeGraph] Loading spo from {}".format(spo_path))
            with open(spo_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        subj, pred, obje = line.strip().split("\t")    
                    except:
                        print("[KnowledgeGraph] Bad spo:", line)

                    # 这里！！！！！！！宾语+谓语，拼接了起来啊
                    value = pred + obje
                    if subj in lookup_table.keys():
                        lookup_table[subj].add(value)
                    else:
                        lookup_table[subj] = set([value])
        return lookup_table

    def _create_lookdown_table(self):
        lookdown_table = {}
        for spo_path in self.spo_file_paths:
            print("[KnowledgeGraph] Loading spo from {}".format(spo_path))
            with open(spo_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        subj, pred, obje = line.strip().split("\t")    
                    except:
                        print("[KnowledgeGraph] Bad spo:", line)
                    value = subj + pred
                    if obje in lookdown_table.keys():
                        lookdown_table[obje].add(value)
                    else:
                        lookdown_table[obje] = set([value])
        return lookdown_table

    def add_knowledge(self, sent, output_file, is_test):
        all_knowledge = []
        if is_test == True:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(sent+'\n')
        split_sent = self.tokenizer.cut(sent)
        if is_test == True:
            with open(output_file, 'a', encoding='utf-8') as f:
                for each in split_sent:
                    f.write(each+'\t')
                f.write('\n')
        know_sent = []
        for token in split_sent:
            # 实体个屁，是谓语+宾语
            entities = list(self.lookup_table.get(token, []))

            for each in entities:
                all_knowledge.append(token+each)
                if is_test == True:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(token)
                        f.write(each)
                        f.write('\n')
            if self.lookdown == True:
                entities = list(self.lookdown_table.get(token, []))
                for each in entities:
                    # 这里！！！！它直接拼接起来了！！！！！！
                    all_knowledge.append(each+token)
                    if is_test == True:
                        with open(output_file, 'a', encoding='utf-8') as f:
                            f.write(each)
                            f.write(token)
                            f.write('\n')
        if is_test == True:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write('\n')

        # all_knowledge是一个list，里面是所有的文本知识
        return all_knowledge
    
    def get_knowledge(self, sentences, output_file='', is_test=False, label=None):
        knowledge = []
        for i in range(len(sentences)):
            if is_test == True:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(str(label[i]))
                    f.write(',')
            # 输入的是一句话，输出的是这句话的所有知识，是一句一句遍历的
            knowledge.append(self.add_knowledge(sentences[i], output_file, is_test))

        return knowledge


# 经过我们分析，我们已经很清楚，KnowledgeGraph是查表然后输出拼接并编码后的语句，现在我们需要改变知识检索的方式，
# 我们不再使用知识元组，我们直接查询文本，然后根据这个文本直接构造知识输出即可
# 出于简短考虑，我们假定，这个知识文本库只有3句固定的话
# 因此，请你编写对象KnowledgeSentence
class KnowledgeSentence(object):
    def __init__(self,device='cuda', tokenizer=None, encoder=None):
        # 这是你的知识库，包含三个句子
        self.knowledge_base = [
            "The cat is a domestic species of small carnivorous mammal.",
            "Dogs are domesticated mammals, not natural wild animals.",
            "The red fox is the largest of the true foxes."
        ]
        if device == 'cpu':
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        if encoder is None:
            self.encoder = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        else:
            self.encoder = BertModel.from_pretrained(encoder).to(self.device)

    def embed(self,text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64).to(
            self.device)  # 将输入数据移动到 GPU
        with torch.no_grad():
            outputs = self.encoder(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().cpu()  # 将结果移回 CPU


    def compute_embeddings(self,text_samples):
        embeddings = []
        for sample in text_samples:
            embeddings.append(self.embed(sample).detach().numpy())  # 添加 detach() 方法
        return embeddings

    def select_knn_examples(self,x_test, train_data, k):
        train_prompts, train_targets = zip(*train_data)
        train_embeddings = self.compute_embeddings(train_prompts)
        test_embedding = self.embed(x_test).unsqueeze(0).numpy()  # 将结果转换为 NumPy 数组

        # 使用kNN查找最近邻
        nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(train_embeddings)
        distances, indices = nbrs.kneighbors(test_embedding)
        # 返回选定的示例
        selected_examples = [train_data[i] for i in indices.flatten()]
        return selected_examples

    def get_knowledge(self, sentences,output_file='./kg/test_triples.txt', is_test=False, label=None):
        # 这个函数返回与输入句子相关的知识
        # 在这个简单的例子中，我们只是返回整个知识库
        # 在实际应用中，你可能需要根据输入句子的内容来选择返回哪些知识
        knowledge = []

        #     创建文件output_file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('')

        for i, _ in enumerate(sentences):
            if is_test == True:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(str(label[i]))
                    f.write(',')
            # 输入的是一句话，输出的是这句话的所有知识，是一句一句遍历的
            knowledge.append(self.knowledge_base)
        return knowledge
