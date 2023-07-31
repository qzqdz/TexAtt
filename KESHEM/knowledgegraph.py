# -*- encoding:utf-8 -*-

import os
import random
import sys
import pickle
import pandas as pd
import numpy as np
import torch
from transformers import  AutoTokenizer, AutoModel
from sklearn.neighbors import NearestNeighbors
import ast
import json
from tqdm import tqdm


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
    def __init__(self, k, device='cuda', kg=None,model_path=None,only_knowledge_mode=True,analysis_match=False):
        self.only_knowledge_mode = only_knowledge_mode
        self.analysis_match = analysis_match
        self.k = k


        # 1.纯匹配样本版

        # 2.知识补充版

        # 3.知识扩展版

        # 4.分析版
        if not kg:
            self.knowledge_base = {
                                            "社区传播没有得到有效遏制，但舆论传播得到了有效遏制":"社区传播被忽视，而对舆论传播采取了严格限制。",
                                            "上述情况被业主指出后，索性垃圾不运了，任由其在楼道口腐烂发臭、滋生蚊虫":"对问题置之不理，导致垃圾积聚、恶臭扩散、蚊虫滋生。",
                                            "上海现在的情况就是用战术的勤奋试图掩盖战略的懒惰":"采取表面上的努力掩盖对长期战略问题的懒散态度。",
                                            "摆脱封建主义的干扰，资本主义的诱惑，这才是真正符合全体中华人民利益的中华复兴！": "强调中华复兴应该避免封建主义和资本主义的负面影响，符合广大中华人民的利益。",
                                            "总结来说上海国际大都市确实不好管，上海政府也没能力管，甚至不添乱已经拜佛了，虽然中央派人督察还各种物资调度，但是也耐不住上海的各级官员的能力。": "对上海政府和官员能力的肯定，尽管面对难以管控的国际大都市问题，仍能做出努力和调度。",
                                            "但毕竟这是我的根，是我成长和安身立命的地方，哪怕有一天走到了天涯海角，我都会始终挂念上海，始终期盼上海会好。": "表达对上海的深情厚意和对上海未来的期盼，将上海视为重要的故乡。",
                                            }
        else:
            self.knowledge_base = kg


        if device == 'cpu':
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if model_path is None:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.encoder = AutoModel.from_pretrained('bert-base-uncased').to(self.device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.encoder = AutoModel.from_pretrained(model_path).to(self.device)


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

        if self.analysis_match==True:
            assert self.only_knowledge_mode==False, "Shut down the knowledge mode!"

        if self.only_knowledge_mode==False:
            if self.analysis_match:
                train_prompts = list(train_data.keys())
            else:
                train_prompts = list(train_data.values())
        else:
            train_prompts = list(train_data.values())


        train_embeddings = self.compute_embeddings(train_prompts)
        test_embedding = self.embed(x_test).unsqueeze(0).numpy()  # 将结果转换为 NumPy 数组

        # 使用kNN查找最近邻
        nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(train_embeddings)

        distances, indices = nbrs.kneighbors(test_embedding)
        # 返回选定的示例

        selected_examples = [train_prompts[i] for i in indices.flatten()]


        if self.only_knowledge_mode==False:
            selected_examples = [train_data[tex] for tex in selected_examples]

        return selected_examples



    def get_knowledge(self, sentences,output_file='./kg/test_triples.txt', is_test=False, label=None):
        # 这个函数返回与输入句子相关的知识
        # 在这个简单的例子中，我们只是返回整个知识库
        # 在实际应用中，你可能需要根据输入句子的内容来选择返回哪些知识
        knowledge = []

        #     创建文件output_file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('')

        # for i, _ in enumerate(sentences):
        print('Matching the knowledge of sentence.')
        for i in tqdm(range(len(sentences)), desc='Processing sentences'):

            if is_test == True:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(str(label[i]))
                    f.write(',')
            # 输入的是一句话，输出的是这句话的所有知识，是一句一句遍历的
            comment = self.select_knn_examples(sentences[i], self.knowledge_base, self.k)


            knowledge.append(comment)

            # print('---------------------------')
            # print(f'sentences[{i}] is here:')
            # print(sentences[i])
            # print('the knowledge of sentences:')
            # print(knowledge[-1])
            # print('---------------------------')


        return knowledge


