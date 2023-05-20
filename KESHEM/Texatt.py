from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class CoAttention(nn.Module):
    def __init__(self, latent_dim=768):
        super(CoAttention, self).__init__()

        self.linearq = nn.Linear(latent_dim, latent_dim)
        self.lineark = nn.Linear(latent_dim, latent_dim)
        self.linearv = nn.Linear(latent_dim, latent_dim)

    def forward(self, sentence_rep, comment_rep):
        query = self.linearq(sentence_rep)
        key = self.lineark(comment_rep)
        value = self.linearv(comment_rep)

        alpha_mat = torch.matmul(query, key.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, value).squeeze(1)

        return x


class BertForSequenceClassification(nn.Module):
    def __init__(self, num_labels=2, device='cuda'):  # Change number of labels here.
        super(BertForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(r'E:\model\white_model\bert')
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.latent_dim = self.bert.config.hidden_size
        self.coattention = CoAttention(self.latent_dim)
        self.classifier = nn.Linear(self.latent_dim * 2, num_labels)
        self.device = device
        nn.init.xavier_normal_(self.classifier.weight)
        self.tf_encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=2)
        self.tf_encoder = nn.TransformerEncoder(self.tf_encoder_layer, num_layers=2)

    def forward_once(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        return pooled_output

    def forward1(self, batch_seqs, batch_seq_masks, batch_knowledges, labels=None):
        # forward pass of input 1
        output1 = self.forward_once(batch_seqs, batch_seq_masks)
        output1 = torch.unsqueeze(output1, 1)
        # forward pass of input 2
        knowledges = batch_knowledges.cpu()
        knowledges = knowledges.numpy().tolist()
        tmp = []

        for each in knowledges:
            print(each)
            batch_seqs_k = [i[0] for i in each]
            batch_seq_masks_k = [i[1] for i in each]
            t_batch_seqs_k = torch.tensor(batch_seqs_k, dtype=torch.long).to(self.device)
            t_batch_seq_masks_k = torch.tensor(batch_seq_masks_k, dtype=torch.long).to(self.device)
            output2 = self.forward_once(t_batch_seqs_k, t_batch_seq_masks_k)
            tmp.append(output2)
        k_emb = torch.stack(tmp, dim=0)
        k_emb = self.tf_encoder(k_emb)
        k_emb = self.dropout(k_emb)
        pooled_output = self.coattention(output1, k_emb)

        pooled_output = torch.cat([output1.squeeze(1), pooled_output], dim=1)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

    def forward(self, batch_seqs, batch_seq_masks, batch_comments, labels=None):
        # forward pass of input 1
        output1 = self.forward_once(batch_seqs, batch_seq_masks)
        output1 = torch.unsqueeze(output1, 1)
        # forward pass of input 2
        comments = batch_comments.cpu()
        comments = comments.numpy().tolist()
        tmp = []
        print(comments)
        for each in comments:

            batch_seqs_k = [i[0] for i in each]
            batch_seq_masks_k = [i[1] for i in each]
            t_batch_seqs_k = torch.tensor(batch_seqs_k, dtype=torch.long).to(self.device)
            t_batch_seq_masks_k = torch.tensor(batch_seq_masks_k, dtype=torch.long).to(self.device)
            output2 = self.forward_once(t_batch_seqs_k, t_batch_seq_masks_k)
            tmp.append(output2)
        k_emb = torch.stack(tmp, dim=0)
        k_emb = self.tf_encoder(k_emb)
        k_emb = self.dropout(k_emb)
        pooled_output = self.coattention(output1, k_emb)

        pooled_output = torch.cat([output1.squeeze(1), pooled_output], dim=1)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True




def collate_fn(batch):
    sentences, comments, labels = zip(*batch)
    sentences = [torch.tensor(sent) for sent in sentences]
    comments = [torch.tensor(comm) for comm in comments]
    sentences = pad_sequence(sentences, batch_first=True, padding_value=0)
    comments = pad_sequence(comments, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    return sentences, comments, labels

def generate_data(batch_size=4):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    sentences = ['The cat sat on the mat.', 'The dog ate my homework.'] * 100
    comments = ['The cat is on the mat.', 'The dog ate my homework.'] * 10
    labels = [0, 1] * 110
    sentences = [tokenizer.encode(sent) for sent in sentences]
    comments = [tokenizer.encode(sent) for sent in comments]
    data = list(zip(sentences, comments, labels))
    data = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return data

# 生成训练数据，实例化模型，训练模型
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertForSequenceClassification().to(device)
    model.freeze_bert_encoder()
    data = generate_data()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(10):
        for batch_seqs, batch_knowledges, batch_labels in data:
            print('------------------')
            print(batch_seqs, batch_knowledges, batch_labels)
            print('------------------')

            batch_seqs = torch.tensor(batch_seqs, dtype=torch.long).to(device)
            batch_knowledges = torch.tensor(batch_knowledges, dtype=torch.long).to(device)

            batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(device)
            loss = model(batch_seqs, batch_knowledges, batch_labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(loss.item())

    return model

# 生成测试数据，测试模型
def test(model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    data = generate_data()
    for batch_seqs, batch_knowledges, batch_labels in data:
        batch_seqs = torch.tensor(batch_seqs, dtype=torch.long).to(device)
        batch_knowledges = torch.tensor(batch_knowledges, dtype=torch.long).to(device)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(device)
        logits = model(batch_seqs, batch_knowledges)
        print(logits)
        print(batch_labels)

# 运行代码
if __name__ == '__main__':
    model = train()
    test(model)
