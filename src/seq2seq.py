import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random


SOS_TOKEN = 0 # 开始符
EOS_TOKEN = 1 # 结束符
clip = 0.5 # 梯度裁剪阈值
teacher_forcing_ratio = 0.5 # 使用teacher forcing的概率
MAX_LENGTH = 15 # 最长句子长度
BEAM_SIZE = 200

class Seq2Seq(nn.Module):
    def __init__(self,input_dim,embedding_dim,hidden_dim,output_dim,n_layers=2,dropout_rate=0.5,use_cuda=False,learning_rate=0.001):
        super(Seq2Seq,self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.use_cuda=use_cuda
        if use_cuda:
            self.encoder = Encoder(input_dim,embedding_dim,hidden_dim,n_layers,dropout_rate,use_cuda).cuda()
            self.decoder = Decoder(output_dim,embedding_dim,hidden_dim,n_layers,dropout_rate,use_cuda).cuda()
        else:
            self.encoder = Encoder(input_dim,embedding_dim,hidden_dim, n_layers, dropout_rate,use_cuda)
            self.decoder = Decoder(output_dim,embedding_dim,hidden_dim, n_layers, dropout_rate,use_cuda)
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(),lr=learning_rate)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(),lr=learning_rate)
        self.encoder_scheduler = optim.lr_scheduler.ExponentialLR(self.encoder_optimizer, 0.96)
        self.decoder_scheduler = optim.lr_scheduler.ExponentialLR(self.encoder_optimizer, 0.96)
        #self.loss_function = nn.CrossEntropyLoss()
        self.loss_function = nn.NLLLoss()



    def forward(self,encoder_input):
        encoder_hidden = self.encoder.init_hidden()
        encoder_output,encoder_hidden = self.encoder(encoder_input,encoder_hidden)
        decoder_hidden = encoder_hidden
        decoder_input = Variable(torch.LongTensor([[SOS_TOKEN]]))
        decoder_output,decoder_hidden = self.decoder(decoder_input,decoder_hidden)
        return decoder_output

    def seq2seq_train(self,input,target):
        input = Variable(torch.LongTensor(input).view(-1,1))
        target = Variable(torch.LongTensor(target).view(-1,1))
        if self.use_cuda:
            input = input.cuda()
            target = target.cuda()
        # 优化器梯度置0
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        # loss用于记录损失函数
        loss = 0
        # 获得input以及target的长度
        input_length = input.size(0)
        target_length = target.size(0)
        # 这部分与forward类似，但由于要用到中间量因此重复
        encoder_hidden = self.encoder.init_hidden()
        encoder_output, encoder_hidden = self.encoder(input, encoder_hidden)
        #for ei in range(input_length):
        #    encoder_output, encoder_hidden = self.encoder(input[ei], encoder_hidden)
        decoder_hidden = encoder_hidden
        decoder_input = Variable(torch.LongTensor([SOS_TOKEN]))
        if self.use_cuda:
            decoder_input = decoder_input.cuda()
        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
        # 判断是否使用teacher forcing，其实teacher_forcing_ratio可视作超参
        use_teacher_forcing = random.random() < teacher_forcing_ratio
        if use_teacher_forcing:
            for di in range(target_length):
                decoder_output,decoder_hidden = self.decoder(decoder_input,decoder_hidden)
                loss += self.loss_function(decoder_output,target[di])
                decoder_input = target[di]
        else:
            # 使用预测值作为下一步输入
            for di in range(target_length):
                #print('Decoder_input',decoder_input)
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                #print('topi:',topi)
                decoder_input = topi.squeeze(1).detach()  # 把取出的结果作为输入
                loss += self.loss_function(decoder_output, target[di])
                if decoder_input.item() == EOS_TOKEN:
                    break

        loss.backward()
        loss = loss.item()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)
        # 参数更新
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        # 返回损失
        return loss / target_length

    def seq2seq_predict(self,input):
        input = Variable(torch.LongTensor(input).view(-1, 1))
        if self.use_cuda:
            input = input.cuda()
        encoder_hidden = self.encoder.init_hidden()
        encoder_output, encoder_hidden = self.encoder(input, encoder_hidden)
        decoder_hidden = encoder_hidden
        decoder_input = Variable(torch.LongTensor([SOS_TOKEN]))
        if self.use_cuda:
            decoder_input = decoder_input.cuda()
        decode_indices = []
        for di in range(MAX_LENGTH):
            # print('Decoder_input',decoder_input)
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            # print('topi:',topi)
            decoder_input = topi.squeeze(1).detach()  # 把取出的结果作为输入
            if decoder_input.item() == EOS_TOKEN:
                break
            else:
                decode_indices.append(topi.item())
        return decode_indices

    def beamsearch_infer(self,sample,index):
        samples = []
        decoder_input = Variable(torch.LongTensor([[sample[0][-1]]]))
        if self.use_cuda:
            decoder_input = decoder_input.cuda()
        sequence,pre_scores, fin_scores ,ave_scores,decoder_hidden = sample
        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
        topk = decoder_output.topk(BEAM_SIZE)
        for index in range(BEAM_SIZE):
            topk_value = topk[0][0][index].item()
            topk_index = int(topk[1][0][index])
            pre_scores += topk_value
            fin_scores = pre_scores - (index * 0.5)
            ave_scores = pre_scores / (len(sequence)+1)
            #ave_scores = fin_scores / (len(sequence) + 1)
            samples.append([sequence+[topk_index],pre_scores,  fin_scores ,ave_scores,decoder_hidden])
        return samples

    def beamsearch(self,input):
        input = Variable(torch.LongTensor(input).view(-1, 1))
        if self.use_cuda:
            input = input.cuda()
        encoder_hidden = self.encoder.init_hidden()
        encoder_output, encoder_hidden = self.encoder(input, encoder_hidden)
        decoder_hidden = encoder_hidden
        decoder_input = Variable(torch.LongTensor([SOS_TOKEN]))
        if self.use_cuda:
            decoder_input = decoder_input.cuda()
        decoder_output,decoder_hidden = self.decoder(decoder_input,decoder_hidden)
        topk =decoder_output.topk(BEAM_SIZE)
        samples = [[] for i in range(BEAM_SIZE)]
        final_samples = []
        for index in range(BEAM_SIZE):
            topk_value = topk[0][0][index].item()
            topk_index = int(topk[1][0][index])
            samples[index] = [[topk_index],topk_value, topk_value - (index * 0.5),topk_value - (index * 0.5),decoder_hidden]
        for _ in range(MAX_LENGTH):
            tmp = []
            for index in range(BEAM_SIZE):
                if samples[index][0][-1]==EOS_TOKEN:
                    tmp.extend([samples[index]])
                else:
                    tmp.extend(self.beamsearch_infer(samples[index],index))
            if tmp == samples:
                break
            tmp.sort(key = lambda x : x[2], reverse = True)
            samples = tmp[:BEAM_SIZE]

        for index in range(MAX_LENGTH):
            final_samples.append(samples[index][0])
        return final_samples



class Encoder(nn.Module):
    def __init__(self,input_dim,embedding_dim,hidden_dim,n_layers=2,dropout_rate=0.5,use_cuda=False):
        super(Encoder,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.use_cuda = use_cuda
        self.embedding = nn.Embedding(input_dim,embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,hidden_dim,num_layers=n_layers,dropout=dropout_rate)

    def forward(self,input,hidden):
        embeds = self.embedding(input).view(len(input),1,-1)
        output,hidden = self.lstm(embeds,hidden)
        return output,hidden

    def init_hidden(self):
        if self.use_cuda:
            hidden = (Variable(torch.zeros(self.n_layers,1,self.hidden_dim)).cuda(),Variable(torch.zeros(self.n_layers,1,self.hidden_dim)).cuda())
        else:
            hidden = (Variable(torch.zeros(self.n_layers, 1, self.hidden_dim)),
                      Variable(torch.zeros(self.n_layers, 1, self.hidden_dim)))
        return hidden

class Decoder(nn.Module):
    def __init__(self,output_dim,embedding_dim,hidden_dim,n_layers=2,dropout_rate=0.5,use_cuda=False):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.use_cuda = use_cuda
        self.embedding = nn.Embedding(output_dim,embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,hidden_dim,num_layers=n_layers,dropout=dropout_rate)
        self.out = nn.Linear(hidden_dim,output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self,input,hidden):
        embeds = self. embedding(input).view(len(input),1,-1)
        #embeds = F.relu(embeds) 这层Relu是optional的，在测试中没有relu的loss更小一些，可能有随机性
        output,hidden = self.lstm(embeds,hidden)
        output = self.out(output[0]) # output[0]的作用其实和squeeze一样，就是去掉一维
        output = self.softmax(output)
        return output,hidden

    def init_hidden(self):
        if self.use_cuda:
            hidden = (Variable(torch.zeros(self.n_layers,1,self.hidden_dim)).cuda(),Variable(torch.zeros(self.n_layers,1,self.hidden_dim)).cuda())
        else:
            hidden = (Variable(torch.zeros(self.n_layers, 1, self.hidden_dim)),
                      Variable(torch.zeros(self.n_layers, 1, self.hidden_dim))) 
        return hidden

