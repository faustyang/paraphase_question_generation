# -*- coding: utf-8 -*- 
import torch
import numpy as np
import nltk
import jieba
from numpy import sort
from threading import Thread
from seq2seq import Seq2Seq
from train import start_train
from data import data
import sys
    
class seq2seq_module():
    def __init__(self,args,seq2seq_data):
        self.args=args
        self.data=seq2seq_data 
        self.load_module()
    
    def load_module(self):
        self.args.mode='test'
        self.module = Seq2Seq(self.args.vocab_size,self.args.embedding_size,self.args.hidden_size,self.args.vocab_size)
        self.module.load_state_dict(torch.load("../model/seq2seq.model"))
        print('seq2seq模型加载完毕,可对外提供服务')
        
    def predict(self):
        #quest=client_params["params"]["question"]
        quest = sys.stdin.readline().strip()
        print('seq2seq预测问题:{}'.format(quest))
        text = self.data.source_to_seq(quest)
        predicts = self.module(text)
        seq2seq_answer=[]
        for index,answer in enumerate(predicts[0]):
            #print(answer)
            result='' 
            for i in answer:
                if self.data.word_int_to_letter[i] == '<EOS>':
                    break
                result+=self.data.word_int_to_letter[i]
            seq2seq_answer.append(result)
        result = self.rerank_by_bleu(quest,seq2seq_answer)
        return result
              
    #对预测结果基于bleu评测指标进行重新排序
    def rerank_by_bleu(self,source,targets):
        new_source=self.data.remove_stop_words(jieba.lcut(source))
        result=[]
        writed_result=set()
        writed_result.add(new_source)
        for index,target in enumerate(targets):
            new_target=self.data.remove_stop_words(jieba.lcut(target))
            #生成模型给的答案的也是可以作为一定的得分的
            score1=(len(targets)-index)/len(targets)
            #bleu的得分
            score2=nltk.translate.bleu_score.sentence_bleu([new_target],new_source)
            #相似性
            final_score=score1*0.7+score2*0.3
            if target not in writed_result and len(target)>=0.5*len(source) and self.data.get_min_editdis(target,writed_result)>2:
                writed_result.add(target)
                result.append((target,final_score,score1,score2)) 
        result=sorted(result,key=lambda x:x[1],reverse = True)
        final_result=[ans[0] for ans in result[0:self.args.topk]]
        print('按照最终结果排序')
        for ans in result[0:self.args.topk]:
            print(ans)
        '''
        print('按照文本生成结果排序')
        result=sorted(result,key=lambda x:x[2],reverse = True)
        for ans in result[0:self.args.topk]:
            print(ans) 
        '''           
        return final_result

    def train(self):
        #启动一个线程开始重新训练
        t=Thread(target=self.retrain)
        t.start()
        
    def retrain(self):
        print('开始重新训练')
        self.args.mode='train'
        #数据可能变化
        self.data=data()
        start_train(self.args,self.data)
        self.load_module()

