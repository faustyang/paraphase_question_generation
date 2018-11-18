from datetime import timedelta
from seq2seq import Seq2Seq
from data import data
from conf import get_args
import time
import torch


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def start_train(args,data):
    if args.gpu == 0:
        use_cuda = False
    else:
        use_cuda = True
    MAX_ITER = 5000
    epochs = 50
    print_every = 5000
    plot_every = 100
    start_time = time.time()
    plot_losses = [] # 存储loss用来绘图
    print_loss_total = 0
    input_dim = args.vocab_size
    output_dim = args.vocab_size
    embedding_dim = args.embedding_size
    hidden_dim = args.hidden_size
    learning_rate = args.learning_rate
#    input_word = ["怎么获得立减券",'手机银行登录密码要去哪里修改','修改缴费号码可以修改吗','儒商卡放进建行atm机拿不出来了','手机银行账户信息查询']

#    input_text = [data.text2index(word) for word in input_word]
    rfile=open('../data/questions_viewer.txt','w',encoding='utf-8')
    if use_cuda:
        seq2seq_model = Seq2Seq(input_dim,embedding_dim,hidden_dim,output_dim,use_cuda=use_cuda,learning_rate=learning_rate).cuda()
    else:
        seq2seq_model = Seq2Seq(input_dim,embedding_dim,hidden_dim,output_dim,use_cuda=use_cuda,learning_rate=learning_rate)
    seq2seq_model.train()
    for epoch in range(epochs):

        #valid_targets, valid_sources, valid_targets_lengths, valid_source_lengths = data.get_valid_batch()
        #valid_loss = seq2seq_model.seq2seq_train(valid_targets, valid_sources)
        for iter,(source, target) in enumerate(zip(data.source_index,data.target_index)):
            #print(source,target)
            loss = seq2seq_model.seq2seq_train(source,target)
            print_loss_total += loss
            if iter % print_every == 0 and iter!=0:
                seq2seq_model.encoder_scheduler.step()
                seq2seq_model.decoder_scheduler.step()
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                time_dif = get_time_dif(start_time)
                print('Epoch {:>3}/{} - Training Loss: {:>6.6f}  Time:{}'.format(epoch,epochs,print_loss_avg,time_dif))
        torch.save(seq2seq_model.state_dict(), args.module_path)
'''
        for item in input_text:
            text=item
            predict_indices = seq2seq_model.seq2seq_predict(text)
            predict_result = data.index2text(predict_indices)
            print(predict_result)
            rfile.write("".join(predict_result))
            rfile.write('\n')
'''


def start_test(args, data):
    # 输入一句话
    if args.gpu == 0:
        use_cuda = False
    else:
        use_cuda = True
    input_dim = args.vocab_size
    output_dim = args.vocab_size
    embedding_dim = args.embedding_size
    hidden_dim = args.hidden_size
    args.mode = 'test'
    while True:
        input_word = input('请输入您的问题:')
        text = data.text2index(input_word)
        if use_cuda:
            seq2seq_model = Seq2Seq(input_dim, embedding_dim, hidden_dim, output_dim, use_cuda=use_cuda).cuda()
        else:
            seq2seq_model = Seq2Seq(input_dim, embedding_dim, hidden_dim, output_dim, use_cuda=use_cuda)
        seq2seq_model.load_state_dict(torch.load(args.module_path))
        seq2seq_model.eval()
        #predict_indices = seq2seq_model.seq2seq_predict(text)
        #predict_result = data.index2text(predict_indices)
        #print(predict_result)
        predict_sample_indices = seq2seq_model.beamsearch(text)
        for predict_indices in predict_sample_indices:
            predict_result = data.index2text(predict_indices)
            print("".join(predict_result[:-1]))
        #rerank_by_bleu(input_word, results)


if __name__=='__main__':
    args=get_args()
    data=data(args)
    data.data_split()
    args.vocab_size=data.vocab_size
    print('参数列表:{}'.format(args))
    if args.mode=='train':
        start_train(args,data)
    else:
        start_test(args,data)

