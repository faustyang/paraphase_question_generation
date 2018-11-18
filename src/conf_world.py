import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # seq2seq参数
    parser.add_argument('-b', '--batch_size', help='seq2seq batch_size', type=int, default='32')
    parser.add_argument('-es', '--embedding_size',help='seq2seq embedding_size',type=int,default='128')
    parser.add_argument('-hs', '--hidden_size',help='seq2seq hidden_size',type=int,default='128')
    parser.add_argument('-v', '--vocab_size',help='seq2seq vocab_size',type=int,default='1443')
    parser.add_argument('-ml', '--max_length',help='max sentence length',type=int,default='15')
    parser.add_argument('-l', '--learning_rate', help='seq2seq learning_rate', type=float, default='0.001')
    parser.add_argument('-n', '--num_layers', help='seq2seq num_layers', type=int, default='2')
    parser.add_argument('-r', '--rnn_size', help='seq2seq rnn_size', type=int, default='128')
    parser.add_argument('-e', '--epochs', help='seq2seq epochs', type=int, default='5000')
    parser.add_argument('-d', '--display_step', help='seq2seq display_step', type=int, default='50')
    parser.add_argument('-p', '--module_path', help='seq2seq runs/seq2seq', type=str,
                        default='../model/seq2seq_word.model')
    parser.add_argument('-lip', '--letter_index_path', help='seq2seq letter to index', type=str,
                        default='../model/letter_index_word.pickle')
    parser.add_argument('-ilp', '--index_letter_path', help='seq2seq letter to index', type=str,
                        default='../model/index_letter_word.pickle')
    parser.add_argument('-k', '--topk', help='seq2seq top k answers', type=int, default='10')
    parser.add_argument('-bs', '--beam_size', help='decoder beam_size', type=int, default='200')
    parser.add_argument('-m', '--mode', help='current mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('-dp', '--dropout', help='dropout', type=float, default='0.8')
    parser.add_argument('-g', '--gpu', help='gpu mode', type=int, default=1)

    args = parser.parse_args()
    return args
