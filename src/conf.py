import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # seq2seq参数
    parser.add_argument('-b', '--batch_size', help='seq2seq batch_size', type=int, default='32')
    parser.add_argument('-es', '--embedding_size',help='seq2seq embedding_size',type=int,default='128')
    parser.add_argument('-hs', '--hidden_size',help='seq2seq hidden_size',type=int,default='128')
    parser.add_argument('-v', '--vocab_size',help='seq2seq vocab_size',type=int,default='111')
    parser.add_argument('-ml', '--max_length',help='max sentence length',type=int,default='15')
    parser.add_argument('-l', '--learning_rate', help='seq2seq learning_rate', type=float, default='0.001')
    parser.add_argument('-n', '--num_layers', help='seq2seq num_layers', type=int, default='2')
    parser.add_argument('-r', '--rnn_size', help='seq2seq rnn_size', type=int, default='128')
    parser.add_argument('-e', '--epochs', help='seq2seq epochs', type=int, default='5000')
    parser.add_argument('-d', '--display_step', help='seq2seq display_step', type=int, default='50')
    parser.add_argument('-p', '--module_path_with_none', help='seq2seq runs/seq2seq', type=str,
                        default='../model/seq2seq.model')
    parser.add_argument('-lip', '--letter_index_path_with_none', help='seq2seq letter to index', type=str,
                        default='../model/letter_index.pickle')
    parser.add_argument('-ilp', '--index_letter_path_with_none', help='seq2seq letter to index', type=str,
                        default='../model/index_letter.pickle')
    parser.add_argument('-p_pos', '--module_path_with_pos', help='seq2seq runs/seq2seq', type=str,
                        default='../model/seq2seq_pos.model')
    parser.add_argument('-lip_pos', '--letter_index_path_with_pos', help='seq2seq letter to index', type=str,
                        default='../model/letter_index_pos.pickle')
    parser.add_argument('-ilp_pos', '--index_letter_path_with_pos', help='seq2seq letter to index', type=str,
                        default='../model/index_letter_pos.pickle')
    parser.add_argument('-p_word', '--module_path_with_word', help='seq2seq runs/seq2seq', type=str,
                        default='../model/seq2seq_word.model')
    parser.add_argument('-lip_word', '--letter_index_path_with_word', help='seq2seq letter to index', type=str,
                        default='../model/letter_index_word.pickle')
    parser.add_argument('-ilp_word', '--index_letter_path_with_word', help='seq2seq letter to index', type=str,
                        default='../model/index_letter_word.pickle')
    parser.add_argument('-p_score', '--module_path_with_score', help='seq2seq runs/seq2seq', type=str,
                        default='../model/seq2seq_score.model')
    parser.add_argument('-lip_score', '--letter_index_path_with_score', help='seq2seq letter to index', type=str,
                        default='../model/letter_index_score.pickle')
    parser.add_argument('-ilp_score', '--index_letter_path_with_score', help='seq2seq letter to index', type=str,
                        default='../model/index_letter_score.pickle')
    parser.add_argument('-k', '--topk', help='seq2seq top k answers', type=int, default='10')
    parser.add_argument('-bs', '--beam_size', help='decoder beam_size', type=int, default='200')
    parser.add_argument('-m', '--mode', help='current mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('-tm','--template_method',help="question template generating method", type=str , default='normal' , choices=['num_word',"score_method","num","normal"])
    parser.add_argument('-T','--transform_method',help="method for template transform",type=str,default='seq2seq',choices=["seq2seq","retrivel","rl"])
    parser.add_argument('-dp', '--dropout', help='dropout', type=float, default='0.8')
    parser.add_argument('-g', '--gpu', help='gpu mode', type=int, default=1)

    args = parser.parse_args()
    return args
