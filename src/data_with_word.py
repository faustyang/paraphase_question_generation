# data.py用于读取文件
import numpy as np
import pickle
import jieba.posseg as pseg
from conf_pos import get_args
class data():
    def __init__(self,args):
        self.args = args
        self.data_process()

    def data_process(self):
        data = []
        data_list=[]
        f = open('../data/train.txt', 'r', encoding='utf-8')
        for line in f.readlines():
            line = self.clean_str(line) # 对字符串去除空格，
            line = line.split('\t')
            try: # 避免一些无法处理的情形
                x1 = line[0]
                x_1=[]
                x2 = line[1]
                x_2=[]
                word1s = pseg.cut(x1)
                for w in word1s:
                    if w.flag == 'n':
                        x_1.append('*')
                    else:
                        x_1.append(w.word)
                word2s = pseg.cut(x2)
                for w in word2s:
                    if w.flag == 'n':
                        x_2.append('*')
                    else:
                        x_2.append(w.word)
                data.append((x_1, x_2))
                #data.append((x_2,x_1))
                query=x_1+x_2
                squence=''
                for i in query:
                    squence=squence+i
                data_list.append(squence)
                #print(squence)
            except:
                #print(line)
                pass

        source_data, target_data = zip(*data)  # 这个写法感觉很高级
        self.index_letter_map, self.letter_index_map = self.extract_character_vocab(source_data + target_data)  # 建立映射表

        f = open(self.args.letter_index_path, 'wb')
        pickle.dump(self.letter_index_map, f)
        f.close()
        f = open(self.args.index_letter_path, 'wb')
        pickle.dump(self.index_letter_map, f)
        f.close()
        np.random.shuffle(data)  # 打乱数据

        source_data, target_data = zip(*data)
        print(len(set(data_list)))
        print(len(data))
        self.source_index = [[self.letter_index_map.get(letter, self.letter_index_map['<UNK>'])
                         for letter in line] for line in source_data]
        self.target_index = [[self.letter_index_map.get(letter, self.letter_index_map['<UNK>'])
                         for letter in line] + [self.letter_index_map['<EOS>']] for line in target_data]  # 加入结束符，开始符和原句开始符呢
        
        print('数据总量:{},词汇表大小:{}'.format(len(source_data), len(self.letter_index_map)))

    def clean_str(self,string):
        # 去除空格,字母需要变为大写（不太肯定大写的意义）
        string = string.replace(' ', '').strip().upper()
        import re
        mark=["/","?",",","(",")","[","]",'{',"}","：",".","？","，","（","）","！","%","|","<",">","“","”"]
        for item in mark:
            string=string.replace(item,"")
        pattern_number=re.compile(r'\d+')
        pattern_char=re.compile(r'[A-Z]+')
        number=pattern_number.findall(string)
        char=pattern_char.findall(string)
        number= sorted(number,key=lambda x: len(x),reverse=True)
        for item in number:
            string=string.replace(item,"N")
        for item in char:
            string=string.replace(item,'C')


        if '{品牌名}' in string:
            string = string.replace('{品牌名}','') # 把品牌名直接变成星号
        if '{产品名}' in string:
            string = string.replace('{产品名}','') # 把品牌名直接变成星号
        if '(商户编号)' in string:
            string = string.replace('(商户编号)','') # 把品牌名直接变成星号        

        return string

    def extract_character_vocab(self,data):
        # 建立映射表
        set_words = []
        special_words = ['<SOS>', '<EOS>', '<PAD>', '<UNK>']  # 分别对应开始符，结束符，填充符，位置符和未知符
        for line in data:
            for character in line:
                if character not in set_words:
                    set_words.append(character)
        # 把四个特殊字符添加进词典
        index_letter_map = {idx: word for idx, word in enumerate(special_words + set_words)}
        letter_index_map = {word: idx for idx, word in index_letter_map.items()}
        return index_letter_map, letter_index_map

    # 文本转换为下标
    def text2index(self,text):
        return [self.letter_index_map.get(word, self.letter_index_map['<UNK>']) for word in text]

    def index2text(self,indices):
        return [self.index_letter_map.get(index, '') for index in indices]



    def data_split(self):
        data = []
        f = open('../data/train.txt', 'r', encoding='utf-8')
        r1 = open('../data/source_word.txt','w',encoding='utf-8')
        r2 = open('../data/target_word.txt','w',encoding='utf-8')
        r3=open('../data/dictionary_word.txt','w',encoding="utf-8")
        dic={}
        for line in f.readlines():
            line = self.clean_str(line) # 对字符串去除空格
            line = line.split('\t')
            try: # 避免一些无法处理的情形
                
                x1 = line[0]
                x_1=[]
                x2 = line[1]
                x_2=[]
                word1s = pseg.cut(x1)
                for w in word1s:
                    if w.flag == 'n':
                        x_1.append('*')
                    else:
                        x_1.append(w.word)
                word2s = pseg.cut(x2)
                for w in word2s:
                    if w.flag == 'n':
                        x_2.append('*')
                    else:
                        x_2.append(w.word)
                data.append((x_1, x_2))
                sentence=x_1+x_2
                for word in sentence:
                    if word not in dic:
                        r3.write(word)
                        r3.write('\n')
                        dic[word]=len(dic)
                r1.write(str(x_1))
                r1.write('\n')
                r2.write(str(x_2))
                r2.write('\n')
            except:
                #print(line)
                pass



if __name__=='__main__':
    args=get_args()
    print('参数列表:{}'.format(args))
    data = data(args)

    data.data_split()
