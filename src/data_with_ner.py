import fool
import json
file=r"../data/target.txt"
file_new=r"../data/target_ner.txt"
fp=open(file,"r", encoding='utf-8')
f=open(file_new,"w",encoding="utf-8")
data=[]
for line in fp.readlines():
    line = line.split('\t')
    datat=fool.analysis(line)
    data.append(datat)
    print (line,"\t", datat,"\n")
data_json=json.dumps(data)
f.write(data_json)