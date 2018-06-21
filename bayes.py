import pandas as pd
import jieba
filename='G:\\python_work\\80w.txt'
data=pd.read_csv(filename,encoding='utf-8',sep='\t',header=None)
data['分词短信']=data[2].apply(lambda x:' '.join(jieba.cut(x)))
x=data['分词短信'].values
y=data[1].values
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.1)
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
vertorizer=CountVectorizer()
x_train_termcounts=vertorizer.fit_transform(train_x)
tfidf_transformer=TfidfTransformer()
x_train_tfidf=tfidf_transformer.fit_transform(x_train_termcounts)
from sklearn.naive_bayes import GaussianNB,MultinomialNB
classifier=MultinomialNB().fit(x_train_tfidf,train_y)
x_input_trrmcounts=vertorizer.transform(test_x)
x_input_tfidf=tfidf_transformer.transform(x_input_trrmcounts)
predicted_categories=classifier.predict(x_input_tfidf)
from sklearn.metrics import accuracy_score
print(accuracy_score(test_y,predicted_categories))
'''
categrory_map={0:'正常',1:'垃圾'}
for sentence,categroty,real in zip(test_x[:20],predicted_categories[:20],test_y[:20]):
    print('\n短信内容:',sentence,'\nPredicted:',categrory_map[categroty],'\n真实性:',categrory_map[real])
'''
testfile='G:\\python_work\\20w.txt'
pc_data=pd.read_csv(testfile,encoding='utf-8',sep='\t',header=None)
fen_x=[]
for k in range(len(pc_data[1])):
    coments = ''
    coments=coments+str(pc_data.iloc[k,1]).strip()
    fen_ci=' '.join(jieba.lcut(coments))
    fen_x.append(fen_ci)
fen_x=pd.DataFrame(fen_x)
pc_x=pd.concat([pc_data,fen_x],axis=1)
pc_x.columns=['号码','短信','分词']
pd_x=pc_x['分词'].values
pc_x_termcounts=vertorizer.transform(pd_x)
pc_tfidf_transformer=tfidf_transformer.transform(pc_x_termcounts)
pc_predicted=classifier.predict(pc_tfidf_transformer)
pc_predicted=pd.DataFrame(pc_predicted)
pc_predicted=pc_predicted.replace(0,"垃圾").replace(1,"正常")
result=pd.concat([pc_data[1],pc_predicted],axis=1)
result.columns=['短信','判断']
print(result)






