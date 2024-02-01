import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


from keras_bert import load_trained_model_from_checkpoint
from keras_bert import Tokenizer
from keras_bert import load_vocabulary


white_path = 'white.csv'
black_path = 'black.csv'

# total_path = 'white_test.csv'

train_path = 'test_dns.csv'


def load_data(path, label):
    data = pd.read_csv('/projects/DNS_o/data/' + path, low_memory=False)
    data = data.dropna(axis=1, how='all')
    data['label'] = label
    return data.sample(n=15000)

def data_concat():
    white = load_data(white_path, label=0)
    print(white.info())
    print(white.describe())
    black = load_data(black_path, label=1)
    print(black.info())
    print(black.describe())
    total = pd.concat([white, black], ignore_index=True)
    # profile_total = total.profile_report(title="Total Dataset")
    # profile_total.to_file(output_file=Path("/home/dell/DNS/Total_report.html"))
    # print(total.head())
    return total

def feature_extract(data):
    columns = ["sourceTransportPort",
               "destinationTransportPort",
               "flowStartSeconds",
               "flowEndSecond",
               "DNSQueryName",
               "label"
               ]

    train_data = data[columns]
    return train_data

def calc_ent(y):
    """
        信息熵计算函数，传入的对象为series中的一个元素
    """
    strs=str(y)
    x=[]
    for ch in strs:
        #print(ch)
        x.append(ch)
    x=np.array(x)

    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp
    #print(ent)
    return ent

def begin_with_num(domainname):
    first_label = str(domainname).split('.')[0]
    first_character = first_label[0]
    return 1 if first_character.isdigit() else 0

#读取数据
def prehandle(data):
    """

    :param data: csv文件
    :return: 连域名带特征的dataframe
    """
    domain = data.copy()

    #1.	Total count of characters in FQDN（全限定域名中的字符总数）：
    domain['Total count of characters in FQDN']=domain['DNSQueryName'].str.len()

    #2.	count of characters in sub-domain（子域名中的字符总数）
    domain['subdomain_len']=domain['DNSQueryName'].str.split('.',expand=True)[0].str.len()

    #3.	count of uppercase characters（大写字母个数）
    domain['upper_count']=domain['DNSQueryName'].str.count('[A-Z]')

    #4.	count of numerical characters（数字个数）
    domain['number_count']=domain['DNSQueryName'].str.count('[0-9]')

    #5. first character is num(数字开头)
    domain['begin_with_num'] = domain['DNSQueryName'].apply(begin_with_num)

    #6.	Entropy(域名的信息熵)
    #定义计算信息熵的函数：计算Infor(D)
    domain['entropy'] = domain['DNSQueryName'].apply(calc_ent)



    #7.	number of labels(域名的部分数)
    # 如www.scholar.google.com
    # label数为4。
    domain['label_count']=domain['DNSQueryName'].str.split('.',expand=True).count(axis=1)

    #计算每一部分label的长度
    subdomain = domain['DNSQueryName'].str.split('.',expand=True)
    for label, content in subdomain.iteritems():
        subdomain[label]=content.str.len()


    #8.	maximum label length（最大部分长度）
    domain['max_label_length']=subdomain.max(axis=1)

    #9.	average label length（平均部分长度）
    domain['avg_label_length']=subdomain.mean(axis=1)


    return domain

def standard(data):
    '''
    标准化

    :param data: df
    :return:
    '''
    x_np = data.values
    x_np[x_np == np.inf] = 1.
    # x_np[np.isnan(x_np)] = 0.
    x_np[pd.isna(x_np)] = 0.

    # stand = StandardScaler()
    # X = np.concatenate([one_hot_x, stand.fit_transform(x_np)], axis=1)
    # X = stand.fit_transform(x_np)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
    return x_np

def main():
    tqdm.pandas(desc='Extracting features ...')
    total = data_concat()

    data = feature_extract(total)
    # data = data.drop('DNSQueryName',axis=1)
    feature_domain = prehandle(data)
    

    print(feature_domain.describe())
    print(feature_domain.info())

    # feature_domain.drop('DNSQueryName', axis=1, inplace=True)
    # feature_domain=feature_domain.drop('DNSQueryName', axis=1)

    feature_domain.to_csv('/projects/DNS_o/bert_dns/' + train_path, index=False, header=True)
if __name__ == '__main__':
    main()





