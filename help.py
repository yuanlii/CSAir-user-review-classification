import jieba
import jieba.posseg as pseg
import jieba.analyse
import re

def get_stopwords(stopwords_file_path):  
    '''read stopwords (txt file) as list'''
    stopwords = [line.strip() for line in open(stopwords_file_path, 'r', encoding='utf-8').readlines()]  
    return stopwords

# def get_tokenized_sents(stopwords, sents):   
def get_tokenized_sent(sent):   
    '''take in sentences, return tokenized sentences ''' 
    # sents_tokenized = []    
    stopwords = get_stopwords('../Source_Data/stopwords.txt')
    # for sent in sents:

    # remove stopwords
    tokenized_sent = ' '.join(word for word in jieba.cut(sent.strip()) if word not in stopwords)
    # remove digits
    tokenized_sent = re.sub(r'\d+','',tokenized_sent)
    # remove punctuation
    tokenized_sent = re.sub(r'[^\w\s]','', tokenized_sent)
    # remove non-chinese characters
    # match all Chinese words
    re_words = re.compile(u"[\u4e00-\u9fa5]+")
    res = re.findall(re_words, tokenized_sent)
    if res:
        valid_tokenized_sent = ' '.join([r for r in res])
        # sents_tokenized.append(valid_tokenized_sent)
    return tokenized_sent

