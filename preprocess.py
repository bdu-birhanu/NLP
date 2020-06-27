import numpy as np
from sklearn.utils import shuffle
from numpy import array
def read_dataset():
    text = open('./text_doc.txt', 'r').read()
    lable = open('./lable.txt', 'r').read()
    text=text.lower()
    lines = text.split('\n')
    # find and list-out unique characters in the texts
    unque_chars = sorted(list(set(text)))
    mapping = dict((c, i) for i, c in enumerate(unque_chars))  # map character to an integer index
    #mapping1 = dict((i, c) for i, c in enumerate(chars))  # mapp index back to character
    sequences = list()
    for line in lines:
        if len(line)==0: continue # ignore if the line is blank
        encoded_seq = [mapping[char] for char in line]
        sequences.append(encoded_seq)
    xs = [list(line) for line in sequences]
    maxlen = max((len(r)) for r in sequences)
    #padding each sequence with 0 at the end
    xtrain = np.asarray([np.pad(r, (0, maxlen - len(r)), 'constant', constant_values=0) for r in xs])

    # 0=Amahric ,1= Afan Oromo, 2=Tigrigna
    lang = ['Amharic', 'Afan_oroma', 'Tigrigna']
    digit = [0, 1, 2]
    lables = lable.split('\n')
    y = dict(zip(lang, digit))
    ylable = []  # to store each characte as a list to compute the accuracy
    for index in lables:
        if index == '' or index == '\n': continue
        ylable.append(y[index])
    ytrain=np.asarray(ylable)
    return  xtrain,ytrain,unque_chars,maxlen

def one_hot_encode(sequencex, n_unique):
    encoding = list()
    for value in sequencex:
        vector = [0 for _ in range(n_unique)]
        vector[value] = 1
        encoding.append(vector)
    return array(encoding)

def encoded_vector():
    data = read_dataset()
    xtrain = data[0]
    ytrain = data[1]
    unique_chars = data[2]
    maxlen = data[3]

    x_one = []
    for i in xtrain:
        j = one_hot_encode(i,len(unique_chars))
        x_one.append(j)

    x_onehot = np.array(x_one)
    y_onehot = np.array(one_hot_encode(ytrain,3))

    xdata, ydata=shuffle(x_onehot, y_onehot)
    return (xdata,ydata, maxlen,unique_chars)

if __name__=="__main__":
    print ("...data loading...")
    data_loaded=encoded_vector()

    print(str(len(data_loaded[0]))+ " Samples with their Corresponding lables are loaded")
    print(str( "The maximum sequnce length is = "+ str(data_loaded[2])))
    print("one-hot encoding is completed")



