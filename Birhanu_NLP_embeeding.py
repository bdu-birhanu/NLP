# import packages (versions---->tf=2.1 and keras=2.3.1, python =3.5)
import numpy as np
from numpy import array
from keras.layers import LSTM, Bidirectional,Input,Dense,Activation,Dropout,Flatten,Embedding
from keras.models import Model
from keras.optimizers import adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
# read a text document line by line from disk to worksapce
def read_dataset():
    text = open('/home/nbm/PycharmProjects/NLP_darm/text_doc.txt', 'r').read()
    lable = open('/home/nbm/PycharmProjects/NLP_darm/lable.txt', 'r').read()
    lines = text.split('\n')
    # Just to find unique words in the document
    word = text.replace("\n", " ")
    word1=word.split(' ')
    no_unique_words=sorted(list(set(word1)))

    mapping = dict((c, i) for i, c in enumerate(no_unique_words))  # map words to an integer index

    sequences = list()
    for line in lines:
        if len(line)==0: continue # ignore if the line is blank
        encoded_seq = [mapping[words] for words in line.split(' ')]
        sequences.append(encoded_seq)
    x1 = [list(line) for line in sequences]
    maxlen = max((len(r)) for r in sequences)
    xtrain = np.asarray([np.pad(r, (0, maxlen - len(r)), 'constant', constant_values=0) for r in x1])

    lang = ['Amharic', 'Afan_oroma', 'Tigrigna']
    digit = [0, 1, 2]
    lables = lable.split('\n')
    y = dict(zip(lang, digit))
    ylable = []  # to store each characte as a list to compute the accuracy
    for index in lables:
        if index == '' or index == '\n': continue
        ylable.append(y[index])
    ytrain=np.asarray(ylable)
    return  xtrain,ytrain,no_unique_words,maxlen


data= read_dataset() # call
xtrain=data[0]
ytrain=data[1]
no_unique_words=data[2]
maxlen=data[3]

#one-hot encoding for lables
def one_hot_encode(sequencex, n_unique):
    encoding = list()
    for value in sequencex:
        vector = [0 for _ in range(n_unique)]
        vector[value] = 1
        encoding.append(vector)
    return array(encoding)
y_onehot = np.array(one_hot_encode(ytrain,3))

xdata, ydata=shuffle(xtrain, y_onehot)

x_train, x_test, y_train, y_test = train_test_split(xdata, ydata, test_size=0.1)

num_class=3
batch_size=16
emb_dim=50
inputs = Input(shape=(maxlen, ))
embedding_layer = Embedding(len(no_unique_words),
                            emb_dim,
                            input_length=maxlen)(inputs)

x =Bidirectional(LSTM(64, return_sequences=True, dropout=0.25))(embedding_layer)
x =Bidirectional(LSTM(64, return_sequences=True, dropout=0.25))(x)
x = Flatten()(x)
x = Dense(32, activation='relu')(x)
predictions = Dense(num_class, activation='softmax')(x)
model = Model(inputs=[inputs], outputs=predictions)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=18)

hist = model.fit(x_train,y_train, batch_size=batch_size,
                 epochs=20,verbose=1, validation_split=0.2,shuffle=True, callbacks=[early_stopping])

y_pred=model.predict(x_test)
predicted = np.argmax(y_pred, axis=1)
score=accuracy_score(np.argmax(y_test, axis=1), predicted)