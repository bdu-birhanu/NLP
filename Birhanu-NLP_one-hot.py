# import packages (versions---->tf=2.1 and keras=2.3.1, python 3.5)
import numpy as np
from numpy import array
from keras.layers import LSTM,Bidirectional,Input,Dense,Activation,Dropout,Flatten,Dot
from keras.models import Model
from keras.optimizers import adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
#==============================================================================
# read a text document from disk to worksapce, preprocess( like lower case, remove white space, padding, ..etc)
def read_dataset():
    text = open('/home/nbm/PycharmProjects/NLP_darm/text_doc.txt', 'r').read()
    lable = open('/home/nbm/PycharmProjects/NLP_darm/lable.txt', 'r').read()
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

# function for one-hot-encoding
def one_hot_encode(sequencex, n_unique):
    encoding = list()
    for value in sequencex:
        vector = [0 for _ in range(n_unique)]
        vector[value] = 1
        encoding.append(vector)
    return array(encoding)

#Function for attention layer(input---> all hidden layers, output---> context_vector)
dot_prod = Dot(axes = 1)
def attention(lstm_out):
    hidden_state = lstm_out
    score = Dense(61, activation='tanh', name='attention_score_vec')(hidden_state)
    attention_weights = Activation('softmax', name='attention_weight')(score)
    context_vector = dot_prod([attention_weights,hidden_state])
    return context_vector

#Network parameter and input to the model
data= read_dataset() # call
xtrain=data[0]
ytrain=data[1]
unique_chars=data[2]
maxlen=data[3]
x_one = []
for i in xtrain:
    j = one_hot_encode(i,len(unique_chars))
    x_one.append(j)

x_onehot = np.array(x_one)
y_onehot = np.array(one_hot_encode(ytrain,3))

xdata, ydata=shuffle(x_onehot, y_onehot)
x_train, x_test, y_train, y_test = train_test_split(xdata, ydata, test_size=0.1)

#Network parameter
num_class=3
rnn_size = 128
act= 'relu'
batch_size=5


#main model( input--> input data, output--model)
def main_model(input_data):
    lstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.25))(input_data)
    lstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.25))(lstm_1)
    context=attention(lstm_2)
    fc = Flatten()(context) #flatten and pass on to the Dense output layer.
    outputs = Dense(num_class, activation='softmax')(fc)

    model=Model(inputs=input_data, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# train the model=============================
input_data = Input(name='input', shape=(maxlen,len(unique_chars)))
model=main_model(input_data)
model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=5)
hist = model.fit(x_train,y_train, batch_size=batch_size,
                 epochs=20, verbose=1, validation_split=0.2,shuffle=True, callbacks=[early_stopping])

#testing the model
y_pred=model.predict(x_test)
predicted = np.argmax(y_pred, axis=1)
score=accuracy_score(np.argmax(y_test, axis=1), predicted)