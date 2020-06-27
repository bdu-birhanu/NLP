# NLP
Language Identification(LI) model  using using one-hot character encoding 
  In one-hot character encoding, we use two LSTM networks embedded with the Attention mechanism, whereas the word embedding techniques use two stacked LSTM networks on the top of the embedding layer.
 We use a small dataset consists of 3 Ethiopian languages (Amharic, Tigrigna, and Afan Oromo). The first two languages use the Abugida writing system while the second (Afan Oromo) language uses Latin alphabets.
  text-languages are given in the text_doc.txt file and the corresponding labels are given in lable.txt file

```
# Pre-process data
python preprocess.py

# Train
python train.py

# Test
python test.py
```



