import pandas as pd
import numpy as np


from keras.models import Model
from keras.layers import Dense,Embedding,Input,LSTM,Bidirectional,GlobalMaxPool1D,Dropout
from keras.preprocessing import text,sequence
from keras.callbacks import EarlyStopping,ModelCheckpoint

#read datas
data_train=pd.read_table("labeledTrainData.tsv")
data_test=pd.read_table("testData.tsv")
sample_submission=pd.read_table("sampleSubmission.csv")


max_features=20000
maxlen=100

## collect the sentences , fill by menanignles term
list_sentences_train=data_train["review"].fillna("XSa652").values
list_sentences_test=data_test["review"].fillna("XSa652").values
y=data_train["sentiment"].values



tokenizer=text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train=tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test=tokenizer.texts_to_sequences(list_sentences_test)
X_t=sequence.pad_sequences(list_tokenized_train,maxlen=maxlen)
x_te=sequence.pad_sequences(list_tokenized_test,maxlen=maxlen)


def get_model():
	embed_size=128
	inp = Input(shape=(maxlen,))
	x=Embedding(max_features,embed_size)(inp)
	x=Bidirectional(LSTM(50,return_sequences=True))(x)
	x=GlobalMaxPool1D()(x)
	x=Dropout(0.1)(x)
	x= Dense(50,activation="relu")(x)
	x=Dropout(0.1)(x)
	output = Dense(1, activation='sigmoid')(x)
	model=Model(inputs=inp,outputs=output)	
	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	return model
model=get_model()
batch_size=32
epochs=1
file_path="weights_best.hdf5"

checkpoint=ModelCheckpoint(file_path,monitor="val_loss",verbose=32,save_best_only=True,mode="min")

early = EarlyStopping(monitor="val_loss", mode="min", patience=20)

callbacks_list = [checkpoint, early] #early
model.fit(X_t, y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)


model.load_weights(file_path)

y_test = model.predict_classes(x_te)

e=[]
for i in y_test:
	if i>0.5:
		e.append(1)
	else:
		e.append(0)

a=pd.DataFrame({"id":test['id'],"sentiment":e})
a.to_csv("subit2.csv",index=False)


 
