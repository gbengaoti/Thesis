# TO DO
# CREATE KERAS IMPLEMENTATION OF A GRU RECURRENT NETWORK
import timeit
start = timeit.default_timer()
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,GRU
from keras import regularizers
from keras.layers import Embedding, Dropout
from keras.layers.wrappers import Bidirectional
from keras.layers.wrappers import TimeDistributed
from keras import optimizers
from sklearn.metrics import mean_squared_error, r2_score
from keras.callbacks import LambdaCallback, EarlyStopping

# train - valid - test split 60-20-20
data = np.load('energy_seq_pssm.npz')
y_data = (data['y']) #all_energies 

X_data = (data['x']) #all_sequences 

NUM_EXAMPLES = 2000
test_input = X_data[NUM_EXAMPLES:NUM_EXAMPLES+1000]
test_output = y_data[NUM_EXAMPLES:NUM_EXAMPLES+1000] 

valid_input = X_data[NUM_EXAMPLES+1000:4000]
valid_output = y_data[NUM_EXAMPLES+1000:4000]

train_input = X_data[:NUM_EXAMPLES]
train_output = y_data[:NUM_EXAMPLES]
print ("Data done generating ....")

chunk_size = 40
n_chunks = 200
batch_size = 100
hidden_units = 64

loss_history = []
val_loss_history = []

save_loss_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: loss_history.append(logs['loss']))
save_valloss_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: val_loss_history.append(logs['val_loss']))
monitor_loss = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')


model = Sequential()

model.add(Bidirectional(GRU(hidden_units, return_sequences=True), input_shape=(n_chunks,chunk_size)))
model.add(Bidirectional(GRU(hidden_units, return_sequences=True)))
model.add(Bidirectional(GRU(hidden_units, return_sequences=True)))
model.add(TimeDistributed(Dense(5, activation='linear')))
#rmsprop = optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)
adam = optimizers.Adam(lr=0.001, beta_1= 0.9, beta_2 = 0.999, epsilon=1e-08,decay=0.0)
model.compile(loss='mse', optimizer=adam)
print(model.summary())

model.fit(np.asarray(train_input), np.asarray(train_output), 
	validation_data=(np.asarray(valid_input), 
		np.asarray(valid_output)), epochs=2000, batch_size=batch_size, callbacks=[save_loss_callback, 
	save_valloss_callback, monitor_loss])

# Final evaluation of the model
print ("VALIDATION SET ")

prediction = model.predict(np.asarray(valid_input), batch_size=batch_size)
prediction = np.transpose(prediction,[1,0,2])
prediction = np.reshape(prediction, [-1, 5])

valid_output = np.transpose(valid_output,[1,0,2])
valid_output = np.reshape(valid_output, [-1, 5])

acc =  r2_score(valid_output, prediction, multioutput='raw_values')
print('Accuracy:', (acc))

stop = timeit.default_timer()

print ('It took', stop-start, 'secs')