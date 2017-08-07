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
from keras.callbacks import LambdaCallback, EarlyStopping,ModelCheckpoint

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
hidden_units = 256
num_energies = 5

loss_history = []
val_loss_history = []

save_loss_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: loss_history.append(logs['loss']))
save_valloss_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: val_loss_history.append(logs['val_loss']))
monitor_loss = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
filepath = "thirdmodel.{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only= True)

model = Sequential()

model.add(Bidirectional(GRU(hidden_units, return_sequences=True), input_shape=(n_chunks,chunk_size)))
model.add(Dropout(0.2))
model.add(Bidirectional(GRU(hidden_units, return_sequences=True, kernel_regularizer=regularizers.l2(0.0001))))
model.add(Dropout(0.1))
model.add(Bidirectional(GRU(hidden_units, return_sequences=True, kernel_regularizer=regularizers.l2(0.0001))))
model.add(Dropout(0.1))
model.add(TimeDistributed(Dense(5, activation='linear')))
adam = optimizers.Adam(lr=0.001, beta_1= 0.9, beta_2 = 0.999, epsilon=1e-08,decay=0.0)
model.compile(loss='mse', optimizer=adam)
print(model.summary())

model.fit(np.asarray(train_input), np.asarray(train_output), 
	validation_data=(np.asarray(valid_input), 
		np.asarray(valid_output)), epochs=2000, batch_size=batch_size, callbacks=[save_loss_callback, 
	save_valloss_callback, monitor_loss, checkpoint])

print ("VALIDATION SET ")

prediction = model.predict(np.asarray(valid_input), batch_size=batch_size)
prediction = np.transpose(prediction,[1,0,2])
prediction = np.reshape(prediction, [-1, num_energies])

valid_output = np.transpose(valid_output,[1,0,2])
valid_output = np.reshape(valid_output, [-1, num_energies])

r2 =  r2_score(valid_output, prediction, multioutput='raw_values')
print('R2 SCORE:', (r2))

print ("TEST SET ")
prediction = model.predict(np.asarray(test_input), batch_size=batch_size)
prediction = np.transpose(prediction,[1,0,2])
prediction = np.reshape(prediction, [-1, num_energies])

test_output = np.transpose(test_output,[1,0,2])
test_output = np.reshape(test_output, [-1, num_energies])

# check for correlation

h_pred = prediction[:,0]
e_pred = prediction[:,1]
hx_pred = prediction[:,2]
s_pred = prediction[:,3]
c_pred = prediction[:,4]

C_sum = np.sum(np.absolute(np.round(c_pred,4)))
HX_sum = np.sum(np.absolute(np.round(hx_pred,4)))
S_sum = np.sum(np.absolute(np.round(s_pred,4)))

print (C_sum + HX_sum)
print (S_sum)

# print test results

print('MSE error:', mean_squared_error(test_output, prediction, multioutput='raw_values'))
r2 =  r2_score(test_output, prediction, multioutput='raw_values')
print('R2 SCORE:', (r2))

stop = timeit.default_timer()

print ('It took', stop-start, 'secs')