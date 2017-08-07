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
from keras.callbacks import LambdaCallback
from scipy.stats import pearsonr

# train - valid - test split 60-20-20
data = np.load('energy_seq.npz')
y_data = (data['y']) #all_energies 

X_data = (data['x']) #all_sequences 

hydrophobic = X_data[:,:,0]
electrostatic = X_data[:,:,1]
helix = X_data[:,:,2]
sheet = X_data[:,:,3]
coil = X_data[:,:,4]

hydrophobic = np.reshape(hydrophobic, [-1, 200, 1])
print (hydrophobic.shape)

NUM_EXAMPLES = 2000
test_input = X_data[NUM_EXAMPLES:NUM_EXAMPLES+1000]
test_output = hydrophobic[NUM_EXAMPLES:NUM_EXAMPLES+1000] 

valid_input = X_data[NUM_EXAMPLES+1000:4000]
valid_output = hydrophobic[NUM_EXAMPLES+1000:4000]

train_input = X_data[:NUM_EXAMPLES]
train_output = hydrophobic[:NUM_EXAMPLES]
print ("Data done generating ....")

chunk_size = 20
n_chunks = 200
batch_size = 100
hidden_units = 4

loss_history = []
val_loss_history = []

save_loss_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: loss_history.append(logs['loss']))
save_valloss_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: val_loss_history.append(logs['val_loss']))

model = Sequential()
model.add(Bidirectional(GRU(hidden_units, return_sequences=True), input_shape=(n_chunks,chunk_size)))
#model.add(Dropout(0.2))
model.add(Bidirectional(GRU(hidden_units, return_sequences=True)))
#model.add(Dropout(0.5))
model.add(Bidirectional(GRU(hidden_units, return_sequences=True)))

model.add(Bidirectional(GRU(hidden_units, return_sequences=True)))
model.add(Bidirectional(GRU(hidden_units, return_sequences=True)))
model.add(Bidirectional(GRU(hidden_units, return_sequences=True)))
#model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(1, activation='linear')))
adam = optimizers.Adam(lr=0.001, beta_1= 0.9, beta_2 = 0.999, epsilon=1e-08,decay=0.0)
model.compile(loss='mse', optimizer=adam)
print(model.summary())

model.fit(np.asarray(train_input), np.asarray(train_output), 
	validation_data=(np.asarray(valid_input), 
		np.asarray(valid_output)), epochs=500, batch_size=batch_size, callbacks=[save_loss_callback, 
	save_valloss_callback])

# Final evaluation of the model
#hydrophobic_pred = np.array(prediction)
#np.savetxt("hydrophobic_pred.txt", hydrophobic_pred, delimiter=",")

numpy_loss_history = np.array(loss_history)
np.savetxt("convergence_second_model8units_train.txt", numpy_loss_history, delimiter=",")

numpy_vloss_history = np.array(val_loss_history)
np.savetxt("convergence_second_model8units_valid.txt", numpy_vloss_history, delimiter=",")

print ("EVALUATION OF THE VALIDATION SET")

prediction = model.predict(np.asarray(valid_input), batch_size=batch_size)
prediction = np.reshape(prediction, [-1, 200])
valid_output = np.reshape(valid_output, [-1, 200])

acc =  r2_score(valid_output, prediction)
print('Accuracy:', (acc))
prediction = np.reshape(prediction, [-1, 1])
valid_output = np.reshape(valid_output, [-1, 1])
n = len(prediction)
rmse = np.linalg.norm(prediction - valid_output)/np.sqrt(n)
print ('RMSE:',rmse )
print (pearsonr(valid_output, prediction))

print ("EVALUATION OF THE TEST SET")
# prediction = model.predict(np.asarray(test_input), batch_size=batch_size)
# prediction = np.reshape(prediction, [-1, 200])
# test_output = np.reshape(test_output, [-1, 200])
# acc =  r2_score(test_output, prediction)
# print('Accuracy:', (acc))
# prediction = np.reshape(prediction, [-1, 1])
# test_output = np.reshape(test_output, [-1, 1])
# n = len(prediction)
# rmse = np.linalg.norm(prediction - test_output)/np.sqrt(n)
# print ('RMSE:',rmse )
# print (pearsonr(test_output, prediction))

stop = timeit.default_timer()

print ('It took', stop-start, 'secs')