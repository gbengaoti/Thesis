import timeit
start = timeit.default_timer()
import edward as ed
from edward.models import Normal, Bernoulli
import timeit
start = timeit.default_timer()
import numpy as np
import theano.tensor as T
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn import MultiRNNCell

# train - valid - test split 60-20-20
data = np.load('energy_seq.npz')
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


chunk_size = 20
n_chunks = 200
batch_size = 100
rnn_size = 128

#data are [Batch Size, Sequence Length, Input Dimension].
data = tf.placeholder('float', [None, n_chunks, chunk_size])
target = tf.placeholder('float', [None,n_chunks, chunk_size])

# Defining the model
n_classes = 5

#inference model

def grucell():
    return rnn.GRUCell(rnn_size)

weights = Normal(loc=tf.zeros([rnn_size,n_classes]), scale=tf.ones([rnn_size,n_classes]))
biases =  Normal(loc=tf.zeros(n_classes), scale=tf.ones(n_classes))

data_t1 = tf.transpose(data,[1,0,2])
data_t2 = tf.reshape(data_t1, [-1, chunk_size])
data_t3 = tf.split(data_t2, n_chunks, 0)

stacked_Gru = MultiRNNCell([grucell() for _ in range(3)]) 
outputs, states = rnn.static_rnn(stacked_Gru, data_t3, dtype=tf.float32)

outputs = tf.transpose(outputs, [1, 0, 2])
outputs = tf.reshape(outputs, [-1, rnn_size])

y = tf.add(tf.matmul(outputs, weights), biases)
y = tf.reshape(y, [-1, n_chunks, n_classes])
prediction  = Normal(y, scale=tf.ones(n_classes))

#variational model
qWeights = Normal(loc=tf.Variable(tf.zeros([rnn_size,n_classes])),
              scale=tf.nn.softplus(tf.Variable(tf.zeros([rnn_size,n_classes]))))

qbaises = Normal(loc=tf.Variable(tf.zeros((n_classes))),
              scale=tf.nn.softplus(tf.Variable(tf.zeros((n_classes)))))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(prediction.get_shape().as_list())
    train_output = np.reshape(train_output, [-1, n_chunks, n_classes])
    inference = ed.KLqp({weights: qWeights, biases: qbaises}, data={data:train_input, prediction:train_output})
    
    inference.run(n_samples=5, n_iter=100)

    out_post = ed.copy(prediction, {weights: qWeights, biases: qbaises})

    print("Accuracy on test data:")
    test_output = np.reshape(test_output, [-1, n_chunks, n_classes])
    print(ed.evaluate('mean_squared_error', data={data: test_input, out_post: test_output}))


stop = timeit.default_timer()

print ('It took', stop-start, 'secs')


