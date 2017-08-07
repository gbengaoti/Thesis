# CREATE PYMC3+LASAGNE IMPLEMENTATION OF A GRU RECURRENT NETWORK
import timeit
start = timeit.default_timer()
import theano
import theano.tensor as T
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import lasagne
from sklearn.metrics import mean_squared_error,r2_score
from lasagne.layers import *

chunk_size = 40
n_chunks = 200
num_classes = 5


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

input_var = theano.shared(np.asarray(train_input).astype(np.float64))
target_var = theano.shared(np.asarray(train_output).astype(np.float64))

N_HIDDEN = 8
#GRAD_CLIP = 100


def build_rnn(init):
    print("Building network ...")

    # (batch size, SEQ_LENGTH, num_features)
    l_in = lasagne.layers.InputLayer(shape=(None,n_chunks, chunk_size),
        input_var= input_var)

    l_forward_1 = lasagne.layers.GRULayer(
            l_in, N_HIDDEN,
            only_return_final=False)

    l_shp = lasagne.layers.ReshapeLayer(l_forward_1, (-1, N_HIDDEN))

    l_dense = lasagne.layers.DenseLayer(l_shp, num_units=5, W=init,
        nonlinearity=lasagne.nonlinearities.linear)

    l_out = lasagne.layers.ReshapeLayer(l_dense,(-1,n_chunks,num_classes))

    prediction = lasagne.layers.get_output(l_out)

    print("Finished building layers...")

    eps = pm.Uniform('eps', lower=0, upper=1)


    # #target

    out = pm.Normal('out',mu=prediction, sd=eps, observed=target_var)
   
    return out


class GaussWeights(object):
    def __init__(self):
        self.count = 0
    def __call__(self, shape):
        self.count += 1
        print (shape)
        return pm.Normal('w%d' % self.count, mu=0, sd=1,
                         testval=np.random.normal(size=shape).astype(np.float64),
                         shape=shape)


with pm.Model() as rnn:
    out = build_rnn(GaussWeights())

from six.moves import zip

input_var.set_value(np.asarray(train_input).astype(np.float64))
target_var.set_value(np.asarray(train_output).astype(np.float64))

# Tensors and RV that will be using mini-batches
minibatch_tensors = [input_var, target_var]
minibatch_RVs = [out]

start = timeit.default_timer()
# Generator that returns mini-batches in each iteration
def create_minibatch(data):
    rng = np.random.RandomState(0)
    
    while True:
        # Return random data samples of set size 100 each iteration
        ixs = rng.randint(len(data), size=500)
        yield data[ixs]

minibatches = zip(
    create_minibatch(np.asarray(train_input)), 
    create_minibatch(np.asarray(train_output)),
)

total_size = len(train_input)

    
with rnn:
    print ("Optimization starts here")
    # Run ADVI which returns posterior means, standard deviations, and the evidence lower bound (ELBO)
    v_params = pm.variational.advi_minibatch(
        n=50000, minibatch_tensors=minibatch_tensors, 
        minibatch_RVs=minibatch_RVs, minibatches=minibatches, 
        total_size=total_size, learning_rate=0.001, epsilon=1.0)
    
    trace = pm.variational.sample_vp(v_params, draws=5000)
    

# should test on test set here
input_var.set_value(np.asarray(test_input).astype(np.float64))
target_var.set_value(np.asarray(test_output).astype(np.float64))

ppc = pm.sample_ppc(trace, model=rnn, samples=500)

print  (ppc['out'].shape)

pred = ppc['out'].mean(axis=0) 

print (pred.shape)

pred = np.transpose(pred,[1,0,2])
pred = np.reshape(pred, [-1, 5])

test_output = np.transpose(test_output,[1,0,2])
test_output = np.reshape(test_output, [-1, 5])


print('Accuracy = ',(r2_score(test_output, pred, multioutput='raw_values')))
print('MSE error:', mean_squared_error(test_output, pred, multioutput='raw_values'))
stop = timeit.default_timer()

print ('It took', stop-start, 'secs')

stop = timeit.default_timer()

print ('It took', stop-start, 'secs')