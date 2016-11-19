from keras.models import Sequential
# TimeDistributedDense layer is deprecated, prefer using the TimeDistributed wrapper
# from keras.layers.core import TimeDistributedMerge, TimeDistributedDense, Dense, Dropout, Activation
from keras.layers.core import TimeDistributedDense, Dense, Dropout, Activation
from keras.layers.core import Lambda
from keras import backend as K
# from keras.layers.core import TimeDistributed   #comment since import error.
# I am using TimeDistributedMerge to merge outputs from lstm over time by returning sequences, 
# since TimeDistributedMerge is deprecated, so I changed my implementation to use 
# timedistributed wrapper then it gives error.
################################################################################
# I have experienced the latest version of Keras in github, and noticed that 
# there is a big difference from previous versions.
# TimeDistributed has now been separated as a wrapper.
# TimeDistributedDense can be replaced by TimeDistributed(Dense())
# how about the TimeDistributedMerge?
# should we write an Lambda layer by ourselves?
##################
# the Merge layer merges several tensors into one, whereas TimeDistributedMerge just collapsed 
# one axis of a single tensor.

# keras.engine.topology.Merge(layers=None, mode='sum', concat_axis=-1, dot_axes=-1, output_shape=None, output_mask=None, node_indices=None, tensor_indices=None, name=None)
from nyse import *
from nn import *
# from keras.optimizers import SGD
from keras.optimizers import Adagrad
# import theano
# theano.compile.mode.Mode(linker='py', optimizer='fast_compile')


class MLP:
    def __init__(self, input_length, hidden_cnt, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_length = input_length
        self.hidden_cnt = hidden_cnt
        self.model = self.__prepare_model()

    def __prepare_model(self):
        print('Build model...')
        model = Sequential()
        model.add(TimeDistributedDense(output_dim=self.hidden_cnt,
                                       input_dim=self.input_dim,
                                       input_length=self.input_length,
                                       activation='sigmoid'))
#         model.add(TimeDistributed(Dense(output_dim=self.hidden_cnt,
#                                         input_dim=self.input_dim,
#                                         input_length=self.input_length,
#                                         activation='sigmoid')))
# my modification since import error from keras.layers.core import TimeDistributedMerge
#         model.add(TimeDistributedMerge(mode='ave'))   #comment by me

##################### my ref #########################################################
# # add a layer that returns the concatenation
# # of the positive part of the input and
# # the opposite of the negative part
# 
# def antirectifier(x):
#     x -= K.mean(x, axis=1, keepdims=True)
#     x = K.l2_normalize(x, axis=1)
#     pos = K.relu(x)
#     neg = K.relu(-x)
#     return K.concatenate([pos, neg], axis=1)
# 
# def antirectifier_output_shape(input_shape):
#     shape = list(input_shape)
#     assert len(shape) == 2  # only valid for 2D tensors
#     shape[-1] *= 2
#     return tuple(shape)
# 
# model.add(Lambda(antirectifier, output_shape=antirectifier_output_shape))
#############################################################################

        model.add(Lambda(function=lambda x: K.mean(x, axis=1), 
                   output_shape=lambda shape: (shape[0],) + shape[2:]))
#         model.add(Dropout(0.5))
        model.add(Dropout(0.93755))
        model.add(Dense(self.hidden_cnt, activation='tanh'))
        model.add(Dense(self.output_dim, activation='softmax'))

        # try using different optimizers and different optimizer configs
        print('Compile model...')
#         sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#         model.compile(loss='categorical_crossentropy', optimizer=sgd)
#         return model
##my add
        adagrad = keras.optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
        model.compile(loss='categorical_crossentropy', optimizer=adagrad)
        return model

    def change_input_dim(self, input_dim):
        self.input_dim = input_dim
        self.model = self.__prepare_model()

    def get_model(self):
        return self.model


def main():
    input_length = 100
    hidden_cnt = 50
    
    nn = NeuralNetwork(MLP(input_length, hidden_cnt))
    data = get_test_data(input_length)
    print("TRAIN")
    nn.train(data)
    print("TEST")
    nn.test(data)
    print("TRAIN WITH CROSS-VALIDATION")
    nn.run_with_cross_validation(data, 2)
    print("FEATURE SELECTION")
    features = nn.feature_selection(data)
    print("Selected features: {0}".format(features))

if __name__ == '__main__':
    main()