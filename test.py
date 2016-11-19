# from keras.layers.core import TimeDistributedMerge, TimeDistributedDense, Dense, Dropout, Activation
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation
# from keras.optimizers import SGD
# 
# model = Sequential()
# # Dense(64) is a fully-connected layer with 64 hidden units.
# # in the first layer, you must specify the expected input data shape:
# # here, 20-dimensional vectors.
# model.add(Dense(64, input_dim=20, init='uniform'))
# model.add(Activation('tanh'))
# model.add(Dropout(0.5))
# model.add(Dense(64, init='uniform'))
# model.add(Activation('tanh'))
# model.add(Dropout(0.5))
# model.add(Dense(10, init='uniform'))
# model.add(Activation('softmax'))
# 
# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy',
#               optimizer=sgd,
#               metrics=['accuracy'])
# 
# model.fit(X_train, y_train,
#           nb_epoch=20,
#           batch_size=16)
# score = model.evaluate(X_test, y_test, batch_size=16)

# This is only to demonstrate how easy it is to write a RNNCell.
 # See recurrentshop/recurrentshop/cells.py for a better version of SimpleRNNCell with more options.
class SimpleRNNCell(RNNCell):

    def build(self, input_shape):
        input_dim = input_shape[-1]
        output_dim = self.output_dim
        h = (-1, output_dim)  # -1 = batch size
        W = (input_dim, output_dim)
        U = (output_dim, output_dim)
        b = (self.output_dim,)

    def step(x, states, weights):
        h_tm1 = states[0]
        W, U, b = weights
        h = K.dot(x, W) + K.dot(h_tm1, U) + b

        self.step = step
        self.weights = [W, U, b]
        self.states = [h]

        super(SimpleRNNCell, self).build(input_shape)

rc = RecurrentContainer()
rc.add(SimpleRNNCell(10, input_dim=20))
rc.add(Activation('tanh'))