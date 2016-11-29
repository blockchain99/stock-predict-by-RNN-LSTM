# stock-predict-by-RNN-LSTM
StockPredictionRNN

High Frequency Trading Price Prediction using LSTM Recursive Neural Networks

In this project we try to use recurrent neural network with long short term memory to predict prices in high frequency stock exchange. This program implements such a solution on data from NYSE OpenBook history which allows to recreate the limit order book for any given time. Everything is described in our paper: project.pdf

Project done for course of Computational Intelligence in Business Applications at Warsaw University of Technology - Department of Mathematics and Computer Science
Karol Dzitkowski 's result is  as follow.(High Frequency Trading Price Prediction 
using LSTM Recursive Neural Networks, Karol Dzitkowski)
RNN avg err = 0 . 5 6 7 3 0 0 7 2 4 6 3 8
epsilon = 0.5, SGD optimizer

#### This part was modified by Yoonsu Park 
#### After I changed epsilon = 0.9375, optimizer to Adagrad optimizer, RNN ave err is as follow. 
2nd RNN AVERAGE ERROR = 0.56443236715
RNN TEST ERRORS = [ 0.57669082  0.56219807  0.56944444  0.55193237  0.57789855  0.57246377
  0.57125604  0.57729469  0.57246377  0.5839372   0.55012077  0.57125604
  0.54770531  0.56038647  0.5531401   0.55736715  0.5736715   0.54770531
  0.55495169  0.55676329]
Now, I need to incorporate Seq2Seq and Recurrent Shop in  this program to tune up the performance at next step.
#### The end of modified pary by Yoonsu Park

### Data

To use this program one has to acquire data first. We need file openbookultraAA_N20130403_1_of_1 from NYSE. It can be downloaded from ftp://ftp.nyxdata.com/Historical%20Data%20Samples/TAQ%20NYSE%20OpenBook/ using FTP. Unzip it and copy to folder src/nyse-rnn.

### Installation and usage

Program is written in Python 2.7 with usage of library Keras - installation instruction To install it one may need Theano installed as well as numpy, scipy, pyyaml, HDF5, h5py, cuDNN (not all are actually needed). -- my version : I installed HDF5 by "sudo apt-get install libhdf5-serial-dev" It is useful to install also OpenBlas.

sudo pip install git+git://github.com/Theano/Theano.git
sudo pip install keras
We use numpy, scipy, matplotlib and pymongo in this project so it will be useful to have them installed.

sudo pip install numpy scipy matplotlib pymongo
To run the program first run nyse.py to create symbols and then main.py (creating folder symbols is necessary):

cd StockPredictionRNN
cd src/nyse-rnn
mkdir symbols
python nyse.py
python main.py
To save data to mongodb one has to install it first mongo install

Look into the code, it may be necessary to uncomment some lines to enable different features.

### Performance

To use CUDA and OpenBlas create file ~/.theanorc and fill it with this content:

[global]
floatX = float32 
device = gpu1

[blas]
ldflags = −L/usr/local/lib −lopenblas

[nvcc]
fastmath = True

### Meta 
Yoonsu Park - http://www.patternics.com
Distributed under the MIT license. See LICENSE for more information( https://en.wikipedia.org/wiki/MIT_License ).
http://www.patternics.com/stock-predict-by-RNN-LSTM/

