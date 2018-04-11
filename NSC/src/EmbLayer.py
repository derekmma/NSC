#-*- coding: UTF-8 -*-  
import theano
import theano.tensor as T
import numpy
import pickle
# import cPickle

class EmbLayer(object):
    def __init__(self, rng, inp, n_voc, dim, name, dataname,prefix=None):
        self.input = inp
        self.name = name

        if prefix == None:
            # f = file('../../../data/'+dataname+'/embinit.save', 'rb')
            f = open('../../../data/'+dataname+'/embinit.save', 'rb')
            W = pickle.load(f, encoding='latin1')
            f.close()
            # W = cPickle.load(f)
            # f.close()
            W = theano.shared(value=W, name='E', borrow=True)    
        else:
            # f = file(prefix + name + '.save', 'rb')
            f = open(prefix + name + '.save', 'rb')
            # W = cPickle.load(f)
            W = pickle.load(f, encoding='latin1')
            f.close()
        self.W = W

        self.output = self.W[inp.flatten()].reshape((inp.shape[0], inp.shape[1], dim))
        self.params = [self.W]

    def save(self, prefix):
        # f = file(prefix + self.name + '.save', 'wb')
        f = open(prefix + self.name + '.save', 'wb')
        for obj in self.params:
            pickle.dump(obj, f)
            # cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
