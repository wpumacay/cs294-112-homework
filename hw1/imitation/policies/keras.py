
import abc
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import callbacks

from tensorboardX import SummaryWriter
from imitation.policies import core

from IPython.core.debugger import set_trace

class BackboneKeras( core.Backbone ) :

    def __init__( self, backboneConfig, **kwargs ) :
        super( BackboneKeras, self ).__init__( backboneConfig, **kwargs )

        # keras.Sequential model
        self._kerasModel = None


    def save( self , filepath, **kwargs ) :
        self._kerasModel.save( filepath )


    def load( self, filepath, **kwargs ) :
        self._kerasModel = models.load_model( filepath )


    def forward( self, x ) :
        return self._kerasModel.predict( x )


    def train( self, X, Y ) :
        self._kerasModel.fit( X, Y )


class BackboneKerasTest( BackboneKeras ) :

    def __init__( self, backboneConfig, **kwargs ) :
        super( BackboneKerasTest, self ).__init__( backboneConfig, **kwargs )

        self._kerasModel = keras.Sequential()
        self._kerasModel.add( layers.Dense( 32, activation = tf.nn.relu, input_shape = self._config.observationsShape ) )
        self._kerasModel.add( layers.Dense( 32, activation = tf.nn.relu ) )
        self._kerasModel.add( layers.Dense( self._config.actionsShape[0] ) )


    @property
    def kerasModel( self ) :
        return self._kerasModel


class ModelKeras( core.Model ) :

    def __init__( self, backbone, **kwargs ) :
        super( ModelKeras, self ).__init__( backbone, **kwargs )

        # define the extension for checkpoint files
        self._extension = '.h5'

        # compile keras model
        self._backbone.kerasModel.compile( loss = 'mse',
                                           optimizer = keras.optimizers.Adam( self._lr ) )

        # some variables for book-keeping
        self._currentTrainLoss = 0.
        self._istep = 0

        # tensorboardX summary object
        self._logger = None


    def predict( self, observation ) :
        return self._backbone.forward( observation )


    def train( self, observations, actions ) :
        # take SGD on just one batch of data
        self._currentTrainLoss = self._backbone.kerasModel.train_on_batch( observations, 
                                                                           actions, 
                                                                           reset_metrics = False )

        # log training information ---------------------------------------------
        if not self._logger :
            # sanity-check
            assert self._logpath != None, 'ERROR> logpath should be defined'
            # create logger once
            self._logger = SummaryWriter( self._logpath )

        self._logger.add_scalar( 'log_1_loss', self._currentTrainLoss, self._istep )
        # ----------------------------------------------------------------------

        # book keeping
        self._istep += 1


    def logs( self ) :
        return 'train-loss = %.5f' % ( self._currentTrainLoss )