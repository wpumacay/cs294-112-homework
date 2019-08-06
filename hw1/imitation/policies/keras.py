
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import callbacks

class MlpPolicy( object ) :

    def __init__( self, obsDimension, actDimension ) :
        super( PolicyModel, self ).__init__()

        self._kerasModel = keras.Sequential()
        self._kerasModel.add( layers.Dense( 32, activation = tf.nn.relu, input_shape = [ obsDimension ] ) )
        self._kerasModel.add( layers.Dense( 32, activation = tf.nn.relu ) )
        self._kerasModel.add( layers.Dense( actDimension ) )

        self._kerasModel.compile( loss = 'mse',
                                   optimizer = keras.optimizers.RMSprop( 0.001 ),
                                   metrics = ['mse'] )

    def save( self, filepath ) :
        self._kerasModel.save( filepath )

    def load( self, filepath ) :
        self._kerasModel = models.load_model( filepath )

    def act( self, obs ) :
        return self._kerasModel.predict( obs )

    def train( self, observations, actions ) :
        _actShape = experienceBatch['act'].shape
        self._kerasModel.fit( experienceBatch['obs'],
                              experienceBatch['act'].reshape( _actShape[0], 
                                                              _actShape[1] * _actShape[2] ),
                              verbose = 0 )