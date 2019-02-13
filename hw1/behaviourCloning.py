
import os
import gym
import pickle
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import utils
utils.loadDynamicDeps()

class ExperienceDataIterator( object ) :

    def __init__( self, pickleFile ) :
        super( ExperienceDataIterator, self ).__init__()

        # load experience stored in pickle file
        with open( pickleFile, 'rb' ) as f :
            _expertData = pickle.loads( f.read() )
            self.m_obsBuffer = _expertData['observations']
            self.m_actBuffer = _expertData['actions']

            _errorMsg = 'Observations and Actions buffers should be same size'
            assert len( self.m_obsBuffer ) == len( self.m_actBuffer ), _errorMsg

            self.m_size = len( self.m_obsBuffer )

        # some book keeping variables
        self.m_currentIndx = 0

    def __iter__( self ) :
        self.m_currentIndx = 0
        return self

    def __next__( self ) :
        _observation = self.m_obsBuffer[ self.m_currentIndx ]
        _action = self.m_actBuffer[ self.m_currentIndx ]

        self.m_currentIndx = ( self.m_currentIndx + 1 ) % self.m_size

        return ( _observation, _action )

    def __len__( self ) :
        return self.m_size

## _dataPicklePath = 'expert_data/Hopper-v2.pkl'
## _dataIterator = ExperienceDataIterator( _dataPicklePath )

## print( 'num (obs,act) of experience data: ', len( _dataIterator ) )

## for _ in range( 10 ) :
##     obs, act = next( _dataIterator )
##     
##     print( 'obs: ', obs )
##     print( 'act: ', act )



class PolicyModel( object ) :

    def __init__( self, obsDimension, actDimension ) :
        super( PolicyModel, self ).__init__()

        self.m_kerasModel = keras.Sequential()
        self.m_kerasModel.add( layers.Dense( 32, activation = tf.nn.relu, input_shape = [ obsDimension ] ) )
        self.m_kerasModel.add( layers.Dense( 32, activation = tf.nn.relu ) )
        self.m_kerasModel.add( layers.Dense( actDimension ) )

        self.m_kerasModel.compile( loss = 'mse',
                                   optimizer = keras.optimizers.RMSprop( 0.001 ),
                                   metrics = ['mse'] )

    def act( self, obs ) :
        ## print( 'observation: ', obs )
        ## print( 'shape: ', obs.shape )
        return self.m_kerasModel.predict( obs )

    def train( self, experienceBatch ) :
        pass


with tf.Session() as sess :

    _env = gym.make( 'Walker2d-v2' )
    _policy = PolicyModel( _env.observation_space.shape[0],
                           _env.action_space.shape[0] )

    _observation = _env.reset()
    _num_steps = 0
    _max_steps = _env.spec.timestep_limit
    while True :

        _action = _policy.act( np.array( _observation ).reshape( 1, 17 ) )
        _observation, _reward, _done, _ = _env.step( _action )
        _env.render()

        _num_steps += 1

        if ( _num_steps > _max_steps ) :
            break