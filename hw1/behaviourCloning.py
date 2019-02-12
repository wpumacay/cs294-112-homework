
import os
import numpy as np
import pickle
import tensorflow as tf


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

_dataPicklePath = 'expert_data/Hopper-v2.pkl'
_dataIterator = ExperienceDataIterator( _dataPicklePath )

print( 'num (obs,act) of experience data: ', len( _dataIterator ) )

for _ in range( 10 ) :
    obs, act = next( _dataIterator )
    
    print( 'obs: ', obs )
    print( 'act: ', act )