
import pickle
import numpy as np

class ExperienceDataIterator( object ) :

    def __init__( self, pickleFile, batchSize, shuffle = False ) :
        super( ExperienceDataIterator, self ).__init__()

        self._batchSize = batchSize

        # load experience stored in pickle file
        with open( pickleFile, 'rb' ) as f :
            _expertData = pickle.loads( f.read() )
            self._obsBuffer = _expertData['observations']
            self._actBuffer = _expertData['actions']

            _errorMsg = 'Observations and Actions buffers should be same size'
            assert len( self._obsBuffer ) == len( self._actBuffer ), _errorMsg

            self._size = len( self._obsBuffer )

        # some book keeping variables
        self._currentIndx = 0
        # indices to get the points from
        self._indices = np.arange( self._size )

        if shuffle :
            np.random.shuffle( self._indices )


    def __iter__( self ) :
        self._currentIndx = 0
        return self

    def __next__( self ) :
        _observations = []
        _actions = []

        for _ in range( self._batchSize ) :
            _observations.append( self._obsBuffer[ self._currentIndx ] )
            _actions.append( self._actBuffer[ self._currentIndx ] )

            self._currentIndx = ( self._currentIndx + 1 ) % self._size

        _batch = { 'obs' : np.array( _observations ),
                   'act' : np.array( _actions ) }

        return _batch

    def __len__( self ) :
        return self._size


## _dataPicklePath = '../expert_data_bp/Walker2d-v2.pkl'
## _dataIterator = ExperienceDataIterator( _dataPicklePath, 10 )
## 
## print( 'num (obs,act) of experience data: ', len( _dataIterator ) )
## print( 'one batch(10):' )
## print( next( _dataIterator ) )