
import pickle
import numpy as np
from IPython.core.debugger import set_trace

class ExperienceDataIterator( object ) :

    def __init__( self, pickleFile, batchSize, percent = 1.0 ) :
        super( ExperienceDataIterator, self ).__init__()

        # storage for the observations taken in the expert's runs
        self._obsBuffer = None
        # storage for the actions taken in the expert's runs
        self._actBuffer = None
        # pointer used for the sampling process (start-position)
        self._currentIndx = 0
        # how many (o,a) pairs to take per sampling
        self._batchSize = batchSize
        # indices used for sampling (np.array of ints)
        self._indices = None
        # flag to check if already finished iteration
        self._finished = False
        # percent of the dataaset to grab
        self._percent = percent

        # grab the save (o,a) pairs from the expert's runs
        self._loadData( pickleFile )


    def _loadData( self, pickleFile ) :
        """Load experience stored in pickle file

        Args:
            pickleFile (str) : absolute path to the pickle file where the 
                               data is saved
        """
        with open( pickleFile, 'rb' ) as f :
            _expertData = pickle.loads( f.read() )
            self._obsBuffer = _expertData['observations']
            self._actBuffer = _expertData['actions']

            _errorMsg = 'ERROR> Observations and Actions buffers should \
                                have the same size'
            assert len( self._obsBuffer ) == len( self._actBuffer ), _errorMsg

            _nsamples = int( self._percent * len( self._obsBuffer ) )
            self._obsBuffer = self._obsBuffer[:_nsamples]
            self._actBuffer = self._actBuffer[:_nsamples]

            self._size = len( self._obsBuffer )
            self._indices = np.arange( self._size )


    def shuffle( self ) :
        """Shuffle indices used for grabbing samples from dataset"""
        np.random.shuffle( self._indices )
        self._finished = False
        self._currentIndx = 0


    def __iter__( self ) :
        """ Creates the appropriate iterator at the starting index"""
        self._currentIndx = 0
        return self


    def __next__( self ) :
        """Returns a batch of (o,a) pairs from the obs. and act. buffers

        Sampling is done using the sampling indices, and looping back 
        in case we ran out of samples to take the batch from.
        """
        if self._finished :
            raise StopIteration()
        else :
            _observations = []
            _actions = []
    
            for _ in range( self._batchSize ) :
                _sampleIndx = self._indices[self._currentIndx]
                _observations.append( self._obsBuffer[_sampleIndx] )
                _actions.append( self._actBuffer[_sampleIndx] )
    
                self._finished = ( self._currentIndx == ( self._size - 1 ) )
                self._currentIndx = ( self._currentIndx + 1 ) % self._size
    
            return np.array( _observations ), np.array( _actions )


    def __len__( self ) :
        return self._size


    @property
    def batchSize( self ) :
        return self._batchSize


if __name__ == '__main__' :
    _dataPicklePath = '../../data/experts/ant.pkl'
    _dataIterator = ExperienceDataIterator( _dataPicklePath, 10 )
    
    print( 'num (obs,act) of experience data: ', len( _dataIterator ) )
    print( 'one batch of size (%d):' % (_dataIterator.batchSize) )
    print( next( _dataIterator ) )
