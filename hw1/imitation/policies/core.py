
import abc
import numpy as np

DEFAULT_LEARNING_RATE = 0.0005

class Backbone( abc.ABC ) :

    def __init__( self, backboneConfig, **kwargs ) :
        super( Backbone, self ).__init__()

        self._config = backboneConfig


    @abc.abstractmethod
    def save( self, filepath, **kwargs ) :
        pass


    @abc.abstractmethod
    def load( self, filepath, **kwargs ) :
        pass


    @property
    def config( self ) :
        return self._config


class Model( object ) :

    def __init__( self, backbone, **kwargs ) :
        super( Model, self ).__init__()

        self._backbone  = backbone
        self._lr        = kwargs['lr'] if 'lr' in kwargs else DEFAULT_LEARNING_RATE
        self._extension = None
        self._logpath   = None


    @abc.abstractmethod
    def predict( self, observation ) :
        pass


    @abc.abstractmethod
    def train( self, observations, actions ) :
        pass


    def setLogPath( self, logpath ) :
        self._logpath = logpath


    def save( self, filepath ) :
        assert self._extension != None, 'ERROR> should define save-file extension'
        self._backbone.save( filepath + self._extension )


    def load( self, filepath ) :
        assert self._extension != None, 'ERROR> should define load-file extension'
        self._backbone.load( filepath + self._extension )

    @abc.abstractmethod
    def logs( self ) :
        pass