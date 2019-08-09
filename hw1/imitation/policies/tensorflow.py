
import abc
import numpy as np
import tensorflow as tf

from imitation.policies import core
from IPython.core.debugger import set_trace

def GetDefaultSession() :
    sess = tf.get_default_session()
    sess = tf.InteractiveSession() if not sess else sess

    return sess

def CloseDefaultSession() :
    sess = tf.get_default_session()
    if sess :
        sess.close()

DEFAULT_SEED = 0
DEFAULT_SESSION = GetDefaultSession()

class BackboneTensorflow( core.Backbone ) :


    def __init__( self, backboneConfig, **kwargs ) :
        super( BackboneTensorflow, self ).__init__( backboneConfig, **kwargs )

        # grab the random seed used for the initialization
        self._seed = kwargs['seed'] if 'seed' in kwargs else DEFAULT_SEED


    def save( self, filepath, **kwargs ) :
        session = kwargs['session']
        saver = kwargs['saver']

        saver.save( session, filepath )


    def load( self, filepath, **kwargs ) :
        session = kwargs['session']
        saver = kwargs['saver']

        saver.restore( session, filepath )


class BackboneTensorflowTest( BackboneTensorflow ) :


    def __init__( self, backboneConfig, **kwargs ) :
        super( BackboneTensorflowTest, self ).__init__( backboneConfig, **kwargs )

        # placeholder for the observations
        self._phObservations = tf.placeholder( tf.float32, shape = (None,) + self._config.observationsShape )

        # define the initializers of the backbone
        self._fc1Init = tf.initializers.glorot_uniform( seed = self._seed, dtype = tf.float32 )
        self._fc2Init = tf.initializers.glorot_uniform( seed = self._seed, dtype = tf.float32 )
        self._fc3Init = tf.initializers.glorot_uniform( seed = self._seed, dtype = tf.float32 )

        # layers used for our mlp
        with tf.name_scope( 'mlp_backbone' ) :
            self._h1 = tf.layers.dense( self._phObservations, 
                                        units = 32, 
                                        activation = tf.nn.relu, 
                                        kernel_initializer = self._fc1Init )

            self._h2 = tf.layers.dense( self._h1, 
                                        units = 32, 
                                        activation = tf.nn.relu,
                                        kernel_initializer = self._fc2Init )

            self._out = tf.layers.dense( self._h2, 
                                         units = self._config.actionsShape[0],
                                         activation = None,
                                         kernel_initializer = self._fc3Init )


    def forward( self, x, session ) :
        return session.run( self._out, feed_dict = { self._phObservations : x } )


    def inputs( self ) :
        return self._phObservations


    def outputs( self ) :
        return self._out


class ModelTensorflow( core.Model ) :

    def __init__( self, backbone, **kwargs ) :
        super( ModelTensorflow, self ).__init__( backbone, **kwargs )

        # grab the session (session must be kept alive for weights to be persist)
        self._session = kwargs['session'] if 'session' in kwargs else DEFAULT_SESSION

        # grab the configuration from the backbone
        _backboneConfig = backbone.config

        # define the extension for checkpoint files
        self._extension = '.ckpt'

        # create the remaining ops for BCloning --------------------------------

        # create target actions placeholder
        self._phActionsTargets = tf.placeholder( tf.float32, shape = (None,) + _backboneConfig.actionsShape )

        # create the loss op (mse-loss between predicted and target actions)
        self._opLoss = tf.losses.mean_squared_error( self._phActionsTargets, self._backbone.outputs() )

        # creata an optimizer and complete the computation graph
        self._optimizer = tf.train.AdamOptimizer( learning_rate = self._lr )
        self._opOptim = self._optimizer.minimize( self._opLoss )

        #-----------------------------------------------------------------------

        # initialize the backbone (@TODO: might want to initialize just this backbone)
        self._session.run( tf.global_variables_initializer() )

        # some variables for book-keeping
        self._currentTrainLoss = 0.
        self._istep = 0

        # summaries for tensorboard (need a file-writer and the required summaries)
        self._tfSummaryLoss = tf.summary.scalar( 'log_1_loss', self._opLoss )
        self._tfFileWriter = None # created on request (logpath given later)

        # saver to save/restore our model
        self._tfSaver = tf.train.Saver()


    def predict( self, observation ) :
        return self._backbone.forward( observation, self._session )


    def train( self, observations, actions ) :
        # define ops to run and data to be passed
        _trainOps = [ self._opLoss, self._opOptim, self._tfSummaryLoss ]
        _trainDict = { self._backbone.inputs() : observations, 
                       self._phActionsTargets : actions }

        # do the thing, run the training ops
        self._currentTrainLoss, _, _summary = self._session.run( _trainOps, _trainDict )

        # log training information ---------------------------------------------
        if not self._tfFileWriter :
            # sanity-check
            assert self._logpath != None, 'ERROR> logpath should be defined'
            # create file-writer
            self._tfFileWriter = tf.summary.FileWriter( self._logpath, self._session.graph )
            
        # append summary to the file-writer
        self._tfFileWriter.add_summary( _summary, self._istep )
        # ----------------------------------------------------------------------

        # book keeping
        self._istep += 1


    def logs( self ) :
        return 'train-loss = %.5f' % ( self._currentTrainLoss )


    # override the save-method, as backbone requires the session
    def save( self, filepath ) :
        assert self._extension != None, 'ERROR> should define save-file extension'
        self._backbone.save( filepath + self._extension, 
                             session = self._session,
                             saver = self._tfSaver )


    # override the load-method, as backbone requires the session
    def load( self, filepath ) :
        assert self._extension != None, 'ERROR> should define load-file extension'
        self._backbone.load( filepath + self._extension, 
                             session = self._session,
                             saver = self._tfSaver )
