
import os
import sys
import gym
import gin
import argparse
import numpy as np

from tqdm import tqdm
from IPython.core.debugger import set_trace

from imitation.utils import loadDynamicDeps
from imitation.utils.config import TrainerConfig
from imitation.utils.config import BackboneConfig
from imitation.utils.iterators import DAggerExperienceDataIterator

# functionality required to load the experts
import tf_util
import load_policy

TRAIN               = True                          # flag use to determine the training mode
EXP_MODELS_FOLDER   = 'experts/'                    # folder where the expert policies are located
EXP_DATA_FOLDER     = 'data/experts/'               # folder where to find the experts' rollouts data
RESULTS_FOLDER      = 'data/results/'               # fodler where the training results will be saved
SESSION_FOLDER      = 'data/results/session_ant/'   # folder related to this training session
NUM_EPOCHS          = 10                            # number of training epochs to run BCloning


def createModel( backend, backboneConfig, learningRate, seed ) :
    if backend == 'pytorch' :
        from imitation.policies.pytorch import BackbonePytorchTest as BackboneT
        from imitation.policies.pytorch import ModelPytorch as ModelT
    elif backend == 'tensorflow' :
        from imitation.policies.tensorflow import BackboneTensorflowTest as BackboneT
        from imitation.policies.tensorflow import ModelTensorflow as ModelT
    elif backend == 'keras' :
        from imitation.policies.keras import BackboneKerasTest as BackboneT
        from imitation.policies.keras import ModelKeras as ModelT

    # create the backbone for our model
    backbone = BackboneT( backboneConfig, lr = learningRate, seed = seed )
    # create the agent's model that uses this backbone
    model = ModelT( backbone )

    return model


def run( env, model, num_episodes ) :
    observations = []

    for  iepisode in tqdm( range( num_episodes ), desc = 'Episode>', leave = True ) :
        obs = env.reset()
        observations.append( obs )

        while True :
            # query the model for the action
            action = model.predict( obs[np.newaxis,...] ).squeeze( 0 )
            # take action in the environment
            obs, _, done, _ = env.step( action )
            # store observations for later usage
            observations.append( obs )

            if done :
                break

    return np.array( observations )


def train( env, model, dataIterator, numEpochs, sessionFolder ) :
    progressbarEpoch = tqdm( range( numEpochs ), desc = 'Epoch>', leave = False )
    savepath = os.path.join( sessionFolder, 'checkpoint' )

    logpath = os.path.join( sessionFolder, 'tensorboard/' )
    model.setLogPath( logpath )

    for iepoch in progressbarEpoch :
        dataIterator.shuffle()
        progressbarBatch = tqdm( dataIterator, desc = 'Training>', leave = False )

        for observations, actions in progressbarBatch :
            model.train( observations, actions )

        model.save( savepath )

        progressbarEpoch.set_description( 'Epoch> ' + model.logs() )
        progressbarEpoch.refresh()


def test( env, model, num_episodes, sessionFolder ) :
    progressbar = tqdm( range( num_episodes ), desc = 'Testing>' )
    savepath = os.path.join( sessionFolder, 'checkpoint' )

    model.load( savepath )

    for iepisode in progressbar :
        obs = env.reset()
        score = 0.

        while True :
            # query the model for the action
            action = model.predict( obs[np.newaxis,...] ).squeeze( 0 )
            # step into the environment
            obs, reward, done, _ = env.step( action )
            # render because ... why not? :D
            env.render()

            # and some book-keeping to check progress
            score += reward

            if done :
                break

        progressbar.set_description( 'Testing> score: %.3f' % score )
        progressbar.refresh()

if __name__ == '__main__':
    # fix for GLEW, in case mujoco does not load
    loadDynamicDeps()

    parser = argparse.ArgumentParser()
    parser.add_argument( 'mode', type = str, choices = ['train', 'test'], 
                         help = 'whether to run in train or test mode ' )
    parser.add_argument( '--config', type = str, default = './configs/bc_ant.gin',
                         help = 'gin-config file to use for configuration params' )
    args = parser.parse_args()

    # parse config file
    gin.parse_config_file( args.config )

    # create the configuration objects
    trainerConfig = TrainerConfig()
    backboneConfig = BackboneConfig()

    TRAIN           = ( args.mode == 'train' )
    SESSION_FOLDER  = os.path.join( RESULTS_FOLDER, trainerConfig.sessionID + '_' + trainerConfig.backend )
    NUM_EPOCHS      = trainerConfig.numEpochs

    # create the appropriate environment
    env = gym.make( trainerConfig.environmentName )

    # seed generators
    env.seed( trainerConfig.seed )
    np.random.seed( trainerConfig.seed )

    # grab environment information for backbone-config object
    backboneConfig.observationsShape = env.observation_space.shape
    backboneConfig.actionsShape = env.action_space.shape

    # create the model using the appropriate configuration
    model = createModel( trainerConfig.backend, backboneConfig, trainerConfig.learningRate, trainerConfig.seed )

    # create the appropriate session
    sess = tf_util.get_default_session()

    # create the expert policy and initialize its resources
    expertModel = load_policy.load_policy( os.path.join( EXP_MODELS_FOLDER, trainerConfig.expertPolicyFile ) )
    tf_util.initialize()

    if TRAIN :
        # create data iterator from wrapping expert data
        dataFile = os.path.join( EXP_DATA_FOLDER, trainerConfig.expertDataFile )
        dataIterator = iter( DAggerExperienceDataIterator( dataFile, trainerConfig.batchSize, percent = 0.4 ) )

        for iDAggerEpoch in tqdm( range( trainerConfig.daggerNumEpochs ), desc = 'DAgger-epochs>' ) :
            # train on dataset (pdata)
            train( env, model, dataIterator, NUM_EPOCHS, SESSION_FOLDER )
            # grab information by running agent (ptheta)
            observations = run( env, model, trainerConfig.daggerNumEpisodes )
            # label these observations with the expert
            actions = expertModel( observations )
            # aggregate these labels actions into dataset
            dataIterator.aggregate( observations, actions )

    else :
        test( env, model, 10, SESSION_FOLDER )

    if sess :
        sess.close()