
import gym
import numpy as np
from tqdm import tqdm
from imitation.Policy import PolicyModel

ENVIRONMENT = 'Walker2d-v2'

TRAIN = False

def train() :
    from imitation.DataIterator import ExperienceDataIterator

    _numEpochs = 10
    _batchSize = 64

    _dataPicklePath = 'expert_data_bp/' + ENVIRONMENT +'.pkl'
    _dataIterator = ExperienceDataIterator( _dataPicklePath, _batchSize )

    _numBatches = ( len( _dataIterator ) * _numEpochs ) // _batchSize

    _env = gym.make( ENVIRONMENT )

    print( 'observation_space: ', _env.observation_space.shape[0] )
    print( 'action_space: ', _env.action_space.shape[0] )

    _policy = PolicyModel( _env.observation_space.shape[0],
                           _env.action_space.shape[0] )

    for _ in tqdm( range( _numBatches ) ) :
        _policy.train( next( _dataIterator ) )

    _policy.saveModel( 'model_' + ENVIRONMENT + '.h5' )

def test() :
    import utils
    utils.loadDynamicDeps()
    
    _env = gym.make( 'Walker2d-v2' )
    _policy = PolicyModel( _env.observation_space.shape[0],
                               _env.action_space.shape[0] )
        
    _policy.loadModel( 'model_' + ENVIRONMENT + '.h5' )

    _observation = _env.reset()
    _num_steps = 0
    _max_steps = _env.spec.timestep_limit
    while True :
    
        _action = _policy.act( np.array( _observation ).reshape( 1, _env.observation_space.shape[0] ) )
        _observation, _reward, _done, _ = _env.step( _action )
        _env.render()
    
        _num_steps += 1
    
        if ( _num_steps > _max_steps ) :
            break

if __name__ == '__main__' :
    if TRAIN :
        train()
    else :
        test()