#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import os
import gym
import pickle
import tf_util
import argparse
import load_policy
import numpy as np
from tqdm import tqdm
from imitation import utils
from IPython.core.debugger import set_trace

def run( env, policy, max_steps_per_episode, num_rollouts, results_filename, render ) :
    with tf_util.single_threaded_session() as session :
        tf_util.initialize()
    
        returns = []
        observations = []
        actions = []
    
        for iepisode in tqdm( range( num_rollouts ) ):
            obs = env.reset()
            totalr = 0.0
    
            for istep in range( max_steps_per_episode ):
                # get action from expert policy (input-shape: (batch,obs.shape))
                action = np.squeeze( policy( obs[None,:] ) )

                # store this (o,a) pair for later usage
                observations.append( obs )
                actions.append( action )

                # take a step in the environment
                obs, r, done, _ = env.step( action )
                totalr += r
    
                if render : env.render()
                if done : break
    
            returns.append( totalr )
    
        expert_data = { 'observations' : np.array( observations ),
                        'actions' : np.array( actions ),
                        'returns' : np.array( returns ) }
    
        with open( results_filename, 'wb') as fhandle :
            pickle.dump( expert_data, fhandle, pickle.HIGHEST_PROTOCOL )

if __name__ == '__main__':
    utils.loadDynamicDeps()

    parser = argparse.ArgumentParser()
    parser.add_argument( 'expert_policy_file', type=str )
    parser.add_argument( 'envname', type=str )
    parser.add_argument( '--render', action='store_true' )
    parser.add_argument( "--max_timesteps", type=int )
    parser.add_argument( '--num_rollouts', type=int, default=20,
                         help='Number of expert roll outs' )
    args = parser.parse_args()

    # create the appropriate environment
    env = gym.make( args.envname )
    env.seed( 0 )

    # create the policy
    policy = load_policy.load_policy( args.expert_policy_file )

    # grab some required information
    render                  = args.render or False
    max_steps_per_episode   = args.max_timesteps or env.spec.timestep_limit
    num_rollouts            = args.num_rollouts
    results_filename        = os.path.join( os.getcwd(), 
                                            'data/experts/' + 
                                            args.envname.split( '-' )[0].lower() + 
                                            '_' + str( num_rollouts ) + '_.pkl' )

    run( env, policy, max_steps_per_episode, num_rollouts, results_filename, render )
