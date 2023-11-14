from configparser import ConfigParser
from argparse import ArgumentParser

import torch
import gymnasium as gym
import numpy as np
import os

from agents.ppo import PPO

from utils.utils import make_transition, Dict, RunningMeanStd

Tensor = torch.Tensor
device = None

def parse_args():
    """ Create argument parsers """
    parser = ArgumentParser('parameters')

    parser.add_argument("--env_name", type=str, default='HalfCheetah-v2', help="'Ant-v2','HalfCheetah-v2','Hopper-v2','Humanoid-v2','HumanoidStandup-v2',\
              'InvertedDoublePendulum-v2', 'InvertedPendulum-v2' (default : Hopper-v2)")
    parser.add_argument("--algo", type=str, default='ppo',
                        help='algorithm to adjust (default : ppo)')
    parser.add_argument('--train', type=bool, default=True, help="(default: True)")
    parser.add_argument('--render', type=bool, default=False, help="(default: False)")
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs, (default: 1000)')
    parser.add_argument('--tensorboard', type=bool, default=False,
                        help='use_tensorboard, (default: False)')
    parser.add_argument("--load", type=str, default='no',
                        help='load network name in ./model_weights')
    parser.add_argument("--save_interval", type=int, default=100,
                        help='save interval(default: 100)')
    parser.add_argument("--print_interval", type=int, default=1,
                        help='print interval(default : 20)')
    parser.add_argument("--use_cuda", type=bool, default=True, help='cuda usage(default : True)')
    parser.add_argument("--reward_scaling", type=float, default=0.1,
                        help='reward scaling(default : 0.1)')
    args = parser.parse_args()

    # Get configuration for experiment
    parser = ConfigParser()
    parser.read('config.ini')
    agent_args = Dict(parser, args.algo)

    return args, agent_args

def setup_experiment(args,
                     agent_args):
    """ Create agents and setup environment """
    os.makedirs('./model_weights', exist_ok=True)

    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.use_cuda == False:
        device = 'cpu'

    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter()
    else:
        writer = None

    """ Select environment """
    env = gym.make(args.env_name)
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    state_rms = RunningMeanStd(state_dim)

    """ Select RL Algorithm"""

    agent = PPO(writer, device, state_dim, action_dim, agent_args)

    attack_objective = PPO(writer, device, state_dim, action_dim, agent_args)
    #TODO: Make parameters random!

    if (torch.cuda.is_available()) and (args.use_cuda):
        agent = agent.cuda()
        attack_policy = attack_objective.cuda()

    if args.load != 'no':
        agent.load_state_dict(torch.load("./model_weights/" + args.load))

    return device, writer, env, state_rms, agent, attack_objective

def run_experiment(agent_args,
                   env,
                   state_rms,
                   agent,
                   attack_objective,
                   writer):
    """ Execute experiment, by training and storing rewards """
    score_lst = []
    budget_lst = []
    state_lst = []

    score = 0.0
    attack_budget = 0.0
    # Get initial state
    state_, _ = (env.reset())
    # input observation standardization
    state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
    for n_epi in range(args.epochs):
        for t in range(agent_args.traj_length):
            if args.render:
                env.render()

            state_lst.append(state_)
            # Obtain action by random sampling from generated distribution
            # Note that state input was standardized
            mu, sigma = agent.get_action(torch.from_numpy(np.array(state)).float().to(device))
            dist = torch.distributions.Normal(mu, sigma[0])  # Stochastic Actions!
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            # Apply action. Note that reward will be scaled (see argument parser)
            next_state_, reward, done, trunc, info = env.step(action.cpu().numpy())
            # Apply reward poisoning attack
            reward, attack_flag = poisoning_attack(attack_objective, action.cpu().numpy(), state, reward)
            # next state observation standardization
            next_state = np.clip((next_state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8),
                                 -5,
                                 5)
            # Creates a dictionary with state, action, reward, next_state, log_prob, and done
            transition = make_transition(state,
                                         action.cpu().numpy(),
                                         np.array([reward * args.reward_scaling]),
                                         next_state,
                                         np.array([done]),
                                         log_prob.detach().cpu().numpy()
                                         )
            agent.put_data(transition)
            score += reward
            attack_budget += reward if attack_flag else 0.
            if done or trunc:
                state_, _ = (env.reset())
                # input observation standardization
                state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5,
                                5)
                score_lst.append(score)
                budget_lst.append(attack_budget)
                if args.tensorboard:
                    writer.add_scalar("score/score", score, n_epi)
                    writer.add_scalar("score/attack_budget", attack_budget, n_epi)
                score = 0
                attack_budget = 0.0
            else:
                state = next_state
                state_ = next_state_
        # Training occurs after obtaining a trajectory.
        agent.train_net(n_epi)
        # Update observation state standardization variables.
        state_rms.update(np.vstack(state_lst))
        if n_epi % args.print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}, attack budget : {:.1f}, network difference : {:.1f}".format(n_epi,
                                                                sum(score_lst) / len(
                                                                    score_lst),
                                                                sum(budget_lst),
                                                                actor_network_comparison(agent, attack_objective) ) )
            score_lst = []
            budget_lst = []
        if n_epi % args.save_interval == 0 and n_epi != 0:
            torch.save(agent.state_dict(), './model_weights/agent_' + str(n_epi))

def actor_network_comparison(a: Tensor, b: Tensor, norm: int=2):
    with torch.no_grad():
        # Skip over the extra std parameters
        actor_a = list(a.actor.parameters())[1:]
        actor_b = list(b.actor.parameters())[1:]
        # Loop over networks
        difference = 0
        for a_layer, b_layer in zip(actor_a, actor_b):
            difference += torch.norm(a_layer - b_layer, p=norm)
    return difference

def poisoning_attack(objective,
                     action,
                     state,
                     reward,
                     delta: float = 0.1,
                     perturbation: float=-10):
    with torch.no_grad():
        # See what would the target policy would do deterministically.
        mu, sigma = objective.get_action(torch.from_numpy(np.array(state)).float().to(device))
        attack_action = mu
        # compare with what was done by the victim policy.
        action_distance = torch.norm(attack_action.cpu() - Tensor(action), 2)
        if action_distance > delta:
            # action taken too different from target by attacker
            reward_ = perturbation * action_distance
            attack = True
        else:
            # victim took correct action
            reward_ = reward
            attack = False
    # return perturbed reward unless victim took the correct action.
    return reward_, attack


if __name__ == '__main__':
    args, agent_args = parse_args()
    device, writer, env, state_rms, agent, attack_objective = setup_experiment(args, agent_args)
    run_experiment(agent_args, env, state_rms, agent, attack_objective, writer)
    print("done")
