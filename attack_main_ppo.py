from configparser import ConfigParser
from argparse import ArgumentParser, Namespace

import torch
import gymnasium as gym
import numpy as np
import os

from agents.ppo import PPO

from utils.utils import make_transition, Dict, RunningMeanStd

from typing import Tuple, Union, List, Callable
import pickle

Tensor = torch.Tensor
TBWriter = Callable
device = None
Parser = Namespace
Distribution = torch.distributions.Distribution
Network = torch.nn.Module

def parse_args():
    """ Create argument parsers """
    parser = ArgumentParser('parameters')

    # Basic experiment configuration
    base_experiment = parser.add_argument_group('Base experiment paramenters')
    base_experiment.add_argument("--env_name", type=str, default='HalfCheetah-v2', help="'Ant-v2','HalfCheetah-v2','Hopper-v2','Humanoid-v2','HumanoidStandup-v2',\
              'InvertedDoublePendulum-v2', 'InvertedPendulum-v2' (default : Hopper-v2)")

    base_experiment.add_argument("--algo", type=str, default='ppo',
                        help='algorithm to adjust (default : ppo)')
    base_experiment.add_argument('--train', type=bool, default=True, help="(default: True)")
    base_experiment.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs, (default: 1000)')
    base_experiment.add_argument('--standardize_state', action='store_true', default=False,
                             help="Flag to perform standardization of states")

    # Adversarial attack configuration
    adversarial = parser.add_argument_group('Adversarial attack paramenters')
    adversarial.add_argument('--attack', action='store_true', default=False,
                        help="Perform adversarial attack on algorithm")
    adversarial.add_argument('--perturbation', type=float, default=-10,
                             help='Perturbation weight amount.')
    adversarial.add_argument('--target_radius', type=float, default=0.1,
                             help='Radius around attack desired action where victim action is considered "fooled".')

    # Misc
    misc = parser.add_argument_group('Misc paramenters')
    misc.add_argument('--file', type=str, default='experiment_result.pkl',
                      help='Exeperiment plottable data output file (pickle file.)')

    misc.add_argument('--render', type=bool, default=False, help="(default: False)")

    misc.add_argument('--tensorboard', type=bool, default=False,
                        help='use_tensorboard, (default: False)')
    misc.add_argument("--load", type=str, default='no',
                        help='load network name in ./model_weights')
    misc.add_argument("--load_target", type=str, default='no',
                      help='load network name in ./model_weights to use as adversarial target')
    misc.add_argument("--save_interval", type=int, default=100,
                        help='save interval(default: 100)')
    misc.add_argument("--print_interval", type=int, default=1,
                        help='print interval(default : 20)')
    misc.add_argument("--use_cuda", type=bool, default=True, help='cuda usage(default : True)')
    misc.add_argument("--reward_scaling", type=float, default=1,
                        help='reward scaling(default : 1)')

    args = parser.parse_args()

    # Get configuration for experiment
    parser = ConfigParser()
    parser.read('config.ini')
    agent_args = Dict(parser, args.algo)

    return args, agent_args

def setup_experiment(args,
                     agent_args) -> Tuple[str, Union[TBWriter, None], gym.Env, RunningMeanStd, PPO, PPO]:
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
    # exclude_current_positions_from_observation allows algorithm have access to x_coord
    env = gym.make(args.env_name,
                   render_mode='human' if args.render else None,
                   exclude_current_positions_from_observation=False)
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
    if args.load_target != 'no':
        attack_objective.load_state_dict(torch.load("./model_weights/" + args.load_target))

    return device, writer, env, state_rms, agent, attack_objective

def standardize_state(args,
                      state_,
                      state_rms):
    if args.standardize_state:
        return np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
    return state

def run_experiment(args,
                   agent_args,
                   env,
                   state_rms,
                   agent,
                   attack_objective,
                   writer) -> Tuple[List[float], List[float], List[float], List[float], List[Tensor], List[Tensor], List[int]]:
    """ Execute experiment, by training and storing rewards """

    # Lists used for storing ongoing performance during training
    score_lst = []
    budget_lst = []
    state_lst = []
    # Lists used for storing performance for plotting purposes
    score_record = []
    budget_record = []
    std_record = []
    network_diff_record = []
    kl_div_record = []
    logit_div_record = []
    attack_record = []
    action_dist_record = []

    score = 0.0
    attack_budget = 0.0
    attack_timesteps = 0
    # Get initial state
    state_, _ = (env.reset())
    # input observation standardization
    state = state_#np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
    for n_epi in range(args.epochs):
        state_buffer = []
        std_buffer = []
        attack_budget = 0.0
        attack_timesteps = 0
        action_distance = 0.
        for t in range(agent_args.traj_length): # ~5,000
            if args.render:
                env.render()

            state_lst.append(state_)
            state_buffer.append(state)
            # Obtain action by random sampling from generated distribution
            # Note that state input was standardized
            mu, sigma = agent.get_action(torch.from_numpy(np.array(state)).float().to(device))
            # Store standard deviation for current trajectory.
            std_buffer.append(float(sigma.mean()))
            dist = torch.distributions.Normal(mu, sigma[0])  # Stochastic Actions!
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            # Apply action. Note that reward will be scaled (see argument parser)
            next_state_, reward, done, trunc, info = env.step(action.cpu().numpy())

            if args.attack:
                # Apply reward poisoning attack
                # NOTE: Temporarilly replaced action.cpu().numpy() for mu.detach().cpu().numpy()
                reward, attack_flag, perturbation, action_diff = poisoning_attack(attack_objective, action.cpu().numpy(), state, reward, delta=args.target_radius, perturbation=args.perturbation)
            else:
                attack_flag = False
            # next state observation standardization
            next_state = next_state_#np.clip((next_state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
            # Creates a dictionary with state, action, reward, next_state, log_prob, and done
            transition = make_transition(state,
                                         action.cpu().numpy(),
                                         np.array([reward * args.reward_scaling]),
                                         next_state,
                                         np.array([done or trunc]),
                                         log_prob.detach().cpu().numpy()
                                         )
            agent.put_data(transition)
            score += float(reward)
            attack_budget += float(perturbation) if attack_flag else 0.
            attack_timesteps += int(attack_flag)
            action_distance += float(action_diff) if args.attack else 0.

            if done or trunc:
                state_, _ = (env.reset())
                # input observation standardization
                state = state_#np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
                score_lst.append(score)
                budget_lst.append(attack_budget)
                if args.tensorboard:
                    writer.add_scalar("score/score", score, n_epi)
                    writer.add_scalar("score/attack_budget", attack_budget, n_epi)
                score = 0
            else:
                state = next_state
                state_ = next_state_
        # Training occurs after obtaining a trajectory.
        agent.train_net(n_epi)
        # Update observation state standardization variables.
        state_rms.update(np.vstack(state_lst))
        if n_epi % args.print_interval == 0 and n_epi != 0:
            # Store metrics for plotting purposes
            score_record.append(sum(score_lst) / len(score_lst))
            budget_record.append(sum(budget_lst))
            network_diff_record.append(actor_network_comparison(agent, attack_objective) if args.attack else 0)
            std_record.append(sum(std_buffer) / len(std_buffer))
            kl_div_record.append(kl_divergence(agent, attack_objective, state_buffer) if args.attack else 0)
            logit_div_record.append(logit_divergence(agent, attack_objective, state_buffer) if args.attack else 0)
            attack_record.append(attack_timesteps)
            action_dist_record.append(action_distance)
            # Print out performance
            print(f"episode: {n_epi},\t avg score: {score_record[-1]:.1f},\t attack budget: {budget_record[-1]:.1f},\t network difference: {network_diff_record[-1]:.1f},\t KL-Div: {kl_div_record[-1]:.2f},\t Logit-Div: {logit_div_record[-1]:.2f},\t std : {std_record[-1]:.2f}\t attacks: {attack_timesteps}/{agent_args.traj_length} @ {action_distance / agent_args.traj_length:.2f}")
            # Clear lists (used only for printing out performance)
            score_lst = []
            budget_lst = []
        if n_epi % args.save_interval == 0 and n_epi != 0:
            torch.save(agent.state_dict(), './model_weights/agent_' + str(n_epi))
    return score_record, budget_record, network_diff_record, std_record, kl_div_record, logit_div_record, attack_record, action_dist_record

def actor_network_comparison(a: Tensor, b: Tensor, norm: int=2):
    with torch.no_grad():
        # Skip over the extra std parameters
        actor_a = list(a.actor.parameters())[1:]
        actor_b = list(b.actor.parameters())[1:]
        # Loop over networks
        difference = 0
        for a_layer, b_layer in zip(actor_a, actor_b):
            difference += torch.norm(a_layer - b_layer, p=norm)
    return difference.item()

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
    return reward_, attack, perturbation*action_distance, action_distance

def store_results(file_name: str,
                  score: List[float],
                  budget: List[float],
                  network_diff: List[float],
                  std_record: list[float],
                  kl_diff: List[Tensor],
                  logit_diff: List[Tensor],
                  attack_record: List[int],
                  action_dist_record: List[float],
                  args: Parser,
                  agent_args: Parser
                  ) -> None:
    ''' Store experiment results '''
    cucumbers = {
        'score': score,
        'budget': budget,
        'network_diff': network_diff,
        'std_record': std_record,
        'kl_diff': kl_diff,
        'logit_diff': logit_diff,
        'attack_record': attack_record,
        'action_dist_record': action_dist_record,
        'args': args,
        'agent_args': agent_args
    }
    with open(file_name, 'wb') as jar:
        pickle.dump(cucumbers, jar)

def kl_divergence(victim: PPO,
                  target: PPO,
                  states: List[np.ndarray]) -> Tensor:
    kl_div = None
    for n, state in enumerate(states):
        # Build Distributions
        mu_v, sigma_v = victim.get_action(torch.from_numpy(np.array(state)).float().to(device))
        mu_t, sigma_t = target.get_action(torch.from_numpy(np.array(state)).float().to(device))
        dist_v = torch.distributions.Normal(mu_v, sigma_v[0])
        dist_t = torch.distributions.Normal(mu_t, sigma_t[0])

        # Compute KL-Divergence and return
        div = torch.distributions.kl.kl_divergence(dist_t, dist_v)
        if n == 0 or div.max() > kl_div.max():
            kl_div = div
    return float(kl_div.max())

def logit_divergence(victim: PPO,
                     target: PPO,
                     states: List[np.ndarray]) -> Tensor:
    tot_div = 0.
    for n, state in enumerate(states):
        # Obtain output difference
        mu_v, sigma_v = victim.get_action(torch.from_numpy(np.array(state)).float().to(device))
        mu_t, sigma_t = target.get_action(torch.from_numpy(np.array(state)).float().to(device))
        # Compute L2 difference (Set as Maximum)
        div = torch.norm(mu_t - mu_v, p=2)
        # Compute cumulative divergence
        tot_div += div
    return float(tot_div / len(states))


if __name__ == '__main__':
    args, agent_args = parse_args()
    device, writer, env, state_rms, agent, attack_objective = setup_experiment(args, agent_args)
    score_record, budget_record, network_diff_record, std_record, kl_div_record, logit_div_record, attack_record, action_dist_record = run_experiment(args, agent_args, env, state_rms, agent, attack_objective, writer)
    store_results(args.file,
                  score_record,
                  budget_record,
                  network_diff_record,
                  std_record,
                  kl_div_record,
                  logit_div_record,
                  attack_record,
                  action_dist_record,
                  vars(args),
                  dict(agent_args))

    print("done")
