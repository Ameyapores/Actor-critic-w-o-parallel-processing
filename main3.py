from model3 import ActorCritic
import gym
import argparse
import torch.optim as optim
import torch
from torch.autograd import Variable
import math
import numpy as np
import matplotlib.pyplot as plt
from setup_env import setup_env
import torch.nn.functional as F
from scipy.misc import imresize
import os
SAVEPATH = os.getcwd() + '/save/mario_a3c_params.pkl'
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--use-cuda',default=True,
                    help='run on gpu.')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--tau', type=float, default=1.00,
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=250,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--env-name', default='SuperMarioBros-1-1-v0',
                    help='environment to train on (default: SuperMarioBros-1-4-v0)')
parser.add_argument('--save-path',default=SAVEPATH,
                    help='model save interval (default: {})'.format(SAVEPATH))
args = parser.parse_args()                   


env = setup_env(args.env_name)

obs_dim= env.observation_space.shape[2]
act_dim= env.action_space.n

model = ActorCritic(obs_dim, act_dim)

if args.use_cuda:
        model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr)
FloatTensor = torch.cuda.FloatTensor if args.use_cuda else torch.FloatTensor
model.train()

prepro = lambda img: imresize(img[0:84].mean(2), (84,84)).astype(np.float32).reshape(1,84,84)/255.

ep_rew_avg = 0
losses =[]
gamma = 0.99
r_list = []
max_eps = 1000000
max_steps = 500
ep_numb = 0
ep_avgs_list = []
done = True
while ep_numb < max_eps:
    values, log_probs, rewards, entropies = [], [], [], []
    ep_numb+=1
    state= prepro(env.reset())
    state = torch.from_numpy(state)
    episode_reward = 0

    if ep_numb % 100 == 0 and ep_numb > 0:  
        torch.save(model.state_dict(), args.save_path)

    model.load_state_dict(model.state_dict())
    if done:
        cx = Variable(torch.zeros(1, 512)).type(FloatTensor)
        hx = Variable(torch.zeros(1, 512)).type(FloatTensor)
    else:
        cx = Variable(cx.data).type(FloatTensor)
        hx = Variable(hx.data).type(FloatTensor)

    for step in range(max_steps):
        state_inp = Variable(state.unsqueeze(0)).type(FloatTensor)
        value, logit, (hx, cx) = model((state_inp, (hx, cx)), False)
        prob = F.softmax(logit, dim=-1)
        log_prob = F.log_softmax(logit, dim=-1)
        entropy = -(log_prob * prob).sum(-1, keepdim=True)
        entropies.append(entropy)
        action = prob.max(-1, keepdim=True)[1].data
        action_out = action.to(torch.device("cpu"))
        log_prob = log_prob.gather(-1, Variable(action))
        log_probs.append(log_prob)

        state, reward, done, _ = env.step(action_out.numpy()[0][0])
        done = False
        reward = max(min(reward, 1), -1)
        #env.render()

        if done:
            episode_length = 0
            state = torch.from_numpy(prepro(env.reset()))

        state = torch.from_numpy(prepro(state))
        values.append(value)
        log_probs.append(log_prob)
        rewards.append(reward)
        episode_reward += reward
        if done:
            break

    R = torch.zeros(1, 1)
    if not done:
        state_inp = Variable(state.unsqueeze(0)).type(FloatTensor)
        value, _, _ = model((state_inp, (hx, cx)), False)
        R = value.data

    ep_rew_avg = (ep_rew_avg*(ep_numb-1))/(ep_numb)  + episode_reward/(ep_numb)
    ep_avgs_list.append(ep_rew_avg)
    rewards = torch.from_numpy(np.array(rewards)).type(FloatTensor)

    if ep_numb % 100==0 and not ep_numb == 0:
        print(" ep rew avg {}, ep reward {}, episode_number {}".format(ep_rew_avg, episode_reward, ep_numb))
    
    values.append(Variable(R).type(FloatTensor))
    policy_loss = 0
    value_loss = 0
    R = Variable(R).type(FloatTensor)
    gae = torch.zeros(1, 1).type(FloatTensor)

    for i in reversed(range(len(rewards))):
        R = gamma * R + rewards[i]
        advantage = R - values[i]
        value_loss = value_loss + 0.5 * advantage.pow(2)

        delta_t = rewards[i] + gamma * \
            values[i + 1].data - values[i].data
        gae = gae * gamma * args.tau + delta_t

        policy_loss = policy_loss - \
            log_probs[i] * Variable(gae).type(FloatTensor) - args.entropy_coef * entropies[i]

    total_loss = policy_loss + args.value_loss_coef * value_loss
    losses.append(total_loss)
    optimizer.zero_grad()

    (total_loss).backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

    optimizer.step()
    if ep_numb % 50 == 0 and not ep_numb == 0:
        plt.plot(losses)
        plt.title("Critic losses")
        plt.show()
        plt.plot(ep_avgs_list)
        plt.title("Average rewards")
        plt.show()
