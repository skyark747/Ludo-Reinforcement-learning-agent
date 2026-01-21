import random
from tqdm import tqdm
from ludo import Ludo
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

STARTING = -1
DESTINATION = 56
SAFE_SQUARES = [0, 8, 13, 21, 25, 26, 34, 39, 47, 51, 52, 53, 54, 55, 56]
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)


class ActorCritic(torch.nn.Module):
    def __init__(self, actor_input_dim,critic_input_dim,actor_output_dim,critic_output_dim):
        super().__init__()

        self.actor= torch.nn.Sequential(
            torch.nn.Linear(actor_input_dim,256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256,256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256,actor_output_dim),

        )

        self.critic= torch.nn.Sequential(
            torch.nn.Linear(critic_input_dim,256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256,256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256,critic_output_dim),

        )

        self.optimizer=torch.optim.AdamW(self.parameters(),lr=0.00009,betas=(0.9,0.999),eps=1e-8,weight_decay=0.01)
        self.scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,30000,1e-6)

    def forward_actor(self,state,device=None):

        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32)

        if device:
            state=state.to(device)
        else:
            state=state.to(DEVICE)

        actions_prob = self.actor(state)
        return actions_prob

    def forward_critic(self,state,device=None):

        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32)
        if device:
            state=state.to(device)
        else:
            state=state.to(DEVICE)
        value = self.critic(state)
        return value


def get_win_percentages(n, policy1, policy2):
    env = Ludo(render_mode="")
    wins = [0, 0]
    policies = [policy1, policy2]

    for i in range(2):  
        for _ in tqdm(range(n // 2)):

            state = env.reset()
            terminated = False
            player_turn = 0

            while not terminated:
                action_space = env.get_action_space()
                action = policies[player_turn].get_action(state, action_space)

                state = env.step(action)
                terminated, player_turn = state[3], state[4]

            # Update win count for correct player
            wins[player_turn - i] += 1

        # Swap policies so each one gets equal first-turn advantage
        policies[0], policies[1] = policies[1], policies[0]

    win_percentages = [(win / n) * 100 for win in wins]

    return wins, win_percentages


def plot_win_graph(wins, win_percentages, labels=("Policy 1", "Policy 2")):
    plt.figure(figsize=(7,5))
    
    plt.bar(labels, win_percentages)
    plt.ylabel("Win Percentage (%)")
    plt.title("Policy Win Comparison")
    for i, pct in enumerate(win_percentages):
        plt.text(i, pct + 1, f"{pct:.1f}%", ha='center')

    plt.ylim(0, 100)
    plt.show()


class Policy_Random:
    def get_action(self, state, action_space):
        if action_space:
            return random.choice(action_space)
        return None


class policy_crow_shadow():
    def __init__(self):

        self.actor=ActorCritic(8,11,1,1)
        
        # input your best .pth file
        best_weight=torch.load(r"best4.pth")
        self.actor.load_state_dict(best_weight['model_state_dict'])

    def get_action(self,state,action_space):
        if len(action_space)==0:
            return None

        state_action_feats,_=self.get_state_values(state,action_space)
        self.actor.eval()
        with torch.no_grad():
               prob=self.actor.forward_actor(state_action_feats,'cpu')

               prob=prob.squeeze(-1)

        prob=torch.softmax(prob,dim=0)
        action_idx = torch.argmax(prob).item()
        action = action_space[action_idx]

        return action

    def get_state_action_features(self,state,action):

        dice_index,goti_index=action

        red_gotis,yellow_gotis,dice_roll,terminated,player_id=state

        if int(player_id)==0:
            my_goti=red_gotis.gotis[goti_index]
            dushman_goti=yellow_gotis.gotis
        else:
            my_goti=yellow_gotis.gotis[goti_index]
            dushman_goti=red_gotis.gotis

        dice_value=dice_roll[dice_index]

        my_goti_pos=my_goti.position
        old_goti_pos=my_goti_pos

        if my_goti_pos==-1 and dice_value==6:
            my_goti_pos=0
        else:
            my_goti_pos=my_goti_pos+dice_value if my_goti_pos+dice_value<=DESTINATION else my_goti_pos


        # is in danger
        is_in_danger=0
        for dushman in dushman_goti:
            if dushman.position != -1:
                if 1<=my_goti_pos-dushman.position<=6:
                    is_in_danger=1
                    break


        # can kill
        can_kill=0
        for dushman in dushman_goti:
            if dushman.position!=-1 and my_goti_pos+dice_value == dushman.position:
                can_kill=1
                break

        # is safe
        is_safe=0
        if my_goti_pos in SAFE_SQUARES:
            is_safe=1


        # distance from home
        dist_home=1
        if my_goti_pos!=-1:
            dist_home=(DESTINATION-my_goti_pos)/DESTINATION



        # is on home path
        is_home_path=0
        if (DESTINATION-my_goti_pos) <=5:
            is_home_path=1


        # can enter board
        can_enter=0
        if old_goti_pos == -1 and dice_value==6:
            can_enter=1


        # progress
        progress=0
        if my_goti_pos !=-1:
            progress=(DESTINATION-my_goti_pos)/DESTINATION


        return [is_in_danger,can_kill,is_safe,dist_home,is_home_path,can_enter,progress,dice_value/6]

    def get_state_features(self,gotis):

        g_home,g_goal,g_safe,min_dist,max_dist=0,0,0,float('inf'),float('-inf')
        for g in gotis.gotis:
            if g.position == -1:
                max_dist=0
                min_dist=0
            if g.position == DESTINATION:
                min_dist=0
                max_dist=0

            if g.position == -1:
                g_home+=1

            if g.position in SAFE_SQUARES:
                g_safe+=1

            if g.position == DESTINATION:
                g_goal+=1

            if g.position != -1 and g.position not in SAFE_SQUARES:
                dist=(DESTINATION-g.position)/56.0

                if dist < min_dist:
                    min_dist=dist

                elif dist > max_dist:
                    max_dist=dist

        return g_home,g_goal,g_safe,min_dist,max_dist

    def get_state_values(self,state,action_space):

        gotis_red,gotis_yellow,dice_roll,terminated,player_turn=state

        r_home,r_goal,r_safe,r_min_dist,r_max_dist=self.get_state_features(gotis_red)
        y_home,y_goal,y_safe,y_min_dist,y_max_dist=self.get_state_features(gotis_yellow)

        r_active=4-(r_home+r_goal)
        y_active=4-(y_home+y_goal)

        # whether any 6 exists in the roll
        has_six = 1 if 6 in dice_roll else 0

        # number of sixes in the roll
        num_sixes = dice_roll.count(6)

        # number of rolls available right now
        num_rolls_available = len(dice_roll)


        state_key=[r_home,r_goal,r_active,r_safe,y_home,y_goal,y_active,y_safe,has_six,num_sixes,num_rolls_available]

        state_key=np.array(state_key)

        state_key=(state_key-np.mean(state_key)/np.std(state_key))
        

        features=[]
        for a in action_space:
            features.append(self.get_state_action_features(state,a))

        features=np.array(features)

        features=(features-np.mean(features)/np.std(features))

        return features,state_key


# Example usage:
wins, win_percentages = get_win_percentages(1000,policy_crow_shadow(),Policy_Random())
print("Wins:", wins)
print("Win Percentages:", win_percentages)

# plot_win_graph(wins, win_percentages)

