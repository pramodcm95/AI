This Repository consists of a solution to a simple gym-platform environment (https://github.com/cycraig/gym-platform.git)

The detais of environment are as follows:
1) The environment "gym-platform" here is a parametric action type environment, which means we will have both discrete and continuous actions
2) The environment here has 3 platforms seperted by 2 gaps between first and second platform, second and third platform respectively. 
3) Our Agent is in the beginning, i.e., on the left most past of 1st platfrom. There are enemy's on both first and second platform who are moving around back and forth(on the same platform, not beyond) with certain velocity in horizontal direction. They can only move forward or backward
4) While our agent is capable of more, three actions to be precise, either to run, hop or leap.
5) The agent's task is to run across the platforms and reach the end of last platform which is decided as an agent's victory
6) However, if the agent find's itself crashing into enemies(in this case, being in the same position on the platfrom as of enemies), or fail to jump across the gaps between two platforms(i.e., landing in a non-platform space) would be considered as end of an episode and agent would loose the challenge
7) The state space design of this environment was already sophesticated in terms of the information given to the agent. Hence no additional feature engineering has been implemented
8) The environment's reward can be better in future, which is now distance based reward normalized between 0 and 1. 

Reinforcement Learning(RL) to the rescue :-) :


Although this seems to be an easy game in first look, there is still some randomness in the process. The agent should learn to pass this challenge most of the times by itself autonomusly. Reinforcement Learning is a perfect candiate for this set up. As this environment can be seen as an MDP whose solution will be learnt by an RL agent.


The design options were many, although parametric action spaces are relatively new. I could have used a custom model with stable baslines with one output for each type of action. But, ray[rllib] already had the support to parametric action spaces which exactly what we want in this setup. I have used ray[rllib] as the reinforcement library. I have used various applicable RL algorithms before finalising use of PPO(proximal Policy Optimization) for providing solution to win the above game.


Instructions to Run:

# 1) Using Docker File:
     docker build -f DockerFile .
     docker run -it <image created above>
# 2) Using Git:
     Clone to this repo using: git clone https://github.com/pramodcm95/int_task_RL.git
     cd gym_platform_RL
     pip install -r requirements.txt
     git clone https://github.com/cycraig/gym-platform.git
     PPO_gym_platform.py
  
