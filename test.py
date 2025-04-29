import nasim
import matplotlib.pyplot as plt
from agents.heuristic_agent import Heuristic

env = nasim.make_benchmark("tiny", fully_obs=True, flat_actions=True, render_mode="human")

obs, info = env.reset()
print(info)
full_reward = 0

done = False
step_limit = False
print(obs)
#env.render()

agent = Heuristic(observation_space_shape= env.observation_space.shape, action_space_n= env.action_space.n, host_vector_count= 4)

while not (done or step_limit):

    #action = int(input())
    action = agent.choose_action(obs, info)
    print(action)
    new_obs, reward, done, step_limit, info = env.step(action)
    print(info)

    print(new_obs)
    obs = new_obs
    full_reward += reward

print(f"Total reward: {full_reward}")