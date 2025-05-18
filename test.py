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

ACTION_NAMES = ["subnet_scan", "os_scan", "service_scan", "process_scan", "e_ssh", "e_ftp", "e_samba", "e_smtp", EXPLOIT_HTTP: "e_http", 
                PRIVI_ESCA_TOMCAT: "pe_tomcat", PRIVI_ESCA_PROFTPD: "pe_daclsvc", PRIVI_ESCA_CRON: "pe_schtask"]

def select_action(action_space, action_name, action_target):
    for i in range(0, action_space.n):
        action = action_space.get_action(i)
        if action.name == action_name and action.target == action_target:
            return action


agent = Heuristic(observation_space_shape= env.observation_space.shape, action_space_n= env.action_space.n, host_vector_count= 4)
i = 0
actions = [4, 2, ]
while not done or step_limit:

    #action = int(input())
    #temp = select_action(env.action_space, "e_ssh", (1,0))
"""     action = agent.choose_action(obs, info)
    print(action)
    new_obs, reward, done, step_limit, info = env.step(action)
    print(info)

    print(new_obs)
    obs = new_obs
    full_reward += reward
 """
print(f"Total reward: {full_reward}")