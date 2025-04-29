import nasim
import matplotlib.pyplot as plt
import time
import logging
from datetime import datetime

from sb3_contrib import RecurrentPPO
from stable_baselines3 import A2C


LEARN_EVERY = 4

def train_agent(n_episodes=20):
    # Настраиваем логирование
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'logs/training_log_{timestamp}.txt'
    
    # Создаем форматтер для логов
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    
    # Настраиваем файловый handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    
    # Настраиваем logger
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    print(f"Starting training DDQN agent.")
    env = nasim.make_benchmark("tiny", fully_obs=True, render_mode="human")
    #env = nasim.make_benchmark("tiny", fully_obs=False )

    agent = A2C("MlpPolicy", env)
    agent.learn(20000, progress_bar=True)
            
    scores = []
    start = time.time()
    
    for i in range(n_episodes):
        terminated = False
        step_limit = False
        obs, _ = env.reset()
        score = 0
        steps = 0
        while not (step_limit or terminated):

                action, _states = agent.predict(observation=obs, deterministic=True)
                new_obs, reward, terminated, step_limit, _ = env.step(action)

                obs = new_obs

                #env.render()
                steps += 1
                score += reward
    
        scores.append(score)

        log_message = 'Episode {} in {:.2f} min. Expected total time for {} episodes: {:.0f} min. [{:.2f}]'.format(
            (i+1), 
            (time.time() - start)/60, 
            n_episodes, 
            (((time.time() - start)/(i+1))*n_episodes)/60, 
            score,
        )
        
        logger.info(log_message)
    
    logger.info("Testing completed!")
    return agent, scores

def evaluate_agent():
    pass

def main():
    agent, scores = train_agent()

    plt.plot(scores, label="iteration reward")
    plt.title("Reward dependence on the number of training iterations")
    plt.ylabel("Reward")
    plt.xlabel("training iterations")
    plt.grid(True)
    #plt.axis([0, 30, 0, 1200])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
