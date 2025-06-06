import nasim
from agents.qlearning_agent import Q_learning
import matplotlib.pyplot as plt
import time
import logging
from datetime import datetime
from statistics import fmean, stdev


SCENARIO_NAME = "tiny"

def train_agent(n_episodes=1200):
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

    print(f"Starting training Q_learning agent on {n_episodes} episodes.")
    env = nasim.make_benchmark(SCENARIO_NAME, fully_obs=True, flat_actions=True, render_mode="human")

    agent = Q_learning(observation_space_shape= env.observation_space.shape, action_space_n= env.action_space.n)

    scores = []
    start = time.time()
    
    for i in range(n_episodes):
        terminated = False
        truncated = False
        state, _ = env.reset()
        score = 0
        while not (terminated or truncated):           

            state = state.astype(int)
            action = agent.choose_action(state)

            new_state, reward, terminated, truncated, _ = env.step(action)
            new_state = new_state.astype(int)

            agent.learn(state, action, reward, new_state)

            state = new_state
            
            score += reward
        
        agent.update_epsilon(i)
        scores.append(score)

        log_message = 'Episode {} in {:.2f} min. Expected total time for {} episodes: {:.0f} min. [{:.2f}]'.format(
            (i+1), 
            (time.time() - start)/60, 
            n_episodes, 
            (((time.time() - start)/(i+1))*n_episodes)/60, 
            score,
        )
        
        logger.info(log_message)
    
    logger.info("Training completed!")
    return agent, scores


def evaluate_agent(agent, n_episodes):
    env = nasim.make_benchmark(SCENARIO_NAME, fully_obs=True, flat_actions=True, render_mode="human")

    scores = []
    trajectory_steps =[]

    for i in range(n_episodes):
        terminated = False
        truncated = False
        state, _ = env.reset()
        score = 0
        steps = 0
        while not (terminated or truncated):

                action = agent.choose_action(state)
                new_state, reward, terminated, truncated, _ = env.step(action)

                state = new_state

                steps += 1
                score += reward
    
        scores.append(score)
        trajectory_steps.append(steps)

    return scores, trajectory_steps



def make_plot(training_scores, is_eval=False):
    if not is_eval:
        plt.plot(training_scores, label="награда за траекторию")
        plt.title("Динамика наград за траекторию в процессе обучения")
        plt.ylabel("Награда")
        plt.xlabel("Эпизод обучения")
        plt.grid(True)
        #plt.axis([0, 30, 0, 1200])
        plt.legend()
        plt.show()
    
    else:
        plt.plot(training_scores, label="награда за траекторию")
        plt.title("Награды за траекторию в процессе тестирования")
        plt.ylabel("Награда")
        plt.xlabel("Эпизод тестирования")
        plt.grid(True)
        #plt.axis([0, 30, 0, 1200])
        plt.legend()
        plt.show()

def main():
    agent, training_scores = train_agent()
    make_plot(training_scores)

    evaluating_scores, trajectory_steps = evaluate_agent(agent, 200)

    make_plot(evaluating_scores, is_eval=True)
    
    print("Среднее значение награды и среднее число шагов")
    print(fmean(evaluating_scores), fmean(trajectory_steps))
    print("стандартное отклонение награды и числа шагов")
    print(stdev(evaluating_scores), stdev(trajectory_steps))

if __name__ == '__main__':
    main()



