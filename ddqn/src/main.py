#!/usr/bin/python

import sys
sys.path.append("../src/")
import gym
import random
import time
from config import *
from dddqn_agent import *
from dueling_dqn_network import *
from pong_wrapper import *
from process_image import *
from replay_buffer import *
from utilities import *

pong_wrapper = PongWrapper(ENV_NAME, NO_OP_STEPS)
print("The environment has the following {} actions: {}".format(pong_wrapper.env.action_space.n, pong_wrapper.env.unwrapped.get_action_meanings()))

MAIN_DQN = build_q_network(pong_wrapper.env.action_space.n, LEARNING_RATE, input_shape=INPUT_SHAPE)
TARGET_DQN = build_q_network(pong_wrapper.env.action_space.n, input_shape=INPUT_SHAPE)

replay_buffer = ReplayBuffer(size=MEMORY_SIZE, input_shape=INPUT_SHAPE)
dddqn_agent = DDDQN_AGENT(MAIN_DQN, TARGET_DQN, replay_buffer, pong_wrapper.env.action_space.n, 
                    input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE, 
                   replay_buffer_start_size=REPLAY_MEMORY_START_SIZE,
                   max_frames=MAX_FRAMES)

if PATH_LOAD_MODEL is not None:
    start_time = time.time()
    print('Loading model and info from the folder ', LOAD_FROM)
    info = dddqn_agent.load(LOAD_FROM, LOAD_REPLAY_BUFFER)

    # Apply information loaded from meta
    frame_number = info['frame_number']
    rewards = info['rewards']
    loss_list = info['loss_list']

    print(f'Loaded in {time.time() - start_time:.1f} seconds')
else:
    frame_number = 0
    rewards = []
    loss_list = []

def main():
    global frame_number, rewards, loss_list
    while frame_number < MAX_FRAMES:
        epoch_frame = 0
        while epoch_frame < EVAL_FREQUENCY:
            start_time = time.time()
            pong_wrapper.reset()
            episode_reward_sum = 0
            for _ in range(MAX_EPISODE_LENGTH):
                action = dddqn_agent.get_action(frame_number, pong_wrapper.state)
                processed_frame, reward, terminal = pong_wrapper.step(action)
                frame_number += 1
                epoch_frame += 1
                episode_reward_sum += reward

                # Add experience to replay memory
                dddqn_agent.add_experience(action=action,
                                     frame=processed_frame[:, :, 0], # shape 84x84, remove last dimension
                                     reward=reward, clip_reward=CLIP_REWARD,
                                     terminal=terminal)

                # Update agent
                if frame_number % UPDATE_FREQ == 0 and dddqn_agent.replay_buffer.count > REPLAY_MEMORY_START_SIZE:
                    loss, _ = dddqn_agent.learn(BATCH_SIZE, gamma=DISCOUNT_FACTOR)
                    loss_list.append(loss)

                # Update target network
                if frame_number % NETW_UPDATE_FREQ == 0 and frame_number > REPLAY_MEMORY_START_SIZE:
                    dddqn_agent.update_target_network()

                # Break the loop when the game is over
                if terminal:
                    terminal = False
                    break

            rewards.append(episode_reward_sum)

            wandb.log({'Game number': len(rewards), '# Frame': frame_number, '% Frame': round(frame_number / MAX_FRAMES, 2), "Average reward": round(np.mean(rewards[-10:]), 2), \
                      "Time taken": round(time.time() - start_time, 2)})
        # Evaluation
        terminal = True
        eval_rewards = []
        evaluate_frame_number = 0

        for _ in range(EVAL_LENGTH):
            if terminal:
                game_wrapper.reset(evaluation=True)
                life_lost = True
                episode_reward_sum = 0
                terminal = False

            action = dddqn_agent.get_action(frame_number, pong_wrapper.state, evaluation=True)

            # Step action
            _, reward, terminal = pong_wrapper.step(action)
            evaluate_frame_number += 1
            episode_reward_sum += reward

            # On game-over
            if terminal:
                eval_rewards.append(episode_reward_sum)

        if len(eval_rewards) > 0:
            final_score = np.mean(eval_rewards)
        else:
            # In case the first game is longer than EVAL_LENGHT
            final_score = episode_reward_sum
        # Log evaluation score
        wandb.log({'# Frame': frame_number, '% Frame': round(frame_number / MAX_FRAMES, 2), 'Evaluation score': final_score})

        # Save the networks, frame number, rewards and losses. 
        if len(rewards) > 500 and PATH_SAVE_MODEL is not None:
            dddqn_agent.save(f'{PATH_SAVE_MODEL}/save_agent_{time.strftime("%Y%m%d%H%M") + "_" + str(frame_number).zfill(8)}', \
                             frame_number=frame_number, rewards=rewards, loss_list=loss_list)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Save the model, I need this in order to save the networks, frame number, rewards and losses. 
        # if I want to stop the script and restart without training from the beginning
        if PATH_SAVE_MODEL is None:
            print("Setting path to ../model/")
            PATH_SAVE_MODEL = "../model/"
        print('Saving the model...')
        dddqn_agent.save(f'{PATH_SAVE_MODEL}/save_agent_{time.strftime("%Y%m%d%H%M") + "_" + str(frame_number).zfill(8)}', \
                             frame_number=frame_number, rewards=rewards, loss_list=loss_list)
        print('Saved.')
