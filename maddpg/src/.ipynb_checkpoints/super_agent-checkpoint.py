
class SuperAgent:
    def __init__(self, env):
        self.replay_buffer = ReplayBuffer(env)
        self.n_agents = len(env.agents)
        self.agents = [Agent(env, agent) for agent in range(self.n_agents)]
        
    def get_actions(self, agents_states):
        list_actions = [self.agents[index].get_actions(agents_states[index]) for index in range(self.n_agents)]
        return list_actions
    
    def save_agents(self):
        for agent in self.agents:
            agent.save()
    
    def load_agents(self):
        for agent in self.agents:
            agent.load()
    
    def train(self):
        if self.replay_buffer.check_buffer_size() == False:
            return
        
        state, reward, next_state, done, actors_state, actors_next_state, actors_action = self.replay_buffer.get_minibatch()
        
        states = tf.convert_to_tensor(state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_state, dtype=tf.float32)
        
        actors_states = [tf.convert_to_tensor(s, dtype=tf.float32) for s in actors_state]
        actors_next_states = [tf.convert_to_tensor(s, dtype=tf.float32) for s in actors_next_state]
        actors_actions = [tf.convert_to_tensor(s, dtype=tf.float32) for s in actors_action]
        
        with tf.GradientTape(persistent=True) as tape:
            target_actions = [self.agents[index].target_actor(actors_next_states[index]) for index in range(self.n_agents)]
            policy_actions = [self.agents[index].actor(actors_states[index]) for index in range(self.n_agents)]
            
            concat_target_actions = tf.concat(target_actions, axis=1)
            concat_policy_actions = tf.concat(policy_actions, axis=1)
            concat_actors_action = tf.concat(actors_actions, axis=1)
            
            target_critic_values = [tf.squeeze(self.agents[index].target_critic(next_states, concat_target_actions), 1) for index in range(self.n_agents)]
            critic_values = [tf.squeeze(self.agents[index].critic(states, concat_actors_action), 1) for index in range(self.n_agents)]
            targets = [rewards[:, index] + self.agents[index].gamma * target_critic_values[index] * (1-done[:, index]) for index in range(self.n_agents)]
            critic_losses = [tf.keras.losses.MSE(targets[index], critic_values[index]) for index in range(self.n_agents)]
            
            actor_losses = [-self.agents[index].critic(states, concat_policy_actions) for index in range(self.n_agents)]
            actor_losses = [tf.math.reduce_mean(actor_losses[index]) for index in range(self.n_agents)]
        
        critic_gradients = [tape.gradient(critic_losses[index], self.agents[index].critic.trainable_variables) for index in range(self.n_agents)]
        actor_gradients = [tape.gradient(actor_losses[index], self.agents[index].actor.trainable_variables) for index in range(self.n_agents)]
        
        for index in range(self.n_agents):
            self.agents[index].critic.optimizer.apply_gradients(zip(critic_gradients[index], self.agents[index].critic.trainable_variables))
            self.agents[index].actor.optimizer.apply_gradients(zip(actor_gradients[index], self.agents[index].actor.trainable_variables))
            self.agents[index].update_target_networks(self.agents[index].tau)
