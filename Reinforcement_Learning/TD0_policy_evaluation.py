import numpy as np
from rl_glue import RLGlue
from Agent import BaseAgent
from Environment import BaseEnvironment
from manager import Manager
from itertools import product


class CliffWalkEnvironment(BaseEnvironment):
    def env_init(self, env_info={}):
        """Setup for the environment called when the experiment first starts.
        Note:
            Initialize a tuple with the reward, first state, boolean
            indicating if it's terminal.
        """

        # Note, we can setup the following variables later, in env_start() as it is equivalent.
        # Code is left here to adhere to the note above, but these variables are initialized once more
        # in env_start() [See the env_start() function below.]

        reward = None
        state = None  # See Aside
        termination = None
        self.reward_state_term = (reward, state, termination)

        # AN ASIDE: Observation is a general term used in the RL-Glue files that can be interachangeably
        # used with the term "state" for our purposes and for this assignment in particular.
        # A difference arises in the use of the terms when we have what is called Partial Observability where
        # the environment may return states that may not fully represent all the information needed to
        # predict values or make decisions (i.e., the environment is non-Markovian.)

        # Set the default height to 4 and width to 12 (as in the diagram given above)
        self.grid_h = env_info.get("grid_height", 4)
        self.grid_w = env_info.get("grid_width", 12)

        # Now, we can define a frame of reference. Let positive x be towards the direction down and
        # positive y be towards the direction right (following the row-major NumPy convention.)
        # Then, keeping with the usual convention that arrays are 0-indexed, max x is then grid_h - 1
        # and max y is then grid_w - 1. So, we have:
        # Starting location of agent is the bottom-left corner, (max x, min y).
        self.start_loc = (self.grid_h - 1, 0)
        # Goal location is the bottom-right corner. (max x, max y).
        self.goal_loc = (self.grid_h - 1, self.grid_w - 1)

        # The cliff will contain all the cells between the start_loc and goal_loc.
        self.cliff = [(self.grid_h - 1, i)
                      for i in range(1, (self.grid_w - 1))]

        # Take a look at the annotated environment diagram given in the above Jupyter Notebook cell to
        # verify that your understanding of the above code is correct for the default case, i.e., where
        # height = 4 and width = 12.

    def env_start(self):
        """The first method called when the episode starts, called before the
        agent starts.

        Returns:
            The first state from the environment.
        """
        reward = 0
        # agent_loc will hold the current location of the agent
        self.agent_loc = self.start_loc
        # state is the one dimensional state representation of the agent location.
        state = self.state(self.agent_loc)
        termination = False
        self.reward_state_term = (reward, state, termination)

    return self.reward_state_term[1]

    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state,
                and boolean indicating if it's terminal.
        """

        if action == 0: # UP (Task 1)
            possible_next_loc = (self.agent_loc[0] - 1, self.agent_loc[1])
            if possible_next_loc[0] >= 0:
                self.agent_loc = possible_next_loc
            else:
                pass
        elif action == 1: # LEFT
            possible_next_loc = (self.agent_loc[0], self.agent_loc[1] - 1)
            if possible_next_loc[1] >= 0: # Within Bounds?
                self.agent_loc = possible_next_loc
            else:
                pass # Stay.
        elif action == 2: # DOWN
            possible_next_loc = (self.agent_loc[0] + 1, self.agent_loc[1])
            if possible_next_loc[0] < self.grid_h: # Within Bounds?
                self.agent_loc = possible_next_loc
            else:
                pass # Stay.
        elif action == 3: # RIGHT
            possible_next_loc = (self.agent_loc[0], self.agent_loc[1] + 1)
            if possible_next_loc[1] < self.grid_w: # Within Bounds?
                self.agent_loc = possible_next_loc
            else:
                pass # Stay.
        else:
            raise Exception(str(action) + " not in recognized actions [0: Up, 1: Left, 2: Down, 3: Right]!")

        reward = -1
        terminal = False

        if self.agent_loc == self.goal_loc: # Reached Goal!
            terminal = True
        elif self.agent_loc in self.cliff: # Fell into the cliff!
            reward = -100
            self.agent_loc = self.start_loc
        else:
            pass

        self.reward_state_term = (reward, self.state(self.agent_loc), terminal)
        return self.reward_state_term

    def env_end(self, reward):
        raise NotImplementedError

    def env_cleanup(self):
        """Cleanup done after the environment ends"""
        self.agent_loc = self.start_loc

    # helper method
    def state(self, loc):
        x, y = loc
        return x * self.grid_w + y


class TDAgent(BaseAgent):
    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts."""

        # Create a random number generator with the provided seed to seed the agent for reproducibility.
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))

        # Policy will be given, recall that the goal is to accurately estimate its corresponding value function.
        self.policy = agent_info.get("policy")
        # Discount factor (gamma) to use in the updates.
        self.discount = agent_info.get("discount")
        # The learning rate or step size parameter (alpha) to use in updates.
        self.step_size = agent_info.get("step_size")

        # Initialize an array of zeros that will hold the values.
        # Recall that the policy can be represented as a (# States, # Actions) array. With the
        # assumption that this is the case, we can use the first dimension of the policy to
        # initialize the array for values.
        self.values = np.zeros((self.policy.shape[0],))

    def agent_start(self, state):
        """The first method called when the episode starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state from the environment's env_start function.
        Returns:
            The first action the agent takes.
        """
        # The policy can be represented as a (# States, # Actions) array. So, we can use
        # the second dimension here when choosing an action.
        action = self.rand_generator.choice(range(self.policy.shape[1]), p=self.policy[state])
        self.last_state = state
        return action

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the
                environment's step after the last action, i.e., where the agent ended up after the
                last action
        Returns:
            The action the agent is taking.
        """
        target = reward + self.discount * self.values[state]
        self.values[self.last_state] = self.values[self.last_state]  + self.step_size * (target - self.values[self.last_state])

        # Having updated the value for the last state, we now act based on the current
        # state, and set the last state to be current one as we will next be making an
        # update with it when agent_step is called next once the action we return from this function
        # is executed in the environment.

        action = self.rand_generator.choice(range(self.policy.shape[1]), p=self.policy[state])
        self.last_state = state

        return action

    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the terminal state.
        """
        target = reward
        self.values[self.last_state] = self.values[self.last_state]  + self.step_size * (target - self.values[self.last_state])

    def agent_cleanup(self):
        """Cleanup done after the agent ends."""
        self.last_state = None

    def agent_message(self, message):
        """A function used to pass information from the agent to the experiment.
        Args:
            message: The message passed to the agent.
        Returns:
            The response (or answer) to the message.
        """
        if message == "get_values":
            return self.values
        else:
            raise Exception("TDAgent.agent_message(): Message not understood!")


def run_experiment(env_info,
                   agent_info,
                   num_episodes=5000,
                   experiment_name=None,
                   plot_freq=100,
                   true_values_file=None,
                   value_error_threshold=1e-8):
    env = CliffWalkEnvironment
    agent = TDAgent
    rl_glue = RLGlue(env, agent)

    rl_glue.rl_init(agent_info, env_info)

    manager = Manager(env_info,
                      agent_info,
                      true_values_file=true_values_file,
                      experiment_name=experiment_name)
    for episode in range(1, num_episodes + 1):
        rl_glue.rl_episode(0)  # no step limit
        if episode % plot_freq == 0:
            values = rl_glue.agent.agent_message("get_values")
            manager.visualize(values, episode)

    values = rl_glue.agent.agent_message("get_values")
    if true_values_file is not None:
        # Grading: The Manager will check that the values computed using your TD agent match
        # the true values (within some small allowance) across the states. In addition, it also
        # checks whether the root mean squared value error is close to 0.
        manager.run_tests(values, value_error_threshold)

    return values