#%%
## IMPORTS

import numpy as np  # tensor maths
import networkx as nx  # operations on graphs
import gym  # RL environment
from abc import ABC, abstractmethod  # interface verification


#%%
## GRAPH MACHINERY


class RailwayGraph:
    """
    Graph of all the pixels which can be crossed at any time by Pacman.
    In order to be a node, a pixel needs to lie in a corridor. Each different pixel
    can be connected to any of the four Von Neumann neighbors (up, right, down, left), provided
    they are admissible. The NetworkX library is used.
    """

    def __init__(self):
        self.graph = None
        self.initialize_graph()
        self.directions = {
            "up": (-1, 0),
            "down": (1, 0),
            "right": (0, 1),
            "left": (0, -1),
        }

    def initialize_graph(self):
        # The graph is initialized and saved
        """
        Initializes the graph. It loop over every corridor pixel over the rails_map matrix and adds
        its corresponding node. For each new node, the presence of neighbors is checked, and they are
        eventually added. Finally, those pixel who are not
        """

        self.graph = nx.Graph()  # NetworkX-provided data structure to represent a graph
        rails_map = np.load("saved_objects/rails_matrix.npy").astype(int)
        corridors_color = 1
        m, n = np.shape(rails_map)

        # Loop over all pixels
        for row in range(m):
            for col in range(n):
                color = rails_map[row, col]
                if (color == corridors_color) and (
                    (row, col) not in self.graph.nodes
                ):  # if corridor
                    self.graph.add_node((row, col))  # add node to the graph

                    # Loop over the neighbors and establish edge if necessary
                    for offset in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
                        neighbor = (row + offset[0], col + offset[1])

                        if (
                            (0 <= neighbor[0] <= m - 1)
                            and (0 <= neighbor[1] <= n - 1)
                            and (rails_map[neighbor[0], neighbor[1]] == corridors_color)
                        ):  # if not out of bounds and colored appropriately
                            self.graph.add_edge((row, col), neighbor)

    def nearest_node_to_pixel(self, pixel_coords):
        """
        This function looks for the node which has the closest key to the given pixel (L1 distance)
        and returns a tuple with its coordinates.
        """
        nodes_arr = np.asarray(self.graph.nodes())
        closest_node = nodes_arr[
            np.argmin(np.linalg.norm(nodes_arr - pixel_coords, ord=1, axis=1))
        ]
        return tuple(closest_node)

    def get_distance(self, source, target):
        """
        Computes the shortest distance from source to target.
        Source is likely to be Pacman, while targets can be the ghosts for example.
        """
        return nx.shortest_path_length(self.graph, source, target)

    def get_distance_after_source_movement(self, source, movement, target):
        """
        Computes the shortest distance from the source after a given movement to target.
        Source is likely to be Pacman, while targets can be the ghosts for example.
        """
        return nx.shortest_path_length(
            self.graph, self.nextNode(source, movement), target
        )

    def nextNode(self, node, movement):
        """
        Returns the node of the graph reached performing a step in a given direction from a given node.
        """
        possibleMovements = self.getPossibleMovements(node)
        if movement in possibleMovements:
            return (
                node[0] + self.directions[movement][0],
                node[1] + self.directions[movement][1],
            )
        else:
            return node

    def getPossibleMovements(self, node):
        """
        Returns the list of possible movements (up, down, right, left) that can be done starting
        from a given node of the graph.
        """
        neighbours = self.getNeighbours(node)
        possibleMovements = []
        for movement in self.directions.keys():
            newNode = (
                node[0] + self.directions[movement][0],
                node[1] + self.directions[movement][1],
            )
            if newNode in neighbours:
                possibleMovements.append(movement)
        return possibleMovements

    def getNeighbours(self, node):
        """
        Returns the list of neighbors of a given node in the graph.
        """
        neighbours = []
        for neighbour in nx.all_neighbors(self.graph, node):
            neighbours.append(neighbour)
        return neighbours


#%%
# FEATURE EXTRACTOR (gym frame -> clever representation)


class Pacman_features_extractor:
    def __init__(self, initial_screen):
        self.positions = {
            "pacman": None,
            "ghosts": None,
            "foods": None,
            "special_food": None,
        }
        self.ghosts_scared = False
        self.epsilon = 3
        self.railwayGraph = RailwayGraph()

        self.initialize_foods()
        self.update(initial_screen)

    def update(self, screen):
        """
        Given a screen, it updates the position and state of all objects.
        """
        self.update_guys(screen)
        self.update_foods()

    # List of features |->

    def nearest_food_distance(self):
        """
        Returns the distance between the pacman beast and the nearest food.
        """
        return self.nearest_entity_distance_from_pacman("foods")

    def nearest_ghost_distance(self):
        """
        Returns the distance between the pacman beast and the nearest ghost.
        """
        return self.nearest_entity_distance_from_pacman("ghosts")

    def nearest_special_food_distance():
        """
        Returns the distance between the pacman beast and the nearest special food.
        """
        return self.nearest_entity_distance_from_pacman("special_food")

    def ghost_are_scared():
        """
        Return true if ghosts are scared, false otherwise.
        """
        return self.ghosts_scared

    # <-| List of features (end)

    # Classes for features initialization/update/passing

    def update_guys(self, screen):
        self.update_ghosts_scared(screen)
        raw_guys_positions = self.extract_raw_guys_positions(screen)
        self.positions["pacman"] = self.railwayGraph.nearest_node_to_pixel(
            raw_guys_positions["pacman"]
        )

        # case 1: ghost are visible (scared or not)
        if raw_guys_positions["ghosts"] != []:  # I hope this is the right condition
            # for pos in (raw_guys_positions['ghosts']):
            self.positions["ghosts"] = [
                self.railwayGraph.nearest_node_to_pixel(pos)
                for pos in raw_guys_positions["ghosts"]
            ]

        # case 2: ghost are not visible -> just do nothing, we keep the old positions

    def initialize_foods(self):
        """
        It set the initial position of every food based on a-priori knowledge.
        """

        foods_list = self.food_initial_raw_positions()
        foods_nodes = []
        for food_pos in foods_list:
            foods_nodes.append(self.railwayGraph.nearest_node_to_pixel(food_pos))
        self.positions["foods"] = foods_nodes

        sp_foods_list = self.sp_food_initial_raw_positions()
        sp_foods_nodes = []
        for sp_food_pos in sp_foods_list:
            sp_foods_nodes.append(self.railwayGraph.nearest_node_to_pixel(sp_food_pos))
        self.positions["special_food"] = sp_foods_nodes

    def update_foods(self):
        foods_distances = np.asarray(
            [
                np.linalg.norm(
                    np.asarray(food_pos) - np.asarray(self.positions["pacman"]), ord=1
                )
                for food_pos in self.positions["foods"]
            ]
        )

        if np.min(foods_distances) < self.epsilon:
            self.positions["foods"].pop(np.argmin(foods_distances))

        sp_foods_distances = np.asarray(
            [
                np.linalg.norm(
                    np.asarray(sp_food_pos) - np.asarray(self.positions["pacman"]),
                    ord=1,
                )
                for sp_food_pos in self.positions["special_food"]
            ]
        )

        if np.min(sp_foods_distances) < self.epsilon:
            self.positions["special_food"].pop(np.argmin(sp_foods_distances))

    def nearest_entity_distance_from_pacman(self, entity_name):
        """
        Returns the distance between the pacman beast and a given entity.
        """

        beast_pos = self.positions["pacman"]
        entity_positions = self.positions[entity_name]
        return min(
            [
                self.railwayGraph.get_distance(beast_pos, e_pos)
                for e_pos in entity_positions
            ]
        )

    def nearest_entity_distance_from_pacman_after_movement(self, movement, entity_name):
        """
        Returns the distance between the pacman beast and a given entity after a given movement of the beast.
        """

        beast_pos = self.positions["pacman"]
        entity_positions = self.positions[entity_name]
        return min(
            [
                self.railwayGraph.get_distance_after_source_movement(
                    beast_pos, movement, e_pos
                )
                for e_pos in entity_positions
            ]
        )

    def update_ghosts_scared(self, screen):
        """
        Update ghosts_scared variable according to the given screen.
        """

        self.ghosts_scared = False

    def extract_raw_guys_positions(self, screen):
        """
        Returns a dictionary with the raw positions of all 'guys', extracted from the given screen.
        """
        guys_pos = self.PacmanAndGhostsCoords(screen)
        return {"pacman": guys_pos[0], "ghosts": guys_pos[1]}

    def food_initial_raw_positions(self):
        """
        Returns a list with the raw positions of all initial foods known a-priori.
        """
        return list(np.load("saved_objects/food_coords.npy"))

    def sp_food_initial_raw_positions(self):
        """
        Returns a list with the raw positions of all initial foods known a-priori.
        """
        return list(np.load("saved_objects/special_food_coords.npy"))

    def center(self, SpecificMatrix):
        """
        Given a matrix with 1 where lies the object you want to detect and 0 elsewhere,
        the position of the center of the object is returned.
        """
        a = np.where(SpecificMatrix == 1)
        y = a[0]
        x = a[1]

        x_bar = (x.max() + x.min()) / 2
        y_bar = (y.max() + y.min()) / 2

        return (x_bar, y_bar)

    def find_location(self, screen, value):
        """
        Find the object corresponding to value within the matrix. If it is not present None is returned.
        """
        SpecificMatrix = (screen == value).astype(int)
        if SpecificMatrix.sum() == 0:
            return None
        else:
            return self.center(SpecificMatrix)

    def PacmanAndGhostsCoords(
        self,
        screen,
        PacmanValue=42,
        WallsFoodValue=74,
        GhostsValues=[70, 38, 184, 88],
        ghosts_scared=False,
    ):
        """
        Given the matrix of the screen, a list with the positions of all the relevant objects is returned.
        """
        pacman_coords = self.find_location(screen, PacmanValue)

        if ghosts_scared:
            pass
        else:
            ghosts_coords = []
            for ghost_value in GhostsValues:
                location = self.find_location(screen, ghost_value)
                if location != None:
                    ghosts_coords.append(self.find_location(screen, ghost_value))

        return [pacman_coords, ghosts_coords]


#%%
# INTERFACE VERIFICATION


class RL_Environment(ABC):
    @abstractmethod
    def getState(self):
        """
        Should return the current environment state as a dictionary of (feature name - feature value).
        """
        pass

    @abstractmethod
    def getActions(self):
        """
        Should return the list of all possible actions.
        """
        pass

    @abstractmethod
    def getReward(self):
        """
        Should return the last reward received.
        """
        pass

    @abstractmethod
    def perform_action(self, a):
        """
        The environment perform the action a and it's state changes.
        """
        pass

    @abstractmethod
    def restart(self):
        """
        Set the environment to the initial configuration.
        """
        pass

    @abstractmethod
    def game_is_over(self):
        """
        Returns true if the game is over.
        """
        pass

    @abstractmethod
    def getCumulativeReward(self):
        """
        Returns the actual cumulative reward.
        """
        pass

    @abstractmethod
    def psi(self, s, a):
        """
        Should return relevant features of the given state-action pair
        as a dictionary of (feature name - feature value).
        """
        pass


# Left here in case a migration of `psi` to `RL_system()` is needed or wanted, as naive migration is impossible.

# class RL_system(ABC):
#    @abstractmethod
#    def psi(self, s, a):
#        """
#        Should return relevant features of the given state-action pair
#        as a dictionary of (feature name - feature value).
#        """
#        pass


#%%
# RL ENVIRONMENT (pt. 1)


class pacman_RL_environment(RL_Environment):
    def __init__(self):
        self.env = gym.make("MsPacman-ram-v0")

        self.state = self.env.reset()  # env ram representation of the current state
        self.skip_intro()  # the firsts steps you can't do anything, so it's better to skip them
        self.current_reward = 0  # last reward received
        self.cumulative_reward = 0
        self.game_over = False

        # the features_extractor is here because it has (and need) a state
        self.features_extractor = Pacman_features_extractor(self.getCurrentScreen())

    def getState(self):
        """
        Returns the current environment state as a dictionary of (feature name - feature value).
        """

        # some examples of state features
        features = {}

        for entity in ["ghosts", "foods", "special_food"]:
            for movement in ["up", "down", "right", "left"]:
                features[
                    "nearest_" + entity + "_distance_after_going_" + movement
                ] = self.features_extractor.nearest_entity_distance_from_pacman_after_movement(
                    movement, entity
                )

        for entity in ["ghosts", "foods", "special_food"]:
            features[
                "nearest_" + entity + "_distance"
            ] = self.features_extractor.nearest_entity_distance_from_pacman(entity)

        # features["ghost_are_scared"] = features_extractor.ghost_are_scared()
        # features["actual_time_step"] =
        # features["last_scared_ghost_time_step"] =

        return features

    def getActions(self):
        """
        Returns the list of all possible actions as strings.
        """
        return list(self.actions_dict().keys())

    def psi(self, s, a):
        """
        Returns relevant features of the given state-action pair
        as a dictionary of (feature name - feature value).
        """

        # these are just examples taken from the paper
        features = {}

        # for entity in ['ghosts', 'foods', 'special_food']:
        #    features["distance_of_the_closest_" + entity] \
        #    = s["nearest_" + entity + "_distance_after_going_" + a]

        for entity in ["ghosts", "foods", "special_food"]:
            features["getting_closer_to" + entity] = (
                s["nearest_" + entity + "_distance"]
                - s["nearest_" + entity + "_distance_after_going_" + a]
            )

        # features["distance_of_the_closest_food"] = distance_of_the_next_closest_food(s,a)
        # features["distance_of_the_closest_ghost"] = distance_of_the_closest_ghost(s,a)
        # features["food_will_be_eaten"] = food_will_be_eaten(s,a)
        # features["ghost_collision_is_possible"] = ghost_collision_is_possible(s,a)

        return features

    def getReward(self):
        """
        Returns the last reward received.
        """
        return self.current_reward

    def perform_action(self, a):
        """
        The environment perform the action a (given as a string) and it's state changes.
        """

        encoded_action = self.actions_dict()[
            a
        ]  # translate the action from string to number
        self.state, self.current_reward, self.game_over, info = self.env.step(
            encoded_action
        )
        self.cumulative_reward += self.current_reward

        # then we have to update the features extractor,
        # since features extraction doesn't depend only on the current screen
        self.features_extractor.update(self.getCurrentScreen())

    def restart(self):
        """
        Set the environment to the initial configuration.
        """

        self.state = self.env.reset()
        self.features_extractor = Pacman_features_extractor(self.getCurrentScreen())

    def game_is_over(self):
        """
        Returns true if the game is over.
        """
        return self.game_over

    def actions_dict(self):
        """
        Returns a dictionary of (action name - action encoded).
        The encoding is needed to give the commands to the env.
        """

        actions_d = {"up": 1, "down": 4, "right": 2, "left": 3}
        return actions_d

    def getCurrentScreen(self):
        """
        Returns the current game screen.
        """
        return self.env.env.ale.getScreen().reshape(210, 160)

    def skip_intro(self):
        intro_duration = 90
        for i in range(intro_duration):
            self.env.step(1)

    def getCumulativeReward(self):
        """
        Returns the actual cumulative reward.
        """
        return self.cumulative_reward


#%%
# RL ENVIRONMENT (pt. 2)

# Left here in case a migration of `psi` to `RL_system()` is needed or wanted, as naive migration is impossible.
# In that case, mind to comment the other class declaration.

# class pacman_RL_system(RL_system):


class pacman_RL_system:
    def __init__(self, _environment):
        self.environment = _environment
        self.old_state = self.environment.getState()
        self.learning_vector = (
            self.initial_learning_vector()
        )  # the list of parameters to learn
        self.eps_greedy = 0  # probability to play a random action
        self.discount_factor = 0.9  # specifies how much long term reward is kept
        self.learning_rate = 0.5

    # Left here in case a migration of `psi` to `RL_system()` is needed or wanted, as naive migration is impossible.

    # def psi(self, s, a):
    #    """
    #    Returns relevant features of the given state-action pair
    #    as a dictionary of (feature name - feature value).
    #    """
    #
    #    ## these are just examples taken from the paper
    #    features = {}
    #
    #    ## for entity in ['ghosts', 'foods', 'special_food']:
    #    ##    features["distance_of_the_closest_" + entity] \
    #    ##    = s["nearest_" + entity + "_distance_after_going_" + a]
    #
    #    for entity in ["ghosts", "foods", "special_food"]:
    #        features["getting_closer_to" + entity] = (
    #            s["nearest_" + entity + "_distance"]
    #            - s["nearest_" + entity + "_distance_after_going_" + a]
    #        )
    #
    #    ## features["distance_of_the_closest_food"] = distance_of_the_next_closest_food(s,a)
    #    ## features["distance_of_the_closest_ghost"] = distance_of_the_closest_ghost(s,a)
    #    ## features["food_will_be_eaten"] = food_will_be_eaten(s,a)
    #    ## features["ghost_collision_is_possible"] = ghost_collision_is_possible(s,a)
    #
    #    return features

    def learn(self, iterations):
        """
        Performs the learning steps a specified number of times.
        """

        for i in range(iterations):
            if self.environment.game_is_over():
                break
            self.learning_step()

        print("cumulative reward:", self.environment.getCumulativeReward())

        return self.learning_vector

    def learning_step(self):
        """
        This is the Q-learning routine.
        """

        # s = self.environment.getState() # current state
        s = (
            self.old_state
        )  # calculated during the previous update_learning_vector(s,a,r)
        # print(s)
        a = self.policy(s)  # action to perform according to the policy
        print("action:", a)
        self.environment.perform_action(a)
        r = self.environment.getReward()  # gained reward
        print("reward:", r)
        print("psi:", self.action_state_features_vector(s, a))
        self.update_learning_vector(s, a, r)
        print("vector:", self.learning_vector)
        print("-------------------:")

    def policy(self, s):
        """
        Returns the action to perform in the state s according to a policy.
        """
        return self.best_Q_policy(s)

    def Q(self, s, a):
        """
        This is the Q function, it returns the expected future discounted reward
        for taking action a ∈ A in state s ∈ S.
        """
        return self.learning_vector.dot(self.action_state_features_vector(s, a))

    def update_learning_vector(self, s, a, r):
        """
        This function updates the learning vector.
            a is the last performed action,
            s is the previous state,
            r is the reward that has been generated by performing a in state s. """

        currentState = self.environment.getState()
        max_Q = max(
            [self.Q(currentState, action) for action in self.environment.getActions()]
        )
        difference = r + self.discount_factor * max_Q - self.Q(s, a)
        self.learning_vector += (
            self.learning_rate * difference * self.action_state_features_vector(s, a)
        )
        self.old_state = currentState

    def best_Q_policy(self, s):
        """
        For a given state It returns the action that maximize the Q_function, but it
        can also return a random action with probability = eps_greedy.
        """

        actions = self.environment.getActions()

        if self.random_boolean(self.eps_greedy):
            return np.random.choice(actions)

        i = np.argmax([self.Q(s, a) for a in actions])
        return actions[i]

    def random_boolean(self, probability_of_true):
        """
        It returns true with the given probability, false otherwise.
        """
        return np.random.random_sample() < probability_of_true

    def initial_learning_vector(self):
        """
        It returns the initial configuration of the learning vector.
        """

        return np.zeros(
            len(
                self.action_state_features_vector(
                    self.old_state, self.environment.getActions()[0]
                )
            )
        )

    def current_state_vector(self):
        """
        It returns the vector of numerical values representing the current state.
        It basically extract the values from the state dictionary of the environment.
        """
        return np.asarray(list(self.environment.getState().values()))

    def action_state_features_vector(self, s, a):
        """
        It returns a vector of numerical values representing relevant features of the state-action pair.
        It basically extract the values from the psi(s,a) dictionary of the environment.
        """

        return np.asarray(list(self.environment.psi(s, a).values()))

    def reset_environment(self):
        """
        It sets the environment to the initial configuration.
        """
        self.environment.restart()

    def reset_learning_vector(self):
        """
        It sets the environment to the initial predefined value.
        """
        self.learning_vector = self.initial_learning_vector()


#%%
# INITIALIZE AGENT

agent = pacman_RL_system(pacman_RL_environment())


#%%
# DO STUFF

iterations: int = 200

try:
    get_ipython().run_line_magic("time", "agent.learn(iterations)")
except NameError:
    agent.learn(iterations)
