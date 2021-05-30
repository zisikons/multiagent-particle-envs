import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import ipdb

class Scenario(BaseScenario):
    def __init__(self):
        self.safe_landmarks = True # Pick the landmarks in a way
                                   # that the robots don't collide

        self.safety_margin  = 0.2  # extra distance to be kept from the agents
        self.target_tol     = 0.02 # the tolerance of achieving the goal

        # Fix numpy seed for reproducibility
        np.random.seed(0)

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 3
        world.collaborative = False # check here for shared reward

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
            agent.max_speed = 0.2   # temp
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world, safe=False):

        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.0, 1, 0.0])

        # set random initial states without collisions
        has_collision = True
        while has_collision:
            for agent in world.agents:
                agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c     = np.zeros(world.dim_c)

            # Check if initial position violates constraints
            has_collision = False
            for i in range(len(world.agents)):
                for j in range(i + 1, len(world.agents), 1):
                    if self.is_safely_initialized(world.agents[i],world.agents[j]):
                        has_collision = True


        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

        # Landmark Initiallization
        if self.safe_landmarks:
            has_collision = True
            while has_collision:
                for i, landmark in enumerate(world.landmarks):
                    landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
                    landmark.state.p_vel = np.zeros(world.dim_p)

                # Check if initial position violates constraints
                has_collision = False
                for i in range(len(world.landmarks)):
                    for j in range(i + 1, len(world.landmarks), 1):
                        if self.is_safely_initialized(world.landmarks[i],world.landmarks[j]):
                            has_collision = True
        else:
            for i, landmark in enumerate(world.landmarks):
                landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def is_safely_initialized(self, agent1, agent2):
        """
            Check if the agents are colliding when the environment is initiallized
        """

        tol = 0.01

        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size + self.safety_margin + tol

        return True if dist < dist_min else False

    def done(self, agent, world):

        # Get Agent index
        idx = int(agent.name.split(' ')[1])

        # get the agent and its target positions
        target_pos = world.landmarks[idx].state.p_pos
        agent_pos  = agent.state.p_pos

        # compute l2 distance
        dist   = np.linalg.norm(target_pos - agent_pos)

        return True if dist<self.target_tol else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, 
        # penalized for collisions
        rew = 0

        # Get Agent index
        idx = int(agent.name.split(' ')[1])

        # Agent's Landmark
        target_pos = world.landmarks[idx].state.p_pos
        agent_pos  = agent.state.p_pos

        dist = np.linalg.norm(target_pos - agent_pos, 1)
        rew  = -dist

        # Promote exact landing to target 
        #if (dist < self.target_tol): # may induce instability so check again
        #    rew += 5

        # Add a small penalty if agents collide
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 0.2
        return rew

    def observation(self, agent, world):
        '''
        x = [my_pos, my_vel, other_agents_pos, my_landmark]
        '''

        #x = np.concatenate([[agent.state.p_pos] + [agent.state.p_vel])
        idx = int(agent.name.split(' ')[1])

        # Landmark
        agent_landmark = world.landmarks[idx].state.p_pos - agent.state.p_pos

        # Other agents position
        other_agents = [a for a in world.agents if a is not agent]
        other_agents_pos = []

        for other in other_agents:
            other_agents_pos.append(other.state.p_pos - agent.state.p_pos)

        # Final state
        x = np.concatenate([agent.state.p_pos] + [agent.state.p_vel]
                          + other_agents_pos + [agent_landmark])

        return x

    def constraints(self, agent, world):
        # Constraint Type 1: Collisions with other robots
        other_agents = [a for a in world.agents if a is not agent]

        collision_signals = np.zeros(len(world.agents) - 1)
        for i, other in enumerate(other_agents):
            collision_signals[i] = np.linalg.norm(other.state.p_pos - agent.state.p_pos)

        return collision_signals

