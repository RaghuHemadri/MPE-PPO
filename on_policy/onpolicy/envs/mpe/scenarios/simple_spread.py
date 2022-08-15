import numpy as np
from onpolicy.envs.mpe.core import World, Agent, Landmark
from onpolicy.envs.mpe.scenario import BaseScenario


class Scenario(BaseScenario):
    """
            Parameters in args
            ––––––––––––––––––
            • num_agents: int
                Number of agents in the environment
                NOTE: this is equal to the number of goal positions
            • collaborative: bool
                If True then reward for all agents is sum(reward_i)
                If False then reward for each agent is what it gets individually
            • obs_type: str
                Choices: ['global', 'local', 'nbd']
                Whether we want include the [pos, vel] of just the neighbourhood
                instead of all entities in the environment for the observation
            ____________________________________________________________________
            If `obs_type== global` then the observation is the concatenation of
                [pos, vel] of all entities in the environment
            If `obs_type== local` then the observation is the [pos, vel] of the
                agent
            If `obs_type == ndb_obs` then each agent's observation is just the 
                [pos, vel] of  neighbouring `n` entities, provided that they are 
                within `max_edge_dist` of each other. If there are less than `n` 
                entities within `max_edge_dist` of each other, then pad with zeros.
            ____________________________________________________________________
            • max_edge_dist: float
                Maximum distance for an entity to be considered as neighbour
            • nbd_entities: int
                Number of neighbouring entities to consider in the environment
            • use_comm: bool
                Whether we want to use communication or not
        """
    def make_world(self, args):
        self.obs_type = args.obs_type
        self.use_comm = args.use_comm
        self.max_edge_dist = args.max_edge_dist
        self.num_nbd_entities = args.num_nbd_entities
        world = World()
        world.world_length = args.episode_length
        # set any world properties first
        world.dim_c = 2
        world.num_agents = args.num_agents
        world.num_landmarks = args.num_landmarks  # 3
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(world.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(world.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        world.assign_agent_colors()

        world.assign_landmark_colors()

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = 0.8 * np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                     for a in world.agents]
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

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                     for a in world.agents]
            rew -= min(dists)

        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def old_observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
    
    def observation(self, agent:Agent, world:World) -> np.ndarray:
        """
            agent: Agent object
            world: World object
            if obs_type=='local':
                return: agent_vel, agent_pos, agent_goal_pos
            if obs_type=='global':
                include information about all entities in the world:
                agent_vel, agent_pos, goal_pos, [other_agents_pos], [obstacles_pos]
            if obs_type=='nbd':
                return: agent_vel, agent_pos, agent_goal_pos, [other_entities_in_nbd_pos]
        """
        # get positions of all entities in this agent's reference frame
        agent_vel, agent_pos = agent.state.p_vel, agent.state.p_pos
        # communication of all other agents
        comm, other_agents_pos, obstacle_pos = [], [], []
        dist_mag = []
        if self.obs_type == 'local':
            other_pos = other_agents_pos + obstacle_pos # remember this is just concatenation of lists
        else:
            # get position of all other agents in this agent's reference frame
            for other in world.agents:
                if other is agent: continue
                comm.append(other.state.c)
                other_agents_pos.append(np.array(other.state.p_pos - agent_pos))
                dist_mag.append(np.linalg.norm(other_agents_pos[-1]))
            # get position of all obstacles in this agent's reference frame
            landmark_pos = []
            for landmark in world.landmarks:
                landmark_pos.append(np.array(landmark.state.p_pos - agent_pos))
                dist_mag.append(np.linalg.norm(landmark_pos[-1]))
            other_pos = np.array(other_agents_pos + landmark_pos)   # remember + is just concatenation of lists
            if self.obs_type == 'nbd':
                # sort other_pos according to distance from agent
                dist_mag_sort, dist_sort_idx = np.sort(dist_mag), np.argsort(dist_mag)
                other_pos = other_pos[dist_sort_idx, :]
                filter = np.array(dist_mag_sort)<self.max_edge_dist
                filter = np.expand_dims(filter, axis=1) # shape (num_entities, 1)
                filter = np.repeat(filter, axis=1, repeats=other_pos.shape[1])
                # all pos_entities outside max_edge_dist are set to 0
                other_pos = other_pos * filter
                # only take the closes num_nbd_entities
                other_pos = other_pos[:self.num_nbd_entities, :]
            other_pos = other_pos.flatten()
        if self.use_comm:
            return np.concatenate([agent_vel, agent_pos, other_pos, 
                                    np.array(comm).flatten()])
        else:
            return np.concatenate([agent_vel, agent_pos, other_pos])
