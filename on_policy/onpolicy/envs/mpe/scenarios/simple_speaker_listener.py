import numpy as np
from onpolicy.envs.mpe.core import World, Agent, Landmark
from onpolicy.envs.mpe.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, args):
        self.obs_type = args.obs_type
        self.use_comm = args.use_comm
        self.max_edge_dist = args.max_edge_dist
        self.num_nbd_entities = args.num_nbd_entities
        world = World()
        world.world_length = args.episode_length
        # set any world properties first
        world.dim_c = 3
        world.num_landmarks = args.num_landmarks  # 3
        world.collaborative = True
        # add agents
        world.num_agents = args.num_agents  # 2
        assert world.num_agents == 2, (
            "only 2 agents is supported, check the config.py.")
        world.agents = [Agent() for i in range(world.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.size = 0.075
        # speaker
        world.agents[0].movable = False
        # listener
        world.agents[1].silent = True
        # add landmarks
        world.landmarks = [Landmark() for i in range(world.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.04
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # assign goals to agents
        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None
        # want listener to go to the goal landmark
        world.agents[0].goal_a = world.agents[1]
        world.agents[0].goal_b = np.random.choice(world.landmarks)
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.25])
        # random properties for landmarks
        world.landmarks[0].color = np.array([0.65, 0.15, 0.15])
        world.landmarks[1].color = np.array([0.15, 0.65, 0.15])
        world.landmarks[2].color = np.array([0.15, 0.15, 0.65])
        # special colors for goals
        world.agents[0].goal_a.color = world.agents[0].goal_b.color + \
            np.array([0.45, 0.45, 0.45])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        return reward(agent, reward)

    def reward(self, agent, world):
        # squared distance from listener to landmark
        a = world.agents[0]
        dist2 = np.sum(np.square(a.goal_a.state.p_pos - a.goal_b.state.p_pos))
        return -dist2

    def old_observation(self, agent, world):
        # goal color
        goal_color = np.zeros(world.dim_color)
        if agent.goal_b is not None:
            goal_color = agent.goal_b.color

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # communication of all other agents
        comm = []
        for other in world.agents:
            if other is agent or (other.state.c is None):
                continue
            comm.append(other.state.c)

        # speaker
        if not agent.movable:
            return np.concatenate([goal_color])
        # listener
        if agent.silent:
            return np.concatenate([agent.state.p_vel] + entity_pos + comm)

    def observation(self, agent:Agent, world:World) -> np.ndarray:
        # get positions of all entities in this agent's reference frame
        agent_vel, agent_pos = agent.state.p_vel, agent.state.p_pos
        goal_color = np.zeros(world.dim_color)
        # if agent.goal_a is not None:
        #     goal_color[0] = agent.goal_a.color
        if agent.goal_b is not None:
            goal_color = agent.goal_b.color
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
            entity_pos = []
            for entity in world.landmarks:
                entity_pos.append(np.array(entity.state.p_pos - agent_pos))
                dist_mag.append(np.linalg.norm(entity_pos[-1]))
            other_pos = np.array(other_agents_pos + entity_pos)   # remember + is just concatenation of lists

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

        # speaker
        if not agent.movable:
            return np.concatenate([goal_color])

        # listener
        if agent.silent:
            if self.use_comm:
                return np.concatenate([agent_vel, other_pos, comm])
            else:
                return np.concatenate([agent_vel, other_pos])