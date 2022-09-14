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
        # set any world properties first
        world.world_length = args.episode_length
        world.dim_c = 10
        world.collaborative = True  # whether agents share rewards
        # add agents
        world.num_agents = args.num_agents  # 2
        assert world.num_agents == 2, (
            "only 2 agents is supported, check the config.py.")
        world.agents = [Agent() for i in range(world.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            # agent.u_noise = 1e-1
            # agent.c_noise = 1e-1
        # add landmarks
        world.num_landmarks = args.num_landmarks  # 3
        world.landmarks = [Landmark() for i in range(world.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # assign goals to agents
        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None
        # want other agent to go to the goal landmark
        world.agents[0].goal_a = world.agents[1]
        world.agents[0].goal_b = np.random.choice(world.landmarks)
        world.agents[1].goal_a = world.agents[0]
        world.agents[1].goal_b = np.random.choice(world.landmarks)
        # random properties for agents
        world.assign_agent_colors()
        # random properties for landmarks
        world.landmarks[0].color = np.array([0.75, 0.25, 0.25])
        world.landmarks[1].color = np.array([0.25, 0.75, 0.25])
        world.landmarks[2].color = np.array([0.25, 0.25, 0.75])
        # special colors for goals
        world.agents[0].goal_a.color = world.agents[0].goal_b.color
        world.agents[1].goal_a.color = world.agents[1].goal_b.color
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = 0.8 * np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        if agent.goal_a is None or agent.goal_b is None:
            return 0.0
        dist2 = np.sum(
            np.square(agent.goal_a.state.p_pos - agent.goal_b.state.p_pos))
        return -dist2  # np.exp(-dist2)

    def old_observation(self, agent, world):
        # goal positions
        # goal_pos = [np.zeros(world.dim_p), np.zeros(world.dim_p)]
        # if agent.goal_a is not None:
        #     goal_pos[0] = agent.goal_a.state.p_pos - agent.state.p_pos
        # if agent.goal_b is not None:
        #     goal_pos[1] = agent.goal_b.state.p_pos - agent.state.p_pos
        # goal color
        goal_color = [np.zeros(world.dim_color), np.zeros(world.dim_color)]
        # if agent.goal_a is not None:
        #     goal_color[0] = agent.goal_a.color
        if agent.goal_b is not None:
            goal_color[1] = agent.goal_b.color

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
        return np.concatenate([agent.state.p_vel] + entity_pos + [goal_color[1]] + comm)

    def observation(self, agent:Agent, world:World) -> np.ndarray:
        # get positions of all entities in this agent's reference frame
        agent_vel, agent_pos = agent.state.p_vel, agent.state.p_pos
        goal_color = [np.zeros(world.dim_color), np.zeros(world.dim_color)]
        # if agent.goal_a is not None:
        #     goal_color[0] = agent.goal_a.color
        if agent.goal_b is not None:
            goal_color[1] = agent.goal_b.color
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
        if self.use_comm:
            return np.concatenate([agent_vel, other_pos, goal_color[1], 
                                    np.array(comm).flatten()])
        else:
            return np.concatenate([agent_vel, other_pos, goal_color[1]])