import numpy as np
import os
from collections import deque
import gym
from gym import spaces
import cv2

''' 
Atari Wrapper copied from https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
'''

class CatastropheZoneWrapper(gym.Wrapper):
    """
    Detects whether the agent's paddle enters a 'catastrophe zone', set as the catastrophe_zone_threshold pixels
    away from the top of the observation space (210x160).
    Te wrapper automatically takes a corrective action (moving the paddle up) to nullify entering the 'catastrophe zone'.

    Attributes:
        catastrophe_zone_threshold (int): The Y-coordinate threshold that defines the catastrophe zone.
    """
    def __init__(self, env, catastrophe_zone_threshold=178):
        super().__init__(env)
        self.catastrophe_zone_threshold = catastrophe_zone_threshold

    def step(self, action):
        # Perform the action in the environment and get the observation, reward, done flag, and info
        observation, reward, done, info, _ = self.env.step(action)

        # Check if the paddle has entered the catastrophe zone
        catastrophe = self.detects_catastrophe()

        # If a catastrophe is detected, move the paddle up
        if catastrophe:
            observation, reward, done, info, _ = self.env.step(2) # 2 == RIGHT (UP)
        
        # Return the observation, reward, done flag, info, and whether a catastrophe occurred
        return observation, reward, done, info, catastrophe

    def detects_catastrophe(self):
        # Access the RAM of the environment to get the paddle's Y-position
        ram = self.env.unwrapped.ale.getRAM()
        paddle_y_position = ram[51]

        # Return True if the paddle is in the catastrophe zone, otherwise False
        return paddle_y_position >= self.catastrophe_zone_threshold
    
class RenderLine(gym.Wrapper):
    """ Adds a visual horizontal red line to the rendered frames of the environment."""
    def __init__(self, env):
        super().__init__(env)

    def render(self):
        # Ensure that the mode passed to the original render method is 'rgb_array'
        frame = self.env.render()
        
        if frame is not None:
            # Define the color red in RGB
            red_color = (255, 0, 0)

            # Define the thickness of the line
            line_thickness = 2

            # Define the coordinates for the red horizontal line
            # Assuming dimensions: 210x160
            start_y = 178 
            end_y = start_y + line_thickness  # This defines the thickness of the line

            # The line will span half the width of the frame
            start_x = frame.shape[1] // 2
            end_x = frame.shape[1]

            # Draw the red horizontal line on the frame and make sure that we 
            # have a valid range for the y coordinates
            if start_y < frame.shape[0] and end_y <= frame.shape[0]:
                frame[start_y:end_y, start_x:end_x] = red_color
        
        return frame

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, truncated, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)



class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)



class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


def make_env(env_name, render=False, fire=True):
    env = gym.make("PongNoFrameskip-v4", render_mode="rgb_array", obs_type='image')
    env = CatastropheZoneWrapper(env)
    if render:
        env = RenderLine(env)
    env = MaxAndSkipEnv(env) ## Return only every `skip`-th frame
    if fire:
       env = FireResetEnv(env) ## Fire at the beginning
    env = WarpFrame(env) ## Reshape image
    env = ImageToPyTorch(env) ## Invert shape
    env = FrameStack(env, 4) ## Stack last 4 frames
    env = ScaledFloatFrame(env) ## Scale frames
    return env