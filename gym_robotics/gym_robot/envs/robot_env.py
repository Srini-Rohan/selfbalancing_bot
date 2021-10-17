import numpy as np
import os
import gym
from gym import error, spaces
from gym import utils
from gym import spaces, logger
from gym.utils import seeding
import pybullet as p
import pybullet_data
import math
import time
class RoBots(gym.Env):

  def __init__(self):
    # Connect to physics Client
    self.physicsClient = p.connect(p.GUI)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    # Observation Space
    self.observation_space=spaces.Box(np.array([-0.418,-np.finfo(np.float32).max],dtype=np.float32),np.array([0.418,np.finfo(np.float32).max],dtype=np.float32))
    # Action Space
    self.action_space=spaces.Discrete(7)
    # end episode if excedes theta stop
    self.theta_stop = 12 * 2 * math.pi / 360
    # Target Velocity
    self.tv=0

    self.steps_done = None
    ...
  # Get State
  def get_state(self):
    return p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.botId)[1])[0],p.getBaseVelocity(self.botId)[1][0]
  #One step through simulation
  def step(self, action):
    cubePos,cubeOrientation=p.getBasePositionAndOrientation(self.botId)
    vel,orien=p.getBaseVelocity(self.botId)
    theta,angular_vel=self.state
    self.state=self.get_state()
    # bool to end episode
    done = bool(
            theta < -self.theta_stop
            or theta > self.theta_stop
            or cubePos[1]>5
            or cubePos[1]<-5
        )

    dv = [ -0.1, -0.03, -0.01, 0, 0.01, 0.03, 0.1]
    #Increase velocity according to action
    velocity_dv = dv[action]
    self.tv += velocity_dv
    #set velocities of wheels
    p.setJointMotorControl2(bodyUniqueId=self.botId,
                                jointIndex=0,
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=self.tv)
    p.setJointMotorControl2(bodyUniqueId=self.botId,
                                jointIndex=1,
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=self.tv)
    p.resetDebugVisualizerCamera(cameraDistance=5,
                                    cameraYaw= 70,
                                    cameraPitch=-35,
                                    cameraTargetPosition =p.getBasePositionAndOrientation(self.botId)[0] )
    p.stepSimulation()
    #Reward
    if not done:
            self.reward = (1.0-abs(self.state[0]))*0.1-abs(self.tv)*0.01
    elif self.steps_done is None:
            self.steps_done = 0
            self.reward =(1.0-abs(self.state[0]))*0.1-abs(self.tv)*0.01
    else:
            if self.steps_done == 0:
                logger.warn("WARNING")
            self.steps_done += 1
            self.reward = 0.0
    if done:
        #Increase reward
        if cubePos[1]>5 or cubePos[1]<-5:
            self.reward+=100

    return np.array(self.state),self.reward,done,{}
    ...
  #Reset
  def reset(self):
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(0.01)
    planeId = p.loadURDF("plane.urdf")
    cubeStartPos = [0, 0, 0.001]
    cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
    self.botId = p.loadURDF("robot.urdf",cubeStartPos, cubeStartOrientation)
    self.state=[0,0]
    self.tv=0
    return np.array(self.state)
    ...

  def render(self, mode='human'):
    pass
