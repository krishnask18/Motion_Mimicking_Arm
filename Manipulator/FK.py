import pybullet as p
import time
import pybullet_data
import math
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")
ur5Id = p.loadURDF("H:\\IvLabs\\Motion Mimicking arm\\git-repo-cloned\\urdf\\ur5.urdf", useFixedBase = True)
joints = p.getNumJoints(ur5Id)
ls = [math.pi]*8
p.setJointMotorControlArray(ur5Id, range(joints), p.POSITION_CONTROL, targetPositions=ls)
for i in range (10000):
    p.stepSimulation()
    time.sleep(1./24.)