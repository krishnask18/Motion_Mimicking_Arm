import pybullet as p
import time
import pybullet_data
import math
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")
ur5Id = p.loadURDF("H:\\IvLabs\\Motion Mimicking arm\\git-repo-cloned\\urdf\\ur5.urdf", useFixedBase = True)
joints = p.getNumJoints(ur5Id)
pos = [5, 0, 0]
ang = p.calculateInverseKinematics(ur5Id, 7, pos)
ang = list(ang)
ang.append(0)
ang.append(0)

p.setJointMotorControlArray(ur5Id, range(joints), p.POSITION_CONTROL, targetPositions=ang)
for i in range (10000):
    p.stepSimulation()
    time.sleep(1./24.)

p.getBasePositionAndOrientation()