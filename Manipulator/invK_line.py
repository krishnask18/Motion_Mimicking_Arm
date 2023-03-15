import pybullet as p
import time
import pybullet_data
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")
p.setGravity(0,0,-9.81)
ur5Id = p.loadURDF("H:\\IvLabs\\Motion Mimicking arm\\git-repo-cloned\\urdf\\ur5.urdf", [0,0,0], useFixedBase = 1)
joints = p.getNumJoints(ur5Id)
x1, y1, z1 = 0.1, 0.1, 0.1;
x2, y2, z2 = 1, 1, 1;
x, y, z = x1, y1, z1
orientation = p.getQuaternionFromEuler([3.14, 0.0 , 3.14 ])
while(x2-x != 0):
    print(x, y, z)
    target = p.calculateInverseKinematics(ur5Id, (joints-2), [x, y, z], targetOrientation = orientation)
    p.setJointMotorControlArray(ur5Id, range(joints-2), p.POSITION_CONTROL, targetPositions= target)
    for i in range (10):
        p.stepSimulation()
        time.sleep(1./24.)
    x = x1 + 0.2*(x2-x1)
    y = y1 + 0.2*(y2-y1)
    z = z1 + 0.2*(z2-z1)