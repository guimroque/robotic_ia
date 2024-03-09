from robodk.robolink import *  # API to communicate with RoboDK

RDK = Robolink()
robot = RDK.Item('', ITEM_TYPE_ROBOT)
robot.setConnectionParams('192.168.0.240', 5653, '', '', '')

print(robot.Name())

success = robot.Connect()
status, status_msg = robot.ConnectedState()
if status != ROBOTCOM_READY:
    raise Exception("Falha ao conectar: " + status_msg)
print("Conectado com sucesso: " + status_msg)

robot.MoveJ([10, 20, 30, 40, 50, 60])


RDK.setRunMode(RUNMODE_RUN_ROBOT)
