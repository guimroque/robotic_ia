from robodk import robolink

from CONSTANTES import LOGS


# [CONNECT FILE]
#
# - connect to RoboDk, if this window open, the connection is successful
# - retrieve all robots in the station by param ITEM_TYPE_ROBOT
# - print the name of each robot
#

RDK = robolink.Robolink()

robots = RDK.ItemList(robolink.ITEM_TYPE_ROBOT)

for robot in robots:
    print(f"{LOGS['CONNECT']} Robot: {robot.Name()}")