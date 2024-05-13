# Essa classe insere os objetos no ROBODK

import random
from robodk import robolink, robomath
from CONSTANTES import REFERENCES, BLOCKS, BLOCKS_COLOR, BLOCKS_POSITION


class Objects:
    def __init__(self):
        self.rdk = robolink.Robolink()
        self.reference_frame = self.rdk.Item(REFERENCES['CENTER_TABLE'], robolink.ITEM_TYPE_FRAME)

    def insert(self, coords=[0, 0], color="BLACK"): #todo: verify why recover a name of the class and use correct color
        #params to new item
        name = BLOCKS[color] + str(random.randint(0, 9999))
        _color = BLOCKS_COLOR[color]
        position = robomath.PosePP(# items are generated just on table
            coords[0], # x_pos
            coords[1], # y_pos,
            BLOCKS_POSITION['TABLE']['z_pos'],
            BLOCKS_POSITION['TABLE']['x_rot'],
            BLOCKS_POSITION['TABLE']['y_rot'],
            BLOCKS_POSITION['TABLE']['z_rot']
        )

        #get exists item to copy
        item = self.rdk.Item(REFERENCES['BLOCK_BLACK'], robolink.ITEM_TYPE_OBJECT)
        if not item.Valid():
            raise Exception('Item not valid')
        item.Copy()

        self.rdk.Render(False)

        #create a new item
        new_item = self.reference_frame.Paste()
        new_item.setName(name)
        new_item.setPose(position)
        new_item.setVisible(True, False)
        new_item.Recolor(_color)

        item.setVisible(False, False)
        self.rdk.Render(True)