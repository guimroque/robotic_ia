import unittest
from classes.frame.frame import Frame

class TestFrame(unittest.TestCase):

    def test_get_distance(self):
        IMAGE = '/Users/guimroque/tcc/mocks/test_image/20240216_19_04_16_Pro.jpg'
        IMAGE_RESOLUTION = (1920, 1080)
        # Cria frames usando os métodos corretamente e desenha-os
        RANDOM_FRAMES = []
        for i in range(3):
            random_frame = Frame.random(IMAGE_RESOLUTION, f"random_frame_{i}")
            RANDOM_FRAMES.append(random_frame)

        FULL_FRAME = Frame.full(IMAGE_RESOLUTION, "FULL_FRAME")
        
        Frame.draw_frames(RANDOM_FRAMES, FULL_FRAME, True, IMAGE)  # Assume que você modificou draw_frames para não precisar de img_size

    

if __name__ == '__main__':
    unittest.main()