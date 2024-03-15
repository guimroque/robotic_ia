import unittest
from classes.frame.frame import Frame


#todo: test all methods of library
class TestFrame(unittest.TestCase):
    def test_get_distance(self):
        IMAGE = '/Users/guimroque/tcc/mocks/test_image/20240216_19_04_16_Pro.jpg'
        IMAGE_RESOLUTION = (1920, 1080)
        RANDOM_FRAMES = []

        for i in range(3):
            random_frame = Frame.random(IMAGE_RESOLUTION, f"random_frame_{i}")
            RANDOM_FRAMES.append(random_frame)

        FULL_FRAME = Frame.full(IMAGE_RESOLUTION, "FULL_FRAME")
        
        Frame.draw_frames(RANDOM_FRAMES, FULL_FRAME, True, IMAGE)

if __name__ == '__main__':
    unittest.main()