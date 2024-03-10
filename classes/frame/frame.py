from typing import Tuple, List
import math
import cv2
import numpy as np
import random
import os
from enum import Enum
import torch


class InfoType(Enum):
    PHYSICAL = 0
    VIRTUAL = 1

class Point(Enum):
    VERTICE_1 = 0  # up-left
    VERTICE_2 = 1  # up-right
    VERTICE_3 = 2  # down-right
    VERTICE_4 = 3  # down-left
    CENTRO = 4      # center

class Info(Enum):
    X = 0
    Y = 1
    Z = 2  # from high, to pixels format this value is 0 by default
    D_CENTER = 3  # distance from center of reference frame

COORDS: List[List[int]] = [ # POINTS X INFO
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
]

DIMENSIONS_NAMES = {
    'WIDTH': 0,
    'HEIGHT': 1,
    'TYPE': 2
}

DIMENSIONS: Tuple[float, float] = (0.0, 0.0)

PIXELS_ON_COORDS = 1  # TODO: change this to a dynamic value
PATH = './results'

#todo:
# - [ ] set reason to reference frame (is optional, use a default value of top length of the class [FRAME])
#       - [ ] make a function to calculate the reason of the reference frame, based on the distance of top vertices and real distance
#       - [ ] if reason is not set, calculate the reason of the reference frame
#       - [ ] create a global variable to store real distance
# - [ ] make a function to convert the pixels to real distance
# - [ ] make a function to convert the real distance to pixels


class Frame:
    def __init__(self, name: str, position: List[List[int]], coords: List[List[int]] = COORDS, reason: int = 1):
        self.name: str = name
        self.reason: int = reason
        self.coords: List[List[int]] = coords
        self.position: List[List[int]] = position
        self.dimensions: Tuple[float, float] = (0.0, 0.0)

    #
    # get_distance
    #
    # - @Description: Calculate the distance between two points using hipothenuse
    # 
    # - @Params: point1: Tuple[int, int], point2: Tuple[int, int]
    # - @Return: float
    #
    @staticmethod
    def get_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

    #
    # calculate_dimensions
    #
    # - @Description: Calculate the width and height of the frame based on the vertices coordinates
    # 
    # - @Params: coords: List[List[int]]
    # - @Return: Tuple[float, float]
    #
    @staticmethod
    def calculate_dimensions(coords: List[List[int]]) -> Tuple[float, float]:
        if len(coords) >= 4:
            s_direito = (coords[Point.VERTICE_2.value][Info.X], coords[Point.VERTICE_2.value][Info.Y])
            s_esquerdo = (coords[Point.VERTICE_1.value][Info.X], coords[Point.VERTICE_1.value][Info.Y])
            i_direito = (coords[Point.VERTICE_3.value][Info.X], coords[Point.VERTICE_3.value][Info.Y])
            w = Frame.get_distance(s_direito, s_esquerdo)
            h = Frame.get_distance(s_direito, i_direito)
            return w, h
        return 0.0, 0.0


    #
    # draw_frames
    #
    # - @Description: Draw a list of frames on the image and mark distances and vertices
    #                 - red point for vertices and center of the frame
    #                 - green lines for the frame
    #                 - yellow lines for the distance between the frames and the reference frame
    # 
    # - @Params: frames: List['Frame'], reference_frame: 'Frame', save: bool = True, background_img_path: str = None, filename: str = 'frames_image.png'
    # - @Return: None
    #
    @staticmethod
    def draw_frames(frames: List['Frame'], reference_frame: 'Frame', save: bool = True, background_img_path: str = None, filename: str = 'frames_image.png', ) -> None:
        if not frames:
            return

        if background_img_path:
            img = cv2.imread(background_img_path)
            if img is None:
                raise FileNotFoundError(f"A imagem de fundo em {background_img_path} não foi encontrada.")
            img_size = (img.shape[1], img.shape[0])
        else:
            max_x = int(Frame.calculate_dimensions(reference_frame.coords)[0])
            max_y = int(Frame.calculate_dimensions(reference_frame.coords)[1])

            img_size = (max_x, max_y)
            img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)

        center_of_reference = reference_frame.coords[4][:2]
        frames.append(reference_frame)
        for frame in frames:
            for i in range(4):
                # write the frame lines in green
                cv2.line(img, tuple(frame.coords[i][:2]), tuple(frame.coords[(i + 1) % 4][:2]), (0, 255, 0), 2)

            # write the vertices in red
            for point in frame.coords[:5]:
                cv2.circle(img, tuple(point[:2]), radius=5, color=(0, 0, 255), thickness=-1)
                label = f"({point[0]}, {point[1]})"
                cv2.putText(img, label, (point[0]+10, point[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # write the distances to the reference frame in yellow
        for frame in frames:
            center_current = frame.coords[4][:2]
            if frame != reference_frame:
                distance = Frame.get_distance(center_current, center_of_reference)
                cv2.line(img, center_current, center_of_reference, (0, 255, 255), 2)
                midpoint = ((center_current[0] + center_of_reference[0]) // 2, (center_current[1] + center_of_reference[1]) // 2)
                cv2.putText(img, f"{distance:.2f}", midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        if save:
            if not os.path.exists(PATH):
                os.makedirs(PATH)
            file_path = os.path.join(PATH, filename)
            cv2.imwrite(file_path, img)
            print(f"Frames salvos em: {file_path}")
        else:
            cv2.imshow('Frames', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    #
    # random
    #
    # - @Description: Generate a new random frame with 4 vertices forming a square and calculate the center
    # 
    # - @Params: img_size: Tuple[int, int], name: str = 'TESTE_NAME', reason: int = 1
    # - @Return: 'Frame'
    #
    @staticmethod
    def random(img_size: Tuple[int, int], name: str = 'TESTE_NAME', reason: int = 1) -> 'Frame':
        side_length = img_size[0] / 10
        x = random.randint(0, img_size[0] - int(side_length))
        y = random.randint(0, img_size[1] - int(side_length))

        # make the frame
        coords = [
            [x, y, 0, 0],  # Vértice 1
            [x + int(side_length), y, 0, 0],  # Vértice 2
            [x + int(side_length), y + int(side_length), 0, 0],  # Vértice 3
            [x, y + int(side_length), 0, 0]  # Vértice 4
        ]

        # calculate the center
        center_x = x + int(side_length / 2)
        center_y = y + int(side_length / 2)
        center = [center_x, center_y, 0, 0]
        coords.append(center)

        return Frame(name, position=coords, coords=coords, reason=reason)

    #
    # full
    #
    # - @Description: Generate a new frame with the full size of the image and calculate the center
    # 
    # - @Params: img_size: Tuple[int, int], name: str = 'FullSizeFrame', reason: int = 1
    # - @Return: 'Frame'
    #
    @staticmethod
    def full(img_size: Tuple[int, int], name: str = 'FullSizeFrame', reason: int = 1) -> 'Frame':
        coords = [
            [0, 0, 0, 0],  # up-left
            [img_size[0]-1, 0, 0, 0],  # up-right
            [img_size[0]-1, img_size[1]-1, 0, 0],  # down-right
            [0, img_size[1]-1, 0, 0],  # down-left
        ]
        center_x = img_size[0] // 2
        center_y = img_size[1] // 2
        center = [center_x, center_y, 0, 0]
        coords.append(center)

        return Frame(name, position=coords, coords=coords, reason=reason)


    #
    # yolov8_infer
    #
    # - @Description: Generate a new frame with output from YOLOv8 and calculate the center
    # 
    # - @Params: box: List[int]
    # - @Return: 'Frame'
    #
    @staticmethod
    def yolov8_infer(box: List[int]) -> 'Frame':
        x1, y1, x2, y2 = map(int, box[:4])
        # (x1, y1): up-left
        # (x2, y1): up-right
        # (x2, y2): down-right
        # (x1, y2): down-left
        coords = [
            [x1, y1, 0, 0],  
            [x2, y1, 0, 0],  
            [x2, y2, 0, 0],  
            [x1, y2, 0, 0]  
        ]
        footer_dist = Frame.get_distance((x2, y2), (x1, y2))/2
        right_dist = Frame.get_distance((x2, y2), (x2, y1))/2

        center_x = x1 + int(footer_dist)
        center_y = y2 - int(right_dist)
        center = [center_x, center_y, 0, 0]
        coords.append(center)
        return Frame(f"frame{x1}", position=coords, coords=coords, reason=1)