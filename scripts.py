import cv2
import numpy as np


def get_qtd_usuarios(imagem):
    usuarios = 0

    template = cv2.imread('./images/pattern_user.png')
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # tratativa para imagem 2d
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(imagem, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where(res >= threshold)

    rect_points = []
    for pt in zip(*loc[::-1]):
        rect = [int(pt[0]), int(pt[1]), int(w), int(h)]
        rect_points.append(rect)
    rect_points, _ = cv2.groupRectangles(rect_points, 1, 0.2)

    for rect in rect_points:
        cv2.rectangle(imagem, (rect[0], rect[1]),
                      (rect[0] + rect[2], rect[1] + rect[3]), (255, 255, 204), 3)
        usuarios += 1

    return usuarios, imagem


