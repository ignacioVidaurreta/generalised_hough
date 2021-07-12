import cv2
import numpy as np

def generalized_hough_guil(args: dict):
    print(args["source_filename"])
    img = cv2.imread(args["source_filename"])
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    template = cv2.imread(args["ref_filename"])
    height, width = template.shape[:2]

    edges = cv2.Canny(template, 200, 250)
    ght = cv2.createGeneralizedHoughGuil()
    ght.setTemplate(edges)

    ght.setMinDist(args["min_dist"])
    ght.setMinAngle(args["min_angle"])
    ght.setMaxAngle(args["max_angle"])
    ght.setAngleStep(args["angle_step"])
    ght.setLevels(args["levels"])
    ght.setMinScale(args["min_scale"])
    ght.setMaxScale(args["max_scale"])
    ght.setScaleStep(args["scale_step"])
    ght.setAngleThresh(args["angle_thresh"])
    ght.setScaleThresh(args["scale_thresh"])
    ght.setPosThresh(args["pos"])
    ght.setAngleEpsilon(args["angle_epsilon"])
    ght.setXi(args["xi"])

    result = ght.detect(img_gray)
    if result is not None and result[0] is not None:
        positions = result[0][0]
        for position in positions:
            center_col = int(position[0])
            center_row = int(position[1])
            scale = position[2]
            angle = int(position[3])

            found_height = int(height * scale)
            found_width = int(width * scale)

            rectangle = ((center_col, center_row),
                        (found_width, found_height),
                        angle)

            box = cv2.boxPoints(rectangle)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

            for i in range(-2, 3):
                for j in range(-2, 3):
                    img[center_row + i, center_col + j] = 0, 0, 255

    cv2.imwrite("results-guil.png", img)
    cv2.imshow("Result Guil", img)
