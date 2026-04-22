"""Head pose estimation via PnP."""
import cv2
import numpy as np


class PoseEstimator:

    def __init__(self, image_width, image_height, model_path):
        self.size = (image_height, image_width)
        self.model_points_68 = self._load_model_points(model_path)

        self.focal_length = self.size[1]
        self.camera_center = (self.size[1] / 2, self.size[0] / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.camera_center[0]],
             [0, self.focal_length, self.camera_center[1]],
             [0, 0, 1]], dtype="double")

        self.dist_coeefs = np.zeros((4, 1))
        self.r_vec = None
        self.t_vec = None

    def _load_model_points(self, path):
        raw = []
        with open(path) as f:
            for line in f:
                raw.append(line)
        pts = np.array(raw, dtype=np.float32)
        pts = np.reshape(pts, (3, -1)).T
        pts[:, 2] *= -1
        return pts

    def solve(self, points):
        _, r_vec, t_vec = cv2.solvePnP(
            self.model_points_68,
            points,
            self.camera_matrix,
            self.dist_coeefs,
            flags=cv2.SOLVEPNP_EPNP)
        self.r_vec = r_vec
        self.t_vec = t_vec
        return r_vec, t_vec

    def visualize(self, image, pose, color=(255, 255, 255), line_width=2):
        """Draw a 3D box indicating head orientation."""
        rotation_vector, translation_vector = pose
        point_3d = []
        rear_size, rear_depth = 75, 0
        front_size, front_depth = 100, 100
        for s, d in [(rear_size, rear_depth), (front_size, front_depth)]:
            point_3d += [(-s, -s, d), (-s, s, d), (s, s, d), (s, -s, d), (-s, -s, d)]
        point_3d = np.array(point_3d, dtype=np.float32).reshape(-1, 3)

        point_2d, _ = cv2.projectPoints(
            point_3d, rotation_vector, translation_vector,
            self.camera_matrix, self.dist_coeefs)
        point_2d = np.int32(point_2d.reshape(-1, 2))

        cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[1]), tuple(point_2d[6]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[2]), tuple(point_2d[7]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[3]), tuple(point_2d[8]), color, line_width, cv2.LINE_AA)
