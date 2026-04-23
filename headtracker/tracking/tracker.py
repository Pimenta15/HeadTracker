"""Hybrid Lucas-Kanade + PnP head tracker."""
import cv2
import numpy as np

from .utils import refine


class HeadTracker:
    """Tracks head pose across frames using optical flow with CNN-based fallback.

    Fast path: Lucas-Kanade optical flow with forward-backward validation.
    Fallback:  Full face detection + CNN landmark inference when tracking fails.
    """

    REPROJ_THRESH = 8.0
    FB_THRESH = 2.0
    MIN_TRACKED = 20
    LK_PARAMS = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
    )
    # Jaw (0-16) e boca (48-67) movem com a fala — excluídos do PnP de reinit
    # para evitar que abrir a boca mude o pitch estimado.
    STABLE_IDX = list(range(17, 48))  # sobrancelhas + nariz + olhos

    def __init__(self, face_detector, mark_detector, pose_estimator):
        self.face_detector = face_detector
        self.mark_detector = mark_detector
        self.pose_estimator = pose_estimator
        self._reset_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def r_vec(self):
        return self._r_vec

    @property
    def t_vec(self):
        return self._t_vec

    @property
    def track_pts_2d(self):
        return self._track_pts_2d

    def update(self, frame, gray):
        """Process one frame. Returns True when pose (r_vec, t_vec) is available.

        `just_reinitialized` is True on the frame where the CNN fallback fired.
        """
        self.just_reinitialized = False
        need_landmarks = not self.is_active

        if self.is_active and self._prev_gray is not None:
            if not self._track_lk(gray):
                need_landmarks = True

        if need_landmarks:
            if self._reinit(frame, gray):
                self.just_reinitialized = True
            else:
                self.is_active = False

        self._prev_gray = gray
        return self.is_active

    def reset(self):
        self._reset_state()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _reset_state(self):
        self._prev_gray = None
        self._track_pts_2d = None
        self._track_pts_3d = None
        self._r_vec = None
        self._t_vec = None
        self.frames_tracked = 0
        self.is_active = False
        self.just_reinitialized = False

    def _track_lk(self, gray):
        """Lucas-Kanade forward-backward tracking. Returns False when tracking fails."""
        new_pts, status_fw, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._track_pts_2d, None, **self.LK_PARAMS)
        back_pts, status_bw, _ = cv2.calcOpticalFlowPyrLK(
            gray, self._prev_gray, new_pts, None, **self.LK_PARAMS)

        fb_err = np.linalg.norm(
            (self._track_pts_2d - back_pts).reshape(-1, 2), axis=1)
        good = (
            (status_fw.ravel() == 1) &
            (status_bw.ravel() == 1) &
            (fb_err < self.FB_THRESH)
        )

        if good.sum() < self.MIN_TRACKED:
            return False

        obs_2d = new_pts[good].reshape(-1, 2).astype(np.float64)
        mod_3d = self._track_pts_3d[good].astype(np.float64)

        ok, r_new, t_new = cv2.solvePnP(
            mod_3d, obs_2d,
            self.pose_estimator.camera_matrix,
            self.pose_estimator.dist_coeefs,
            rvec=self._r_vec.copy(), tvec=self._t_vec.copy(),
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE)

        if not ok:
            return False

        proj, _ = cv2.projectPoints(
            mod_3d, r_new, t_new,
            self.pose_estimator.camera_matrix,
            self.pose_estimator.dist_coeefs)
        reproj_err = float(np.mean(
            np.linalg.norm(proj.reshape(-1, 2) - obs_2d, axis=1)))

        if reproj_err >= self.REPROJ_THRESH:
            return False

        self._r_vec = r_new
        self._t_vec = t_new
        self._reproject_all()
        self.frames_tracked += 1
        return True

    def _reinit(self, frame, gray):
        """Re-initialize tracking from CNN landmarks. Returns True on success."""
        h, w = frame.shape[:2]
        faces, _ = self.face_detector.detect(frame, 0.7)
        if len(faces) == 0:
            return False

        face = refine(faces, w, h, 0.15)[0]
        rx1, ry1, rx2, ry2 = face[:4].astype(int)
        rx1, ry1 = max(0, rx1), max(0, ry1)
        rx2, ry2 = min(w, rx2), min(h, ry2)
        patch = frame[ry1:ry2, rx1:rx2]
        if patch.size == 0:
            return False

        marks = self.mark_detector.detect([patch])[0].reshape([68, 2])
        marks *= (rx2 - rx1)
        marks[:, 0] += rx1
        marks[:, 1] += ry1

        s = self.STABLE_IDX
        ok, r_init, t_init = cv2.solvePnP(
            self.pose_estimator.model_points_68[s].astype(np.float64),
            marks[s].astype(np.float64),
            self.pose_estimator.camera_matrix,
            self.pose_estimator.dist_coeefs,
            flags=cv2.SOLVEPNP_EPNP)

        if not ok:
            return False

        self._r_vec = r_init
        self._t_vec = t_init
        self._reproject_all()
        self.frames_tracked = 0
        self.is_active = True
        return True

    def _reproject_all(self):
        proj, _ = cv2.projectPoints(
            self.pose_estimator.model_points_68,
            self._r_vec, self._t_vec,
            self.pose_estimator.camera_matrix,
            self.pose_estimator.dist_coeefs)
        self._track_pts_2d = proj.astype(np.float32)
        self._track_pts_3d = self.pose_estimator.model_points_68.copy()
