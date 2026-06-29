# weld_utils.py
import numpy as np
import cv2


class GeometricQuantifier:
    """学术级几何特征量测引擎"""

    @staticmethod
    def compute_skeleton_length(polygon_pts):
        """骨架化曲线长度测量算法（支持弯曲裂纹）"""
        pts = np.array(polygon_pts, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        pad = 15
        mask = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.uint8)
        shifted_pts = pts - [x - pad, y - pad]
        cv2.fillPoly(mask, [shifted_pts], 255)

        skeleton = np.zeros_like(mask)
        try:
            from skimage.morphology import skeletonize
            binary = mask > 0
            skel_bool = skeletonize(binary)
            skeleton[skel_bool] = 255
        except Exception:
            # Fallback：OpenCV 细化逼近
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            temp = mask.copy()
            done = False
            while not done:
                eroded = cv2.erode(temp, element)
                temp_open = cv2.dilate(eroded, element)
                temp_sub = cv2.subtract(temp, temp_open)
                skeleton = cv2.bitwise_or(skeleton, temp_sub)
                temp = eroded.copy()
                if cv2.countNonZero(temp) == 0:
                    done = True

        skel_pixels = np.argwhere(skeleton == 255)
        if len(skel_pixels) == 0:
            return 0.0

        visited = set()
        total_len = 0.0
        for py, px in skel_pixels:
            visited.add((py, px))
            for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                ny, nx = py + dy, px + dx
                if (ny, nx) not in visited and skeleton[ny, nx] == 255:
                    weight = 1.414 if (dy != 0 and dx != 0) else 1.0
                    total_len += weight

        if total_len == 0.0 and len(skel_pixels) > 0:
            total_len = float(len(skel_pixels))
        return total_len


class WeldTrajectoryMerger:
    """时空与几何特征二次去重融合引擎"""

    @staticmethod
    def merge_trajectories(raw_tracks, max_gap_frames=60, max_spatial_distance=80.0):
        """二次去重去跳变算法"""
        tracks = sorted(raw_tracks, key=lambda x: x["first_seen_frame"])
        merged_tracks = []
        merged_ids = {}

        for current_tr in tracks:
            curr_id = current_tr["id"]
            curr_class = current_tr["class"]
            curr_start = current_tr["first_seen_frame"]
            curr_first_centroid = current_tr["centroid_history"][0]

            matched_prev_idx = -1
            for idx, merged_tr in enumerate(merged_tracks):
                if merged_tr["class"] != curr_class:
                    continue

                prev_end = merged_tr["last_seen_frame"]
                prev_last_centroid = merged_tr["centroid_history"][-1]

                frame_gap = curr_start - prev_end
                if 0 < frame_gap <= max_gap_frames:
                    dist = np.sqrt(
                        (curr_first_centroid[0] - prev_last_centroid[0]) ** 2 +
                        (curr_first_centroid[1] - prev_last_centroid[1]) ** 2
                    )
                    if dist <= max_spatial_distance:
                        matched_prev_idx = idx
                        break

            if matched_prev_idx != -1:
                target = merged_tracks[matched_prev_idx]
                target_id = target["id"]
                merged_ids[curr_id] = target_id

                target["max_area"] = max(target["max_area"], current_tr["max_area"])
                target["max_len"] = max(target["max_len"], current_tr["max_len"])
                target["centroid_history"].extend(current_tr["centroid_history"])
                target["last_seen_frame"] = current_tr["last_seen_frame"]
            else:
                current_tr["last_seen_frame"] = current_tr["first_seen_frame"] + len(current_tr["centroid_history"]) - 1
                merged_tracks.append(current_tr)
                merged_ids[curr_id] = curr_id

        return merged_tracks, merged_ids