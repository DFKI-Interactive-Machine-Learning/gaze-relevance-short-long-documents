# Vendored from the gazeRE dataset repository (GPL-3.0):
#   https://github.com/DFKI-Interactive-Machine-Learning/gazeRE-dataset  (commit 9af27c6)
# Changes in this copy (all marked "PATCH"): package-relative imports; np.float -> float (removed in NumPy >= 1.24).
import pandas as pd
import numpy as np
import logging
import math
from scipy.spatial import ConvexHull
from typing import List


class FixationEvent:

    def __init__(self, fixation_id, fixation_data):
        self._df = fixation_data
        self._id = fixation_id

    @property
    def duration(self):
        return self.end_time - self.start_time

    @property
    def start_time(self):
        return self._df["timestamp"].values[0]

    @property
    def end_time(self):
        return self._df["timestamp"].values[-1]

    @property
    def gaze_x(self):
        return self._df["gaze_x"].mean()

    @property
    def gaze_y(self):
        return self._df["gaze_y"].mean()

    @property
    def gaze_y_stimulus(self):
        return self._df["gaze_y_abs"].mean()

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> List:
        fixation_groups = df.groupby(by="fixation_id")
        return [FixationEvent(_id, _df) for _id, _df in fixation_groups if _id != np.nan]


class SaccadeEvent:

    def __init__(self, event_id: int, start_fixation: FixationEvent, end_fixation: FixationEvent):
        self._duration = end_fixation.start_time - start_fixation.end_time
        self._amplitude_h = np.fabs(end_fixation.gaze_x - start_fixation.gaze_x)
        self._amplitude_v = np.fabs(end_fixation.gaze_y_stimulus - start_fixation.gaze_y_stimulus)
        self._id = event_id

    @property
    def duration(self):
        return self._duration

    @property
    def amplitude_h(self):
        return self._amplitude_h

    @property
    def amplitude_v(self):
        return self._amplitude_v

    @property
    def velocity_h(self):
        return self.amplitude_h / self.duration

    @property
    def velocity_v(self):
        return self.amplitude_v / self.duration

    @staticmethod
    def from_fixations(fixations: List[FixationEvent]):
        saccades = []
        for i in range(len(fixations)-1):
            saccades.append(SaccadeEvent(event_id=i, start_fixation=fixations[i], end_fixation=fixations[i+1]))

        return saccades


def extract_fixation_features(fixations):
    if len(fixations) == 0:
        logging.warning("Cannot extract features from zero fixations. Returning np.nan")
        return {
            "fixn_n": np.nan,
            "fixn_dur_sum": np.nan,
            "fixn_dur_avg": np.nan,
            "fixn_dur_sd": np.nan
        }

    durations = np.array([f.duration for f in fixations], dtype=float)
    return {
        "fixn_n": len(fixations),
        "fixn_dur_sum": durations.sum(),
        "fixn_dur_avg": durations.mean(),
        "fixn_dur_sd": durations.std()
    }


def extract_saccade_features(saccades, time_elapsed=1, h=1, w=1):
    if len(saccades) == 0:
        return {}

    scan_distance_h = np.sum([s.amplitude_h for s in saccades]) / w
    scan_distance_v = np.sum([s.amplitude_v for s in saccades]) / h
    scan_distance_euclid = sum([math.sqrt((s.amplitude_h / w)**2 + (s.amplitude_v / h) ** 2) for s in saccades])
    scan_hv_ratio = scan_distance_h / scan_distance_v if scan_distance_v > 0 else scan_distance_h

    avg_saccade_length = scan_distance_euclid / len(saccades)

    scan_speed_h = scan_distance_h / time_elapsed
    scan_speed_v = scan_distance_v / time_elapsed
    scan_speed = scan_distance_euclid / time_elapsed

    return {
        "scan_distance_h": scan_distance_h,
        "scan_distance_v": scan_distance_v,
        "scan_distance_euclid": scan_distance_euclid,
        "scan_hv_ratio": scan_hv_ratio,
        "avg_sacc_length": avg_saccade_length,
        "scan_speed_h": scan_speed_h,
        "scan_speed_v": scan_speed_v,
        "scan_speed": scan_speed
    }


def extract_area_features(fixations, input_features, stimulus_area=1):
    total_time = input_features["total_time"]
    if "scan_distance_h" not in input_features or "scan_distance_v" not in input_features:
        logging.warning("no valid saccade features")
        return {}
    box_area = input_features["scan_distance_h"] * input_features["scan_distance_v"]

    # convex hull features
    hull_area = np.nan
    hull_area_per_time = np.nan
    fixns_per_hull_area = np.nan
    if len(fixations) >= 3:
        fixation_pts = np.array([[f.gaze_x, f.gaze_y_stimulus] for f in fixations])
        hull = ConvexHull(fixation_pts)
        hull_area = hull.volume / stimulus_area  # for 2D: area is the perimeter, volume is the area
        hull_area_per_time = hull_area / total_time
        fixns_per_hull_area = len(fixations) / hull_area
    else:
        logging.warning(f"not enough fixations ({len(fixations)}) for convex hull features")

    return {
        "box_area": box_area,
        "box_area_per_time": box_area / total_time,
        "fixns_per_box_area": box_area / len(fixations),
        # "hull_area": hull_area,
        "hull_area_per_time": hull_area_per_time,
        "fixns_per_hull_area": fixns_per_hull_area
    }


def extract_features(df, scaling_x=1, scaling_y=1):  # , additional_info=None
    """ Feature extraction. """
    fixations = FixationEvent.from_dataframe(df)
    saccades = SaccadeEvent.from_fixations(fixations)
    total_time = df["timestamp"].values[-1] - df["timestamp"].values[0]

    if len(saccades) == 0:
        logging.warning("no saccades for feature computation.")

    if len(fixations) == 0:
        logging.warning("no fixations for feature computation.")

    if len(saccades) - len(fixations) > 1:
        logging.warning(f"{len(fixations)} with {len(saccades)} saccades.")

    features = {
        "total_time": total_time
    }
    features.update(extract_fixation_features(fixations))
    features.update(
        extract_saccade_features(saccades=saccades, time_elapsed=total_time, w=scaling_x, h=scaling_y)
    )
    features.update(
        extract_area_features(fixations, features, stimulus_area=scaling_x * scaling_y)
    )

    return features
