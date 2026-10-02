"""Constants shared by all analyses."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(os.environ.get("GAZERE_RESULTS", REPO / "results"))        # generated tables, figures, fold files
CACHE_DIR = Path(os.environ.get("GAZERE_CACHE", REPO / ".cache"))          # extracted features / visits

SCREEN_W, SCREEN_H = 2560, 1440        # gazeRE screen resolution (px)
MIN_VISIT_SECONDS = 3                  # minimum longest-visit duration used in the paper
FOLD_SEED = 256                        # random_state of every StratifiedGroupKFold in the paper
N_FOLDS = 5

# The 17 features of Table 1 (f_total_time is also extracted but is not one of the 17).
F17 = ["f_fixn_n", "f_fixn_dur_sum", "f_fixn_dur_avg", "f_fixn_dur_sd",
       "f_scan_distance_h", "f_scan_distance_v", "f_scan_distance_euclid", "f_scan_hv_ratio",
       "f_avg_sacc_length", "f_scan_speed_h", "f_scan_speed_v", "f_scan_speed",
       "f_box_area", "f_box_area_per_time", "f_fixns_per_box_area", "f_hull_area_per_time", "f_fixns_per_hull_area"]
HULL2 = ["f_hull_area_per_time", "f_fixns_per_hull_area"]          # features 16 and 17 (convex hull)
FEATURE_LABELS = {
    "f_fixn_n": "1. Number of fixations", "f_fixn_dur_sum": "2. Sum of fixation durations",
    "f_fixn_dur_avg": "3. Mean of fixation durations", "f_fixn_dur_sd": "4. SD of fixation durations",
    "f_scan_distance_h": "5. Sum of horizontal saccade amplitudes", "f_scan_distance_v": "6. Sum of vertical saccade amplitudes",
    "f_scan_distance_euclid": "7. Sum of saccade amplitudes", "f_scan_hv_ratio": "8. Horizontal/vertical amplitude ratio",
    "f_avg_sacc_length": "9. Average saccade amplitude", "f_scan_speed_h": "10. Horizontal saccade velocity",
    "f_scan_speed_v": "11. Vertical saccade velocity", "f_scan_speed": "12. Saccade velocity",
    "f_box_area": "13. Area scanned by saccades", "f_box_area_per_time": "14. Scanned area per time",
    "f_fixns_per_box_area": "15. Fixations per scanned area", "f_hull_area_per_time": "16. Convex hull area per time",
    "f_fixns_per_hull_area": "17. Fixations per convex hull area",
}

