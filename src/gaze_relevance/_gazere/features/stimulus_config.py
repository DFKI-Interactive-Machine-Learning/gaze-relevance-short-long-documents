# Vendored from the gazeRE dataset repository (GPL-3.0):
#   https://github.com/DFKI-Interactive-Machine-Learning/gazeRE-dataset  (commit 9af27c6)
# Changes in this copy (all marked "PATCH"): package-relative imports; np.float -> float (removed in NumPy >= 1.24).
from typing import Tuple

googleNQ_coordinates = {'nq_5p_a0_LTIz': {4: {'y_top': 759, 'y_bottom': 70}, 3: {'y_top': 1309, 'y_bottom': 779},
                                          2: {'y_top': 1912, 'y_bottom': 1329}, 1: {'y_top': 2303, 'y_bottom': 1932},
                                          0: {'y_top': 2853, 'y_bottom': 2323}, -1: {'y_top': 3073, 'y_bottom': 2873},
                                          -2: {'y_top': 3073, 'y_bottom': 0}},
                        'nq_7p_a2_LTYz': {6: {'y_top': 653, 'y_bottom': 70}, 5: {'y_top': 991, 'y_bottom': 673},
                                          4: {'y_top': 1276, 'y_bottom': 1011}, 3: {'y_top': 1826, 'y_bottom': 1296},
                                          2: {'y_top': 2376, 'y_bottom': 1846}, 1: {'y_top': 2873, 'y_bottom': 2396},
                                          0: {'y_top': 3264, 'y_bottom': 2893}, -1: {'y_top': 3484, 'y_bottom': 3284},
                                          -2: {'y_top': 3484, 'y_bottom': 0}},
                        'nq_6p_a5_LTkw': {5: {'y_top': 282, 'y_bottom': 70}, 4: {'y_top': 1097, 'y_bottom': 302},
                                          3: {'y_top': 1435, 'y_bottom': 1117}, 2: {'y_top': 1826, 'y_bottom': 1455},
                                          1: {'y_top': 2376, 'y_bottom': 1846}, 0: {'y_top': 2767, 'y_bottom': 2396},
                                          -1: {'y_top': 2987, 'y_bottom': 2787}, -2: {'y_top': 2987, 'y_bottom': 0}},
                        'nq_7p_a1_Mzgy': {6: {'y_top': 335, 'y_bottom': 70}, 5: {'y_top': 991, 'y_bottom': 355},
                                          4: {'y_top': 1276, 'y_bottom': 1011}, 3: {'y_top': 1932, 'y_bottom': 1296},
                                          2: {'y_top': 2376, 'y_bottom': 1952}, 1: {'y_top': 2820, 'y_bottom': 2396},
                                          0: {'y_top': 3052, 'y_bottom': 2840}, -1: {'y_top': 3272, 'y_bottom': 3072},
                                          -2: {'y_top': 3272, 'y_bottom': 0}},
                        'nq_5p_a4_LTI3': {4: {'y_top': 494, 'y_bottom': 70}, 3: {'y_top': 938, 'y_bottom': 514},
                                          2: {'y_top': 1117, 'y_bottom': 958}, 1: {'y_top': 1508, 'y_bottom': 1137},
                                          0: {'y_top': 2270, 'y_bottom': 1528}, -1: {'y_top': 2490, 'y_bottom': 2290},
                                          -2: {'y_top': 2490, 'y_bottom': 0}},
                        'nq_6p_a1_LTEy': {5: {'y_top': 282, 'y_bottom': 70}, 4: {'y_top': 567, 'y_bottom': 302},
                                          3: {'y_top': 958, 'y_bottom': 587}, 2: {'y_top': 1614, 'y_bottom': 978},
                                          1: {'y_top': 2111, 'y_bottom': 1634}, 0: {'y_top': 2290, 'y_bottom': 2131},
                                          -1: {'y_top': 2510, 'y_bottom': 2310}, -2: {'y_top': 2510, 'y_bottom': 0}},
                        'nq_5p_a3_LTYx': {4: {'y_top': 494, 'y_bottom': 70}, 3: {'y_top': 1203, 'y_bottom': 514},
                                          2: {'y_top': 1806, 'y_bottom': 1223}, 1: {'y_top': 2250, 'y_bottom': 1826},
                                          0: {'y_top': 2747, 'y_bottom': 2270}, -1: {'y_top': 2967, 'y_bottom': 2767},
                                          -2: {'y_top': 2967, 'y_bottom': 0}},
                        'nq_5p_a2_MTgz': {4: {'y_top': 388, 'y_bottom': 70}, 3: {'y_top': 1097, 'y_bottom': 408},
                                          2: {'y_top': 1382, 'y_bottom': 1117}, 1: {'y_top': 2144, 'y_bottom': 1402},
                                          0: {'y_top': 2376, 'y_bottom': 2164}, -1: {'y_top': 2596, 'y_bottom': 2396},
                                          -2: {'y_top': 2596, 'y_bottom': 0}},
                        'nq_5p_a0_LTcw': {4: {'y_top': 600, 'y_bottom': 70}, 3: {'y_top': 1044, 'y_bottom': 620},
                                          2: {'y_top': 1700, 'y_bottom': 1064}, 1: {'y_top': 2303, 'y_bottom': 1720},
                                          0: {'y_top': 2747, 'y_bottom': 2323}, -1: {'y_top': 2967, 'y_bottom': 2767},
                                          -2: {'y_top': 2967, 'y_bottom': 0}},
                        'nq_6p_a3_MzA5': {5: {'y_top': 388, 'y_bottom': 70}, 4: {'y_top': 1097, 'y_bottom': 408},
                                          3: {'y_top': 1859, 'y_bottom': 1117}, 2: {'y_top': 2250, 'y_bottom': 1879},
                                          1: {'y_top': 2641, 'y_bottom': 2270}, 0: {'y_top': 2820, 'y_bottom': 2661},
                                          -1: {'y_top': 3040, 'y_bottom': 2840}, -2: {'y_top': 3040, 'y_bottom': 0}},
                        'nq_7p_a5_NTE0': {6: {'y_top': 547, 'y_bottom': 70}, 5: {'y_top': 1044, 'y_bottom': 567},
                                          4: {'y_top': 1435, 'y_bottom': 1064}, 3: {'y_top': 2144, 'y_bottom': 1455},
                                          2: {'y_top': 2588, 'y_bottom': 2164}, 1: {'y_top': 2926, 'y_bottom': 2608},
                                          0: {'y_top': 3370, 'y_bottom': 2946}, -1: {'y_top': 3590, 'y_bottom': 3390},
                                          -2: {'y_top': 3590, 'y_bottom': 0}},
                        'nq_6p_a4_ODQz': {5: {'y_top': 388, 'y_bottom': 70}, 4: {'y_top': 938, 'y_bottom': 408},
                                          3: {'y_top': 1223, 'y_bottom': 958}, 2: {'y_top': 1720, 'y_bottom': 1243},
                                          1: {'y_top': 2217, 'y_bottom': 1740}, 0: {'y_top': 3032, 'y_bottom': 2237},
                                          -1: {'y_top': 3252, 'y_bottom': 3052}, -2: {'y_top': 3252, 'y_bottom': 0}}}

g_rel_coordinates = {'g-rel_q075-1': {0: {'y_top': 1440, 'y_bottom': 373}, -1: {'y_top': 1440, 'y_bottom': 1387},
                                      -2: {'y_top': 1440, 'y_bottom': 303}},
                     'g-rel_q128-1': {0: {'y_top': 1440, 'y_bottom': 247},
                                      -1: {'y_top': 1440, 'y_bottom': 1387}, -2: {'y_top': 1440, 'y_bottom': 177}},
                     'g-rel_q094-2': {0: {'y_top': 1440, 'y_bottom': 267}, -1: {'y_top': 1440, 'y_bottom': 1334},
                                      -2: {'y_top': 1440, 'y_bottom': 197}},
                     'g-rel_q134-3': {0: {'y_top': 1440, 'y_bottom': 214}, -1: {'y_top': 1440, 'y_bottom': 1387},
                                      -2: {'y_top': 1440, 'y_bottom': 144}},
                     'g-rel_q103-1': {0: {'y_top': 1440, 'y_bottom': 320}, -1: {'y_top': 1440, 'y_bottom': 1334},
                                      -2: {'y_top': 1440, 'y_bottom': 250}},
                     'g-rel_q097-2': {0: {'y_top': 1440, 'y_bottom': 202},
                                      -1: {'y_top': 1440, 'y_bottom': 1387}, -2: {'y_top': 1440, 'y_bottom': 132}},
                     'g-rel_q118-1': {0: {'y_top': 1440, 'y_bottom': 320}, -1: {'y_top': 1440, 'y_bottom': 1387},
                                      -2: {'y_top': 1440, 'y_bottom': 250}},
                     'g-rel_q076-1': {0: {'y_top': 1440, 'y_bottom': 141},
                                      -1: {'y_top': 1440, 'y_bottom': 1387}, -2: {'y_top': 1440, 'y_bottom': 71}},
                     'g-rel_q122-2': {0: {'y_top': 1440, 'y_bottom': 108}, -1: {'y_top': 1440, 'y_bottom': 1387},
                                      -2: {'y_top': 1440, 'y_bottom': 38}},
                     'g-rel_q116-1': {0: {'y_top': 1440, 'y_bottom': 280}, -1: {'y_top': 1440, 'y_bottom': 1387},
                                      -2: {'y_top': 1440, 'y_bottom': 210}},
                     'g-rel_q085-2': {0: {'y_top': 1440, 'y_bottom': 320}, -1: {'y_top': 1440, 'y_bottom': 1334},
                                      -2: {'y_top': 1440, 'y_bottom': 250}},
                     'g-rel_q088-1': {0: {'y_top': 1440, 'y_bottom': 247},
                                      -1: {'y_top': 1440, 'y_bottom': 1387}, -2: {'y_top': 1440, 'y_bottom': 177}}}


def googleNQ_stimulus_height(document: str) -> int:
    # For each document googleNQ_coordinates[document][-2]['y_bottom'] returns the stimulus size with a padding of 70px
    # after the last paragraph
    return googleNQ_coordinates[document][-2]['y_top'] - (googleNQ_coordinates[document][-2]['y_bottom'] + 70)


def gREL_stimulus_height(document: str) -> int:
    # For each document g_rel_coordinates[document][-2]['y_bottom'] returns the stimuli size with a padding of 70px
    # after the last paragraph
    return g_rel_coordinates[document][0]['y_top'] - g_rel_coordinates[document][0]['y_bottom']


def get_stimulus_size(document: str) -> Tuple[int, int]:
    if document.startswith("g-rel"):
        h = gREL_stimulus_height(document)
    elif document.startswith("nq"):
        h = googleNQ_stimulus_height(document)
    else:
        raise NotImplementedError(f"The document {document} is not supported.")

    w = 2560
    return w, h


def get_paragraph_size(document: str, paragraph_id: int) -> Tuple[int, int]:
    if document.startswith("g-rel"):
        h = gREL_stimulus_height(document)
    elif document.startswith("nq"):
        s = googleNQ_coordinates[document][paragraph_id]
        h = s['y_top'] - s['y_bottom']
    else:
        raise NotImplementedError(f"The document {document} is not supported.")

    # width is always constant (can be adjusted here is width changes)
    x1, x2 = 749, 1773
    w = x2 - x1

    return w, h
