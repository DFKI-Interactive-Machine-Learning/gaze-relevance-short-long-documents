# Vendored from the gazeRE dataset repository (GPL-3.0):
#   https://github.com/DFKI-Interactive-Machine-Learning/gazeRE-dataset  (commit 9af27c6)
# Changes in this copy (all marked "PATCH"): package-relative imports; np.float -> float (removed in NumPy >= 1.24).
import datetime
import os

import numpy as np
import pandas as pd
import logging

from ..data_loading.visit_extractor import extract_paragraph_visits_vectorized  # PATCH: relative import
from . import extract_features  # PATCH: relative import
from .stimulus_config import get_paragraph_size  # PATCH: relative import

logger = logging.getLogger(__name__)
# PATCH: removed logging.basicConfig(level=logging.INFO) - a library must not configure the root logger


class GazeRelevanceDataset:

    def __init__(self):
        self._data_dict = {
            "user": [],
            "document": [],
            "corpus": [],
            "paragraph": [],
            "visit": [],
            "system_relevance": [],
            "perceived_relevance": [],
            "system_relevance_type": [],
            "method": []
        }
        self._df = None
        self._corpus = None
        self._method = None

    @property
    def dataframe(self) -> pd.DataFrame:
        self.finalize()
        return self._df

    def finalize(self):
        if not self.is_finalized:
            self._df = pd.DataFrame.from_dict(self._data_dict)
            self._data_dict = None  # free some memory space

    @property
    def is_finalized(self) -> bool:
        return self._df is not None

    def add_sample(self, user, corpus, document, paragraph,
                   system_relevance, perceived_relevance, system_relevance_type, visit, method, features):
        assert not self.is_finalized, "You cannot add samples to a finalized dataset."
        if self._corpus is None:
            self._corpus = corpus
        if self._corpus != corpus:
            self._corpus = "multi"
        if self._method is None:
            self._method = method.__class__.__name__
        if self._method != method.__class__.__name__:
            self._method = "multi"

        self._data_dict["user"].append(user)
        self._data_dict["corpus"].append(corpus)
        self._data_dict["method"].append(method.__class__.__name__)
        self._data_dict["document"].append(document)
        self._data_dict["paragraph"].append(paragraph)
        self._data_dict["visit"].append(visit)
        self._data_dict["system_relevance"].append(system_relevance)
        self._data_dict["perceived_relevance"].append(perceived_relevance)
        self._data_dict["system_relevance_type"].append(system_relevance_type)
        # append features, one feature per column
        for k, v in features.items():
            f_key = f"f_{k}"
            if f_key not in self._data_dict:
                self._data_dict[f_key] = []
            self._data_dict[f_key].append(v)

    def save(self, target_dir):
        self.finalize()

        filename = "{}_{}_{}.csv".format(self._corpus, self._method, datetime.datetime.now().strftime("%Y%m%d-%H%M"))
        filepath = os.path.join(target_dir, filename)

        self.dataframe.to_csv(path_or_buf=filepath, sep=";", index=False)


class FeatureExtractor(object):

    def __init__(self, min_visit_duration=0, min_fixations=3, screen_width=1, screen_height=1):
        self._w, self._h = screen_width, screen_height
        self._min_visit_duration = min_visit_duration
        self._min_fixations = min_fixations

    def _filter_visits(self, visits):
        v1 = [v for v in visits if v.duration >= self._min_visit_duration]
        if len(v1) < len(visits):
            logger.debug(f"skipped {len(visits) - len(v1)} visits in paragraph {visits[0].paragraph_id} -- duration")
        v2 = [v for v in v1 if v.num_fixations >= self._min_fixations]
        if len(v2) < len(v1):
            logger.debug(f"skipped {len(v1) - len(v2)} visits in paragraph {visits[0].paragraph_id} -- num_fixations")
        return v2

    def extract(self, visits):
        raise NotImplementedError()


class VisitBasedMeanFeatureExtractor(FeatureExtractor):

    def extract(self, visits):
        """ Compute a feature vector for each visit and return their mean as a sample."""
        visits_ = self._filter_visits(visits)
        if len(visits_) == 0:
            return []

        features = []
        for v in visits_:
            # Paragraph size is used for feature scaling to better generalize across paragraphs of different lengths.
            w, h = get_paragraph_size(v.document, v.paragraph_id)
            features.append(extract_features(df=v.data, scaling_x=w, scaling_y=h))
            # prior call was: scaling_x=self._w, scaling_y=self._h

        df = pd.DataFrame(data=features)
        if df.shape[1] != 18:
            raise AssertionError(f"incomplete features: {df.shape[1]}/18 (num_visits: {len(visits)})")

        return [{
            "visit": np.nan,
            "features": df.mean(axis=0)  # ignores nan cells for mean statistic
        }]


class FirstVisitFeatureExtractor(VisitBasedMeanFeatureExtractor):

    def extract(self, visits):
        """ Compute and return the feature vector of the **first** visit. """
        return super(FirstVisitFeatureExtractor, self).extract([visits[0]])


class LastVisitFeatureExtractor(VisitBasedMeanFeatureExtractor):

    def extract(self, visits):
        """ Compute and return the feature vector of the **last** visit. """
        return super(LastVisitFeatureExtractor, self).extract([visits[-1]])


class LongestVisitFeatureExtractor(VisitBasedMeanFeatureExtractor):

    def extract(self, visits):
        """ Compute and return the feature vector of the **longest** visit (recommended). """
        longest_visit = visits[np.argmax([v.duration for v in visits])]
        return super(LongestVisitFeatureExtractor, self).extract([longest_visit])


def extract_training_data(study_data: dict, feature_extractor: FeatureExtractor, target_dir=None):
    if target_dir is not None:
        assert os.path.exists(target_dir) and os.path.isdir(target_dir), \
            f"invalid path for storing training data: {target_dir}"

    dataset = GazeRelevanceDataset()

    for user, user_data in study_data.items():
        for document, document_data in user_data.items():
            logger.debug(f"processing {user}-{document}")
            corpus = document_data["corpus"]
            num_paragraphs = document_data["num_paragraphs"]
            gaze = document_data["dataframe"]
            visits = extract_paragraph_visits_vectorized(gaze, document, min_visit_duration=.5, max_gap_duration=.2)

            # populate training data (ignore None (-2) and headline (-1))
            for p in range(num_paragraphs):
                system_relevance = document_data["system_relevance"][p]
                perceived_relevance = document_data["perceived_relevance"][p]
                # keep the g-REL based encoding of system-relevance
                if corpus == "g-rel":
                    system_relevance_type = document_data["g-rel_relevance"]
                else:
                    system_relevance_type = "r" if system_relevance else "i"
                # filter visits to the current paragraph where, at least, one sample belongs to a fixation
                p_visits = [v for v in visits if v.paragraph_id == p and v.has_fixation]
                if len(p_visits) == 0:
                    logger.debug(f"skip paragraph {document}-{p} not visited (participant {user})")
                    continue

                samples = feature_extractor.extract(p_visits)
                for s in samples:
                    dataset.add_sample(user=user, corpus=corpus, document=document, paragraph=p,
                                       system_relevance=system_relevance, perceived_relevance=perceived_relevance,
                                       system_relevance_type=system_relevance_type,
                                       visit=s["visit"], method=feature_extractor, features=s["features"])

    if target_dir is not None:
        dataset.save(target_dir=target_dir)

    return dataset
