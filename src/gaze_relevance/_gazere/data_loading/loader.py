# Vendored from the gazeRE dataset repository (GPL-3.0):
#   https://github.com/DFKI-Interactive-Machine-Learning/gazeRE-dataset  (commit 9af27c6)
# Changes in this copy (all marked "PATCH"): package-relative imports; np.float -> float (removed in NumPy >= 1.24); loader state is per instance instead of per class.
import os
import logging
import pandas as pd


class gazeRE_DataLoader:
    # PATCH: these were class attributes in the original, so a second loader instance kept (and appended to) the
    # first instance's state; they are now initialised per instance in __init__.

    @property
    def num_participants(self):
        return len(self.grel.keys())

    @property
    def num_scanpaths_nq(self):
        return self.num_scanpaths("nq")

    @property
    def num_scanpaths_grel(self):
        return self.num_scanpaths("grel")

    def num_scanpaths(self, corpus="grel"):
        d = None
        if corpus == "grel":
            d = self.grel
        elif corpus == "nq":
            d = self.google_nq

        return sum([len(p.keys()) for k, p in d.items()])

    @property
    def num_paragraphs_nq(self):
        return self.num_paragraphs("nq")

    @property
    def num_paragraphs_grel(self):
        return self.num_paragraphs("grel")

    def num_paragraphs(self, corpus="grel"):
        d = None
        if corpus == "grel":
            d = self.grel
        elif corpus == "nq":
            d = self.google_nq
        return sum([sum([s["num_paragraphs"] for _, s in p.items()]) for _, p in d.items()])

    def __init__(self, data_dir: str, googleNQ=True, gREL=True, gaze_data=True,
                 include_users=None, exclude_users=None):
        """
        @param data_dir: The path to the data directory that contains the gazeRE data recordings.
        @param googleNQ: Specifies whether data for GoogleNQ stimuli is loaded
        @param gREL: Specifies whether data for gREL stimuli is loaded
        @param gaze_data: Specifies whether the gaze data is loaded (less memory load, if ratings are required only)
        @param include_users: A list of users for which the data shall be loaded.
        If set to None, all but the excluded users are loaded.
        @param exclude_users: A list of users for which no data shall be loaded
        """
        # PATCH: per-instance state (see class comment)
        self._EXCLUDE_USER_LIST, self._INCLUDE_USER_LIST, self._INCLUDE_DATA_SOURCE = [], [], []
        self.google_nq, self.grel = {}, {}
        # set configuration parameters
        self.data_dir = data_dir
        self.gaze_data = gaze_data
        if googleNQ:
            self._INCLUDE_DATA_SOURCE.append("GoogleNQ")
        if gREL:
            self._INCLUDE_DATA_SOURCE.append("g-rel")
        if include_users is not None:
            assert isinstance(include_users, list)
            self._INCLUDE_USER_LIST = include_users
        if exclude_users is not None:
            assert isinstance(exclude_users, list)
            self._EXCLUDE_USER_LIST = exclude_users

        # load data from the gazeRE study
        self._load_data()

    def _load_data(self):
        """
        initialize and load the study data
        """
        for user_data in os.scandir(self.data_dir):
            if user_data.is_file():
                logging.warning("Data directory contains an unknown file!")
            elif user_data.name not in self._EXCLUDE_USER_LIST and (
                    not self._INCLUDE_USER_LIST or user_data.name in self._INCLUDE_USER_LIST):
                logging.debug(user_data.name)
                self._load_user_data(user_data)

    def _load_user_data(self, user_data):
        if "g-rel" in self._INCLUDE_DATA_SOURCE:
            self._load_gREL(user_data)
        if "GoogleNQ" in self._INCLUDE_DATA_SOURCE:
            self._load_googleNQ(user_data)

    def _init_user_rating(self, path):
        with open(os.path.join(path, "User_Rating")) as rating_file:
            rating = []
            for line in rating_file:
                if line.strip():
                    rating.append([r == "True" for r in line.strip().split("|")])
            return rating

    def _load_googleNQ(self, user_data):
        """
            load GoogleNQ data
        """
        current_path = os.path.join(user_data.path, "GoogleNQ")
        self.google_nq[user_data.name] = {}
        answers = self._init_user_rating(current_path)
        for log_file in os.scandir(current_path):
            if log_file.is_file() and log_file.name.endswith(".csv"):
                try:
                    index, corpus, par_size, answ_par, name = \
                        log_file.name[:-4].replace(".", "_").split("_")
                except ValueError:
                    raise ValueError("Wrong filename format!")

                num_paragraphs = int(par_size[:-1])
                answer_paragraph_id = int(answ_par[1:])
                sys_relevance = [False] * num_paragraphs
                sys_relevance[answer_paragraph_id] = True

                obj = {
                    "index": int(index), "doc": "_".join([par_size, answ_par, name]), "corpus": corpus,
                    "file": log_file.name, "num_paragraphs": num_paragraphs,
                    "answer_paragraph_id": answer_paragraph_id, "perceived_relevance": answers[int(index)],
                    "system_relevance": sys_relevance
                }
                if self.gaze_data:
                    df = pd.read_csv(log_file.path, delimiter="|", encoding='utf-8', float_precision='round_trip',
                                     na_values="None")
                    obj["dataframe"] = df
                self.google_nq[user_data.name]["_".join([corpus, par_size, answ_par, name])] = obj

    def _load_gREL(self, user_data):
        """
           load g-rel data
        """
        current_path = os.path.join(user_data.path, "g-rel")
        self.grel[user_data.name] = {}
        answers = self._init_user_rating(current_path)
        for log_file in os.scandir(current_path):
            if log_file.is_file() and log_file.name.endswith(".csv"):
                try:
                    index, corpus, doc, sys_relevance = \
                        log_file.name[:-4].replace(".", "_").split("_")
                except ValueError:
                    raise ValueError("Wrong filename format!")
                obj = {
                    "index": int(index), "doc": doc, "corpus": corpus, "file": log_file.name,
                    "perceived_relevance": answers[int(index)], "num_paragraphs": 1,
                    "system_relevance": [sys_relevance == "r"], "g-rel_relevance": sys_relevance
                }
                if self.gaze_data:
                    df = pd.read_csv(log_file.path, delimiter="|", encoding='utf-8', float_precision='round_trip',
                                     na_values="None")
                    obj["dataframe"] = df
                    self.grel[user_data.name][("_".join([corpus, doc]))] = obj
