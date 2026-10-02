# Vendored from the gazeRE dataset repository (GPL-3.0):
#   https://github.com/DFKI-Interactive-Machine-Learning/gazeRE-dataset  (commit 9af27c6)
# Changes in this copy (all marked "PATCH"): package-relative imports; np.float -> float (removed in NumPy >= 1.24).
import pandas as pd
from typing import List


class ParagraphVisit:

    def __init__(self, visit_id: int, df: pd.DataFrame, document: str):
        self._id = visit_id
        self._document = document
        self._df = df

    @staticmethod
    def from_visits(visits):
        df = pd.concat([v.data for v in visits])
        return ParagraphVisit(visit_id=visits[0].visit_id, df=df, document=visits[0].document)

    @property
    def visit_id(self):
        return self._id

    @property
    def data(self) -> pd.DataFrame:
        return self._df

    @property
    def start_time(self):
        return self._df["timestamp"].values[0]

    @property
    def end_time(self):
        return self._df["timestamp"].values[-1]

    @property
    def duration(self):
        return self.end_time - self.start_time

    @property
    def document(self):
        return self._document

    @property
    def paragraph_id(self) -> int:
        return int(self._df["paragraph_id"].values[0])

    @property
    def is_background_visit(self):
        return self.paragraph_id < 0

    @property
    def has_fixation(self):
        return self.data["fixation_id"].any()  # returns True, if any row has a fixation_id other than np.nan

    @property
    def num_fixations(self):
        return len(self.data["fixation_id"].unique())

    def __str__(self):
        return "visit to {} for {:.2f}s (id:{})".format(self.paragraph_id, self.duration, self._id)


def has_short_gap(visits, max_gap_duration):
    for i in range(1, len(visits) - 1):
        if visits[i].is_background_visit and visits[i].duration < max_gap_duration and \
                visits[i - 1].paragraph_id == visits[i + 1].paragraph_id and not visits[i - 1].is_background_visit:
            return i
    return False


def merge_visits(visits, i):
    v_merged = ParagraphVisit.from_visits(visits=visits[i - 1:i + 2])
    return visits[:i - 1] + [v_merged] + visits[i + 2:]


def drop_background_visits(visits):
    return [v for v in visits if not v.is_background_visit]


def drop_short_visits(visits, min_duration):
    return [v for v in visits if v.duration > min_duration]


def extract_paragraph_visits_vectorized(df: pd.DataFrame, document, min_visit_duration=.5, max_gap_duration=.2,
                                        ignore_background=True) -> List[ParagraphVisit]:
    """
    Extracts visits to paragraphs, i.e., consecutive sequences of gazes to one paragraph.
    Short gaps and short visits are ignored (in this order).

    @param df: gaze data from a single trial (user and document).
    @param min_visit_duration: minimum duration of a visit [s]; must be greater than max_gap_duration.
    @param max_gap_duration: maximum duration of gaps that shall be ignored [s].
    @param ignore_background: if True, no background visits are returned.
    @return: List of ParagraphVisit instances
    """
    assert min_visit_duration > max_gap_duration

    # Extract visits as groups of consecutive paragraph_id values: the visits series assigns rows to visit IDs.
    # diff: reveals difference to previous samples; ne (not equal) converts values to True, if there is a change and to
    # False, otherwise; cumsum is a filter that sums the current and the previous values (=increases by one, whenever
    # there is a change in the paragraph_id signal);
    visit_ids = df["paragraph_id"].diff().ne(0).cumsum()
    df.insert(loc=6, column="visit_id", value=visit_ids)
    visits = [ParagraphVisit(_id, _df, document) for _id, _df in df.groupby(by="visit_id")]

    # merge as long as short gaps are available
    short_gap = has_short_gap(visits, max_gap_duration)
    while short_gap > 0:
        visits = merge_visits(visits, short_gap)
        short_gap = has_short_gap(visits, max_gap_duration)

    if ignore_background:
        visits = drop_background_visits(visits)
    visits = drop_short_visits(visits, min_visit_duration)
    return visits

