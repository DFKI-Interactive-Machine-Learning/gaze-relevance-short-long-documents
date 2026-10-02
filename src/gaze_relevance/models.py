"""Model definitions."""
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 47   # PyCaret session_id of the original analysis

# The twelve classical algorithms of the PyCaret pool that have a scikit-learn implementation, default hyperparameters
# (PyCaret's "SVM - Linear Kernel" is an SGDClassifier with hinge loss).
CLASSICAL_POOL = {
    "LR": lambda: LogisticRegression(max_iter=1000), "KNN": KNeighborsClassifier, "NB": GaussianNB,
    "DT": lambda: DecisionTreeClassifier(random_state=SEED), "SVM-linear": lambda: SGDClassifier(loss="hinge", random_state=SEED),
    "Ridge": lambda: RidgeClassifier(random_state=SEED), "RF": lambda: RandomForestClassifier(random_state=SEED, n_jobs=-1),
    "QDA": QuadraticDiscriminantAnalysis, "AdaBoost": lambda: AdaBoostClassifier(random_state=SEED),
    "GB": lambda: GradientBoostingClassifier(random_state=SEED), "LDA": LinearDiscriminantAnalysis,
    "ET": lambda: ExtraTreesClassifier(random_state=SEED, n_jobs=-1),
}

# Models of the transfer selection grid.
TRANSFER_MODELS = {
    "LDA": LinearDiscriminantAnalysis, "LR": lambda: LogisticRegression(max_iter=3000), "SVM": SVC,
    "RF": lambda: RandomForestClassifier(300, min_samples_leaf=5, random_state=0, n_jobs=-1), "NB": GaussianNB,
}


def minmax_pipeline(estimator):
    """Min-Max normalisation fitted on the training data, followed by the estimator (as in the paper)."""
    return make_pipeline(MinMaxScaler(), estimator() if callable(estimator) else clone(estimator))


def fit_predict_classical(name, X, y, X_test):
    """Fit one pool algorithm; an algorithm that cannot be fitted (e.g. QDA with a singular covariance) falls back to
    the majority class, as PyCaret does."""
    try:
        return minmax_pipeline(CLASSICAL_POOL[name]).fit(X, y).predict(X_test)
    except Exception:
        return DummyClassifier().fit(X, y).predict(X_test)


def within_googlenq_svm():
    """The within-GoogleNQ reference model: class-weighted RBF-SVM (on per-reader normalised features)."""
    return minmax_pipeline(lambda: SVC(class_weight="balanced"))
