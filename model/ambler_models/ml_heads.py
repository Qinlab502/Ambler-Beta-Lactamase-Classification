"""
Traditional machine learning head module.
Contains the SVM and Random Forest classifiers.
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class SVMHead:
    """
    SVM classification head.
    Uses an RBF kernel and supports probability output.
    """
    def __init__(
        self,
        C: float = 1.0,
        kernel: str = 'rbf',
        gamma: str = 'scale',
        probability: bool = True,
        random_state: int = 42
    ):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.probability = probability
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self._build_model()

    def _build_model(self):
        """Build the SVM model (with feature standardization)."""
        self.scaler = StandardScaler()
        self.model = Pipeline([
            ('scaler', self.scaler),
            ('svm', SVC(
                C=self.C,
                kernel=self.kernel,
                gamma=self.gamma,
                probability=self.probability,
                random_state=self.random_state
            ))
        ])

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the SVM model.
        Args:
            X: [N, hidden_dim] feature matrix
            y: [N] labels
        """
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        Args:
            X: [N, hidden_dim] feature matrix
        Returns:
            [N] predicted labels
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        Args:
            X: [N, hidden_dim] feature matrix
        Returns:
            [N, num_labels] probability matrix
        """
        return self.model.predict_proba(X)

    def save(self, path: str):
        """Save the model."""
        import joblib
        joblib.dump(self.model, path)

    def load(self, path: str):
        """Load the model."""
        import joblib
        self.model = joblib.load(path)


class RFHead:
    """
    Random Forest classification head.
    """
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = 'sqrt',
        random_state: int = 42,
        n_jobs: int = -1
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = None
        self.scaler = None
        self._build_model()

    def _build_model(self):
        """Build the RF model."""
        self.scaler = StandardScaler()
        self.model = Pipeline([
            ('scaler', self.scaler),
            ('rf', RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            ))
        ])

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the RF model.
        Args:
            X: [N, hidden_dim] feature matrix
            y: [N] labels
        """
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        Args:
            X: [N, hidden_dim] feature matrix
        Returns:
            [N] predicted labels
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        Args:
            X: [N, hidden_dim] feature matrix
        Returns:
            [N, num_labels] probability matrix
        """
        return self.model.predict_proba(X)

    def feature_importance(self) -> np.ndarray:
        """Return feature importances."""
        return self.model.named_steps['rf'].feature_importances_

    def save(self, path: str):
        """Save the model."""
        import joblib
        joblib.dump(self.model, path)

    def load(self, path: str):
        """Load the model."""
        import joblib
        self.model = joblib.load(path)
