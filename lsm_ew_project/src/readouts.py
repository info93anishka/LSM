"""
src/readouts.py
=================
Implements the MULTI-TASK READOUT LAYER -- the only TRAINED part of the
entire LSM pipeline.

RECALL THE LSM DESIGN PHILOSOPHY:
The reservoir (src/reservoir.py) is a FIXED random network -- we never
update its weights. All of the "learning" in this project happens here,
in the readout layer, which takes the reservoir's state vector (a single
N-dimensional vector summarising how the reservoir responded to an input
signal) and maps it to a class label using a SIMPLE, CHEAP, TRAINABLE
classifier.

WHY THREE SEPARATE READOUT HEADS?
This project performs THREE classification tasks using the SAME
underlying reservoir states:
    Head 1: Modulation type classification  (24 classes) -- RadioML data
    Head 2: Jamming type classification      (6 classes)  -- synthetic EW data
    Head 3: Emitter identification           (10 classes) -- synthetic emitter data
Each head is a SEPARATE classifier algorithm, chosen because its
mathematical properties suit that specific task:
    - Multinomial Logistic Regression for modulation (many classes,
      probabilistic output is useful for confidence estimates)
    - Linear SVM for jamming detection (maximises the margin between
      classes, often very effective for well-separated jamming waveforms)
    - Ridge Classifier for emitter ID (handles CORRELATED features well,
      which matters because RF-fingerprint-driven reservoir states from
      similar-hardware emitters may be highly correlated)

IMPORTANT: because each task uses a DIFFERENT dataset (RadioML vs.
synthetic jamming vs. synthetic emitter), each head is trained completely
INDEPENDENTLY, on its own reservoir-state dataset. They share the
ARCHITECTURE PATTERN (reservoir -> readout) but not the actual training
data or weights.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import CONFIG


# ==============================================================================
# Base Readout Wrapper
# ==============================================================================

class ReadoutHead:
    """
    A thin wrapper around a scikit-learn classifier that ALSO handles
    feature standardisation (scaling reservoir states to zero mean / unit
    variance before classification, which is standard practice and
    typically improves linear classifier performance and training
    stability).

    WHY STANDARDISE THE RESERVOIR STATES FIRST?
    Different neurons in the reservoir may have very different typical
    firing rates (some neurons might rarely fire, others fire often,
    purely due to the random connectivity they happened to receive).
    Linear classifiers like logistic regression and SVM are sensitive to
    the SCALE of input features -- a feature with naturally large values
    can dominate the learned decision boundary even if it's not more
    informative than a small-scale feature. Standardisation equalises
    this:
        x_scaled = (x - mean(x)) / std(x)
    so every neuron's firing rate is rescaled to have mean 0 and standard
    deviation 1 across the training set, BEFORE the classifier ever sees it.
    """

    def __init__(self, classifier, name: str):
        self.classifier = classifier
        self.name = name
        self.scaler = StandardScaler()

        '''
        self.pca = PCA(
            n_components=0.99,
            whiten=True,
            random_state=CONFIG["seed"]
        )
        '''

        self.is_fitted = False

    def fit(self, X_states: np.ndarray, y_labels: np.ndarray):
        """Standardises the input states, then fits the classifier."""
        X_scaled = self.scaler.fit_transform(X_states)
        # X_scaled = self.pca.fit_transform(X_scaled)

        self.classifier.fit(X_scaled, y_labels)
        self.is_fitted = True
        return self

    def predict(self, X_states: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"ReadoutHead '{self.name}' has not been fitted yet.")
        X_scaled = self.scaler.transform(X_states)
        # X_scaled = self.pca.transform(X_scaled)

        return self.classifier.predict(X_scaled)

    def predict_proba(self, X_states: np.ndarray) -> np.ndarray:
        """
        Returns class probabilities, if the underlying classifier supports
        it (LogisticRegression does; LinearSVC and RidgeClassifier do not
        natively -- for those we fall back to decision_function scores,
        which are still useful for ROC curve computation even though they
        aren't true probabilities).
        """
        X_scaled = self.scaler.transform(X_states)
        # X_scaled = self.pca.transform(X_scaled)
        if hasattr(self.classifier, "predict_proba"):
            return self.classifier.predict_proba(X_scaled)
        elif hasattr(self.classifier, "decision_function"):
            scores = self.classifier.decision_function(X_scaled)
            # Convert decision scores to pseudo-probabilities using the
            # softmax function, so downstream code (e.g. ROC curve
            # plotting) can treat every readout head uniformly:
            #     softmax(z)_k = exp(z_k) / sum_j exp(z_j)
            if scores.ndim == 1:
                # binary case: decision_function returns 1 score per sample
                scores = np.stack([-scores, scores], axis=1)
            exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
            return exp_scores / exp_scores.sum(axis=1, keepdims=True)
        else:
            raise AttributeError(
                f"Classifier {type(self.classifier)} has neither predict_proba "
                "nor decision_function."
            )

    def evaluate(self, X_states: np.ndarray, y_true: np.ndarray, class_names: list = None) -> dict:
        """
        Computes accuracy, a full classification report (precision,
        recall, F1 per class), and a confusion matrix.
        """
        y_pred = self.predict(X_states)
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True
        )
        cm = confusion_matrix(y_true, y_pred)

        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm,
            "y_pred": y_pred,
        }

    def save(self, filepath: str):
        """Saves the fitted scaler + classifier together as one file."""
        joblib.dump({
        "scaler": self.scaler,
        # "pca": self.pca,
        "classifier": self.classifier,
        "name": self.name
    }, filepath)

    @classmethod
    def load(cls, filepath: str):
        """Loads a previously saved ReadoutHead."""
        bundle = joblib.load(filepath)
        head = cls(bundle["classifier"], bundle["name"])
        head.scaler = bundle["scaler"]
        # head.pca = bundle["pca"]
        head.is_fitted = True
        return head


# ==============================================================================
# Factory Functions for Each Readout Head
# ==============================================================================

def build_modulation_readout(C: float = 1.0, max_iter: int = 1000) -> ReadoutHead:
    """
    HEAD 1: Modulation Classification (24 classes)
    Algorithm: Multinomial Logistic Regression

    WHY LOGISTIC REGRESSION HERE?
    With 24 classes, we want a classifier that naturally outputs a full
    probability distribution over all classes (useful for understanding
    classifier confidence and for later computing ROC/PR curves per
    class). Multinomial logistic regression directly models:

        P(y = k | x) = exp(w_k . x + b_k) / sum_j exp(w_j . x + b_j)

    which is exactly the softmax function applied to a set of linear
    scores -- one linear model per class, normalised so probabilities
    sum to 1 across all 24 classes. With the 'lbfgs' solver and more than
    two classes, scikit-learn automatically uses this multinomial
    (softmax) formulation rather than a one-vs-rest scheme. C controls
    the inverse strength of L2 regularisation (smaller C = more
    regularisation = simpler decision boundary, helps prevent overfitting
    to noise in the reservoir states).
    """
    classifier = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
        random_state=CONFIG["seed"],
    )
    return ReadoutHead(classifier, name="modulation_readout")


def build_jamming_readout(C: float = 1.0, max_iter: int = 2000) -> ReadoutHead:
    """
    HEAD 2: Jamming Type Classification (6 classes)
    Algorithm: Linear Support Vector Machine (Linear SVM)

    WHY LINEAR SVM HERE?
    Linear SVM finds the hyperplane that MAXIMISES THE MARGIN between
    classes -- the decision boundary is placed as far as possible from
    the nearest training examples of each class, rather than simply
    minimising classification error. This tends to generalise well when
    classes are reasonably separable, which is expected for jamming
    waveform classification since the six jamming types in this project
    have quite distinctive temporal/spectral structures.

    The optimisation problem solved is:
        minimize   (1/2)*||w||^2 + C * sum_i max(0, 1 - y_i*(w.x_i + b))
    The first term maximises the margin (by minimising the weight
    vector's norm); the second term ("hinge loss") penalises
    misclassified or too-close-to-the-boundary points, with C
    controlling the trade-off between margin width and classification
    error tolerance.
    """
    classifier = LinearSVC(
        C=C,
        max_iter=max_iter,
        dual = False,
        random_state=CONFIG["seed"],
    )
    return ReadoutHead(classifier, name="jamming_readout")


def build_emitter_readout(alpha: float = 1.0) -> ReadoutHead:
    """
    HEAD 3: Emitter Identification (10 classes)
    Algorithm: Ridge Classifier

    WHY RIDGE CLASSIFIER HERE?
    Ridge Classifier solves a REGULARISED LEAST-SQUARES problem (rather
    than logistic loss or hinge loss):
        minimize   sum_i ||y_i - w.x_i - b||^2 + alpha * ||w||^2
    The L2 penalty term (alpha * ||w||^2) is especially effective when
    INPUT FEATURES ARE CORRELATED with each other -- which is exactly
    the situation we expect for the emitter-identification task: since
    different emitters in our simulation share the same underlying
    carrier/jamming structure and differ only in subtle hardware
    "fingerprint" parameters (frequency drift, phase noise, amplifier
    nonlinearity), the resulting reservoir states for different emitters
    are likely to be highly correlated with each other. Ridge
    regression's regularisation handles this multicollinearity better
    than an unregularised least-squares or even logistic regression
    would, by shrinking correlated coefficients together rather than
    arbitrarily favouring one over another.
    """
    classifier = RidgeClassifier(alpha=alpha, random_state=CONFIG["seed"])
    return ReadoutHead(classifier, name="emitter_readout")


# ==============================================================================
# Multi-Task Trainer -- Convenience Wrapper
# ==============================================================================

class MultiTaskReadout:
    """
    Bundles all three readout heads together for convenience. Note that
    even though they are bundled in one object for ease of use, they are
    STILL trained completely independently -- this class does not share
    any weights or gradients between heads. It simply avoids having to
    juggle three separate variables throughout the rest of the codebase.
    """

    def __init__(self):
        self.modulation_head = build_modulation_readout()
        self.jamming_head = build_jamming_readout()
        self.emitter_head = build_emitter_readout()

    def fit_modulation(self, X_states, y_labels):
        print("Training Head 1 (Modulation classification, Logistic Regression)...")
        self.modulation_head.fit(X_states, y_labels)
        return self

    def fit_jamming(self, X_states, y_labels):
        print("Training Head 2 (Jamming classification, Linear SVM)...")
        self.jamming_head.fit(X_states, y_labels)
        return self

    def fit_emitter(self, X_states, y_labels):
        print("Training Head 3 (Emitter identification, Ridge Classifier)...")
        self.emitter_head.fit(X_states, y_labels)
        return self

    def save_all(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        if self.modulation_head.is_fitted:
            self.modulation_head.save(os.path.join(output_dir, "modulation_readout.joblib"))
        if self.jamming_head.is_fitted:
            self.jamming_head.save(os.path.join(output_dir, "jamming_readout.joblib"))
        if self.emitter_head.is_fitted:
            self.emitter_head.save(os.path.join(output_dir, "emitter_readout.joblib"))
        print(f"Saved trained readout heads to: {output_dir}")


# ==============================================================================
# Script Entry Point (sanity check using random dummy data)
# ==============================================================================

if __name__ == "__main__":
    print("Testing readout heads with random dummy data...\n")

    rng = np.random.default_rng(CONFIG["seed"])

    # Simulate fake "reservoir states" and labels just to confirm every
    # readout head can be fitted and evaluated without errors.
    n_samples = 500
    n_features = 200  # pretend reservoir size

    X_dummy = rng.standard_normal((n_samples, n_features))

    # Head 1 test: 24-class modulation
    y_mod = rng.integers(0, 24, n_samples)
    mod_head = build_modulation_readout()
    mod_head.fit(X_dummy, y_mod)
    mod_results = mod_head.evaluate(X_dummy, y_mod)
    print(f"Modulation readout (dummy data) -- train accuracy: {mod_results['accuracy']:.3f}")
    print("(With random labels and random features, ~1/24 = 0.042 accuracy is EXPECTED.)")

    # Head 2 test: 6-class jamming
    y_jam = rng.integers(0, 6, n_samples)
    jam_head = build_jamming_readout()
    jam_head.fit(X_dummy, y_jam)
    jam_results = jam_head.evaluate(X_dummy, y_jam)
    print(f"\nJamming readout (dummy data) -- train accuracy: {jam_results['accuracy']:.3f}")
    print("(With random labels, ~1/6 = 0.167 accuracy is EXPECTED.)")

    # Head 3 test: 10-class emitter
    y_emit = rng.integers(0, 10, n_samples)
    emit_head = build_emitter_readout()
    emit_head.fit(X_dummy, y_emit)
    emit_results = emit_head.evaluate(X_dummy, y_emit)
    print(f"\nEmitter readout (dummy data) -- train accuracy: {emit_results['accuracy']:.3f}")
    print("(With random labels, ~1/10 = 0.100 accuracy is EXPECTED.)")

    print("\nAll three readout heads trained and evaluated successfully.")
