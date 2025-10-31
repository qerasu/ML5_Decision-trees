from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np

@dataclass
class Node:
    depth: int              # for root: 0; for children: >= 1
    idxs: np.ndarray        # indices of samples belonging to this node
    impurity: float         # for classifier: gini; for regressor: variance
    prediction: np.ndarray  # for classifier: class probs; for regressor: scalar as array([mean])

    # split info
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None

    @property
    def is_leaf(self):
        return self.left is None and self.right is None


class _BaseTree: # Base tree with shared infrastructure
    def __init__(
        self,
        max_depth: int = 7,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0.0,
        random_state: Optional[int] = None,
    ):
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.random_state = np.random.RandomState(random_state)

        # will show up after calling .fit()
        self.X_: Optional[np.ndarray] = None
        self.y_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
        self.n_classes_: Optional[int] = None
        self.root_: Optional[Node] = None


    @property
    def _is_classification(self) -> bool:
        return False


    def _find_best_split(self, idxs: np.ndarray):
        X, y = self.X_, self.y_

        parent_imp = self._impurity(y[idxs])
        best = None

        for feat in range(X.shape[1]):
            x = X[idxs, feat]
            order = np.argsort(x, kind="mergesort")  # stable to keep equal values grouped
            x_sorted = x[order]
            y_sorted = y[idxs][order]

            # candidate thresholds are midpoints between distinct values
            uniq_mask = np.diff(x_sorted, prepend=x_sorted[0] - 1) != 0
            uniq_positions = np.where(uniq_mask)[0]

            # extracting midpoints from between unique consecutive values
            for pos in uniq_positions[1:]:
                left_y = y_sorted[:pos]
                right_y = y_sorted[pos:]

                if left_y.size == 0 or right_y.size == 0:
                    continue

                thr = float((x_sorted[pos - 1] + x_sorted[pos]) / 2.0)
                child_imp, gain = self._score_split(left_y, right_y, parent_imp)

                if (best is None) or (child_imp < best[4]):
                    left_idxs = idxs[order[:pos]]
                    right_idxs = idxs[order[pos:]]
                    best = (feat, thr, left_idxs, right_idxs, child_imp, gain)

        return best
        

    def _find_random_split(self, idxs: np.ndarray, *, max_features: Optional[int], n_thresholds: int = 16):
        X, y = self.X_, self.y_

        parent_imp = self._impurity(y[idxs])
        rng = self.random_state

        d = X.shape[1]                                  # num of features
        m = max_features or int(np.ceil(np.sqrt(d)))    # if max_features not set counts sqrt(d) and rounds it up
        m = max(1, min(d, m))                           # range
        features = rng.choice(d, size=m, replace=False) # choosing m unique features randomly

        best = None
        for feat in features:
            x = X[idxs, feat]
            if x.size == 0:
                continue

            x_min, x_max = np.min(x), np.max(x)
            if x_min == x_max:
                continue

            # sample thresholds uniformly in [min, max]
            for _ in range(max(1, int(n_thresholds))):
                thr = float(rng.uniform(x_min, x_max))
                mask = x < thr
                left_idxs = idxs[mask]
                right_idxs = idxs[~mask]

                if left_idxs.size == 0 or right_idxs.size == 0:
                    continue

                child_imp, gain = self._score_split(self.y_[left_idxs], self.y_[right_idxs], parent_imp)
                if (best is None) or (child_imp < best[4]):
                    best = (feat, thr, left_idxs, right_idxs, child_imp, gain)

        return best


    def _build(self, node: Node, *, splitter: str, max_features: Optional[int], n_thresholds: int) -> None:
        # stopping rules
        depth = node.depth
        idxs = node.idxs

        X, y = self.X_, self.y_
        assert X is not None and y is not None

        if depth >= self.max_depth or idxs.size < self.min_samples_split or np.all(y[idxs] == y[idxs][0]):
            return

        # choose split
        if splitter == "best":
            split = self._find_best_split(idxs)
        elif splitter == "random":
            split = self._find_random_split(idxs, max_features=max_features, n_thresholds=n_thresholds)
        else:
            raise ValueError("splitter must be 'best' or 'random'.")

        if split is None:
            return  # node becomes a leaf

        feat, thr, left_idxs, right_idxs, _child_impurity, impurity_decrease = split
        # left_idxs - idxs of objects that have x[:, feat] < thr; right - remainder

        if impurity_decrease < self.min_impurity_decrease:
            return

        # creating children
        left_node = Node(
            depth=depth + 1,
            idxs=left_idxs,
            impurity=self._impurity(y[left_idxs]),
            prediction=self._leaf_value(y[left_idxs]),
        )

        right_node = Node(
            depth=depth + 1,
            idxs=right_idxs,
            impurity=self._impurity(y[right_idxs]),
            prediction=self._leaf_value(y[right_idxs]),
        )

        node.feature_index = int(feat)
        node.threshold = float(thr)
        node.left = left_node
        node.right = right_node

        # recursively grow left and right child nodes
        self._build(left_node, splitter=splitter, max_features=max_features, n_thresholds=n_thresholds)
        self._build(right_node, splitter=splitter, max_features=max_features, n_thresholds=n_thresholds)


    # inference
    def _traverse(self, x: np.ndarray, node: Node) -> Node:
        while not node.is_leaf:
            assert node.feature_index is not None and node.threshold is not None
            
            if x[node.feature_index] < node.threshold:
                node = node.left  
            else:
                node = node.right 

        return node


    def fit(self, X, y, splitter: str = "best", *, max_features: Optional[int] = None, n_thresholds: int = 16) -> "_BaseTree":
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.ndim != 2:
            raise ValueError("X must be 2D array.")  # [n_samples, n_features]

        if y.ndim != 1:
            raise ValueError("y must be 1D array.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        self.X_ = X
        self.y_ = y
        self.n_features_ = X.shape[1]

        if self._is_classification:
            self.n_classes_ = int(np.max(y)) + 1

        idxs = np.arange(X.shape[0])
        root_imp = self._impurity(y)

        # root creation
        self.root_ = Node(
            depth=0,
            idxs=idxs,
            impurity=root_imp,
            prediction=self._leaf_value(y),
        )

        self._build(self.root_, splitter=splitter, max_features=max_features, n_thresholds=n_thresholds)

        return self


    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise RuntimeError("Call fit() before predict().")

        X = np.asarray(X)
        out = []
        for i in range(X.shape[0]):
            leaf = self._traverse(X[i], self.root_)
            out.append(self._prediction_from_leaf(leaf))

        return np.asarray(out)


class S21DecisionTreeClassifier(_BaseTree):
    def __init__(self, max_depth: int = 7, min_samples_split: int = 2, min_impurity_decrease: float = 0.0, random_state: Optional[int] = None) -> None:
        super().__init__(max_depth, min_samples_split, min_impurity_decrease, random_state)

    @staticmethod
    def _gini_impurity(y: np.ndarray) -> float:
        if y.size == 0:
            return 0.0

        counts = np.bincount(y)
        p = counts[counts > 0].astype(float)
        p /= y.size

        return 1.0 - np.sum(p * p)


    @property
    def _is_classification(self) -> bool:
        return True


    def _impurity(self, y: np.ndarray) -> float:
        return self._gini_impurity(y)


    def _leaf_value(self, y: np.ndarray) -> np.ndarray:
        n_classes = int(np.max(self.y_) + 1) if self.y_ is not None else int(np.max(y) + 1)
        counts = np.bincount(y, minlength=n_classes).astype(float)
        probs = counts / y.size

        return probs


    def _score_split(self, y_left: np.ndarray, y_right: np.ndarray, parent_impurity: float) -> Tuple[float, float]:
        n = y_left.size + y_right.size
        w_imp = (y_left.size * self._gini_impurity(y_left) + y_right.size * self._gini_impurity(y_right)) / n
        gain = parent_impurity - w_imp

        return float(w_imp), float(gain)


    def _prediction_from_leaf(self, leaf: Node) -> int:
        return int(np.argmax(leaf.prediction))


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise RuntimeError("Call fit() before predict_proba().")

        X = np.asarray(X)
        probs = np.zeros((X.shape[0], self.n_classes_), dtype=float)

        for i in range(X.shape[0]):
            leaf = self._traverse(X[i], self.root_)
            probs[i] = leaf.prediction

        return probs


class S21DecisionTreeRegressor(_BaseTree):
    def __init__(self, max_depth: int = 7, min_samples_split: int = 2, min_impurity_decrease: float = 0.0, random_state: Optional[int] = None) -> None:
        super().__init__(max_depth, min_samples_split, min_impurity_decrease, random_state)


    @staticmethod
    def _variance_impurity(y: np.ndarray) -> float:
        if y.size == 0:
            return 0.0
            
        return float(np.var(y, ddof=0))


    def _impurity(self, y: np.ndarray) -> float:
        return self._variance_impurity(y)


    def _leaf_value(self, y: np.ndarray) -> np.ndarray:
        return np.asarray([float(np.mean(y))])


    def _score_split(self, y_left: np.ndarray, y_right: np.ndarray, parent_impurity: float) -> Tuple[float, float]:
        n = y_left.size + y_right.size
        w_imp = (y_left.size * self._variance_impurity(y_left) + y_right.size * self._variance_impurity(y_right)) / n
        gain = parent_impurity - w_imp

        return float(w_imp), float(gain)


    def _prediction_from_leaf(self, leaf: Node) -> float:
        return float(leaf.prediction[0])


class S21RandomForestClassifier:
    def __init__(
        self,
        n_estimators: int = 10,
        max_depth: int = 7,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0.0,
        max_features: Optional[int] = None,
        bootstrap: bool = True,
        splitter: str = "best",
        n_thresholds: int = 16,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.max_features = max_features
        self.bootstrap = bool(bootstrap)
        self.splitter = splitter
        self.n_thresholds = int(n_thresholds)
        self.random_state = np.random.RandomState(random_state)

        # will show up after calling .fit()
        self.estimators_: List[S21DecisionTreeClassifier] = []
        self.n_features_in_: Optional[int] = None
        self.n_classes_: Optional[int] = None


    def fit(self, X, y) -> "S21RandomForestClassifier":
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be 2D array.")  # [n_samples, n_features]

        if y.ndim != 1:
            raise ValueError("y must be 1D array.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive.")

        n_samples = X.shape[0]
        self.n_features_in_ = X.shape[1]
        self.n_classes_ = int(np.max(y)) + 1
        self.estimators_ = []

        max_features = self.max_features
        if max_features is None:
            # default RandomForest heuristic: sqrt(d) features per split
            max_features = int(np.sqrt(self.n_features_in_))
            max_features = max(1, max_features)

        for _ in range(self.n_estimators):
            if self.bootstrap:
                # bootstrap sample with replacement
                idxs = self.random_state.choice(n_samples, size=n_samples, replace=True)
            else:
                # or use the whole dataset without resampling
                idxs = np.arange(n_samples)

            # individual seed per tree keeps randomness reproducible across trees
            tree_seed = self.random_state.randint(0, 2**31 - 1)
            tree = S21DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_impurity_decrease=self.min_impurity_decrease,
                random_state=int(tree_seed),
            )
            tree.fit(
                X[idxs],
                y[idxs],
                splitter=self.splitter,
                max_features=max_features,
                n_thresholds=self.n_thresholds,
            )
            self.estimators_.append(tree)

        return self


    def predict_proba(self, X) -> np.ndarray:
        if not self.estimators_:
            raise RuntimeError("Call fit() before predict_proba().")

        X = np.asarray(X)
        probs = np.zeros((X.shape[0], self.n_classes_), dtype=float)

        for tree in self.estimators_:
            probs += tree.predict_proba(X)

        probs /= len(self.estimators_)

        return probs


    def predict(self, X) -> np.ndarray:
        probs = self.predict_proba(X)  # average probabilities, then take the argmax

        return np.argmax(probs, axis=1)


class S21GradientBoostingClassifier:
    def __init__(
        self,
        *,
        number_of_trees: int = 50,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        max_features: Optional[int] = None,
        min_samples_split: int = 2,
        random_state: Optional[int] = None,
    ) -> None:
        self.number_of_trees = int(number_of_trees)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.max_features = max_features
        self.min_samples_split = int(min_samples_split)
        self.random_state = np.random.RandomState(random_state)

        self.init_score_: Optional[float] = None
        self.estimators_: List[S21DecisionTreeRegressor] = []


    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -20.0, 20.0)

        return 1.0 / (1.0 + np.exp(-z))


    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.init_score_ is None:
            raise RuntimeError("Call fit() before predict().")

        X = np.asarray(X)
        raw = np.full(X.shape[0], self.init_score_, dtype=float)

        for tree in self.estimators_:
            raw += self.learning_rate * tree.predict(X) # tuning model

        return raw


    def fit(self, X: np.ndarray, y: np.ndarray) -> "S21GradientBoostingClassifier":
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be 2D array.")

        if y.ndim != 1:
            raise ValueError("y must be 1D array.")

        uniques = np.unique(y)
        if uniques.size != 2 or not np.all(np.isin(uniques, [0, 1])):
            raise ValueError("y must contain exactly two classes encoded as 0 and 1.")

        if self.number_of_trees <= 0:
            raise ValueError("number_of_trees must be positive.")

        pos_rate = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        self.init_score_ = float(np.log(pos_rate / (1.0 - pos_rate)))

        current_score = np.full(y.shape[0], self.init_score_, dtype=float)
        self.estimators_ = []

        for _ in range(self.number_of_trees): # creates small stupid trees and tunes the next on previous
            proba = self._sigmoid(current_score)
            residual = y - proba

            tree_seed = self.random_state.randint(0, 2**31 - 1) # as in sklearn
            tree = S21DecisionTreeRegressor( 
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=int(tree_seed),
            )

            tree.fit(
                X,
                residual,
                splitter="best",
                max_features=self.max_features,
            )

            update = tree.predict(X)
            current_score += self.learning_rate * update
            self.estimators_.append(tree)

        return self


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self._decision_function(X)
        proba_pos = self._sigmoid(raw)
        proba_neg = 1.0 - proba_pos

        return np.column_stack([proba_neg, proba_pos])


    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)

        return (proba[:, 1] >= 0.5).astype(int)