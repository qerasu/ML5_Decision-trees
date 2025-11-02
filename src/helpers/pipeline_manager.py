from category_encoders import CountEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np


class S21Pipeline:
    def __init__(self, name, model, Xs, ys):
        self.X_train, self.X_val, self.X_test = Xs[0], Xs[1], Xs[2]
        self.y_train,self.y_val, self.y_test = ys[0], ys[1], ys[2]
        self.model = model
        self.model_name = name 


    def _preprocess_builder(self):
        X_train = self.X_train

        cat_cols = X_train.select_dtypes(include=["object","category"]).columns
        num_cols = X_train.select_dtypes(include=["number"]).columns

        num_block = Pipeline([
            ("impute", SimpleImputer(strategy="median")), # for NaN
            ("sc",     StandardScaler())
        ])

        cat_block = Pipeline([
            ("cnt", CountEncoder(cols=cat_cols, handle_unknown=0, handle_missing=0, normalize=True))
        ])

        self.preprocess = ColumnTransformer(
            transformers=[
                ("num", num_block, num_cols),
                ("cat", cat_block, cat_cols),
            ]
        )

        return self

    
    def _pipeline_builder(self):
        pipe = Pipeline([
            ("preprocess", self.preprocess),
            ("model", self.model)
        ])

        self.pipe = pipe.fit(self.X_train, self.y_train)

        return self


    @staticmethod
    def _gini(y_true, pred):
        auc = roc_auc_score(y_true, pred)

        return float(2.0 * auc - 1.0)


    def build_evaluate(self, X=None, y=None):
        self._preprocess_builder()._pipeline_builder()
        X = self.X_val if X is None else X
        y = self.y_val if y is None else y 

        if "Classifier" in self.model_name:
            clf_val_proba = self.pipe.predict_proba(X)[:, 1]
            gini = self._gini(y, clf_val_proba)
        else:
            reg_val_pred = self.pipe.predict(X)
            gini = self._gini(y, reg_val_pred)

        return gini


    # Random search over hyperparameters, keeping the best fitted pipeline.
    def tune_model(
        self,
        estimator_cls,
        base_params,
        param_distributions,
        *,
        model_label=None,
        n_iter: int = 12,
        random_state: int = 42,
        X_eval=None,
        y_eval=None,
    ):
        if n_iter < 1:
            raise ValueError("n_iter must be at least 1.")

        if not param_distributions:
            raise ValueError("param_distributions cannot be empty.")

        eval_X = self.X_val if X_eval is None else X_eval
        eval_y = self.y_val if y_eval is None else y_eval

        if eval_X is None or eval_y is None:
            raise ValueError("Validation data must be provided for tuning.")

        sampler = ParameterSampler(
            param_distributions,
            n_iter=n_iter,
            random_state=np.random.RandomState(random_state),
        )

        original_model = self.model
        original_name = self.model_name
        original_pipe = getattr(self, "pipe", None)
        original_preprocess = getattr(self, "preprocess", None)

        best_score = -np.inf
        best_pipeline = None
        best_model = None
        label = model_label or estimator_cls.__name__

        try:
            for params in sampler:
                init_params = {**base_params, **params}
                candidate_model = estimator_cls(**init_params)
                self.model = candidate_model
                self.model_name = label

                gini = self.build_evaluate(eval_X, eval_y)
                if gini > best_score:
                    best_score = gini
                    best_pipeline = self.pipe
                    best_model = candidate_model
        finally:
            if best_pipeline is None:
                self.model = original_model
                self.model_name = original_name
                self.pipe = original_pipe
                self.preprocess = original_preprocess
            else:
                self.model = best_model
                self.model_name = label
                self.pipe = best_pipeline
                self.preprocess = self.pipe.named_steps.get("preprocess")

        return best_score