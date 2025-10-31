import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from category_encoders import CountEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

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


    def build_evaluate(self, X, y):
        self._preprocess_builder()._pipeline_builder()

        if "Classifier" in self.model_name:
            clf_val_proba = self.pipe.predict_proba(X)[:, 1]
            gini = self._gini(y, clf_val_proba)
        else:
            reg_val_pred = self.pipe.predict(X)
            gini = self._gini(y, reg_val_pred)

        return gini