import pandas as pd


class S21SplitByThirds:
    def __init__(self, df, date_column: str = "PurchDate"):
        if date_column not in df.columns:
            raise KeyError(f"Column `{date_column}` is not present in the provided DataFrame.")

        self.date_column = date_column
        self.df = self._prepare(df, date_column)


    @staticmethod
    def _prepare(df, column):
        ordered = df.copy()
        ordered[column] = pd.to_datetime(ordered[column], errors="coerce")

        if ordered[column].isna().any():
            raise ValueError(f"Column `{column}` contains invalid dates that could not be parsed.")

        return ordered.sort_values(column).reset_index(drop=True)


    def _cut_dates(self):
        u = self.df[self.date_column].drop_duplicates().to_numpy()
        if len(u) < 3:
            raise ValueError("Need at least 3 distinct dates to split into 3 parts.")

        i1 = max(1, int(round(len(u) / 3)))
        i2 = max(i1 + 1, int(round(2 * len(u) / 3)))
        d1 = u[i1]
        d2 = u[i2]

        return d1, d2


    def split(self):
        d1, d2 = self._cut_dates()
        dt = self.df[self.date_column]

        df_train = self.df[dt < d1].copy()
        df_valid = self.df[(dt >= d1) & (dt < d2)].copy()
        df_test  = self.df[dt >= d2].copy()

        return df_train.reset_index(drop=True), df_valid.reset_index(drop=True), df_test.reset_index(drop=True)