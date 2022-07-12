import pandas as pd
import numpy as np
from typing import Tuple
from feat.feature_builder import FeatureHandler

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
import yaml
from yaml.loader import SafeLoader
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from sklearn import neighbors

from sklearn import metrics
from sklearn.metrics import f1_score, recall_score


class ModelHandler:

    def __init__(self, config_file: str):

        # Open the file and load the file
        with open(config_file) as f:
            data = yaml.load(f, Loader=SafeLoader)

        self.features = data["features"]
        estimators = [
            (
                'rf',
                RandomForestClassifier(
                    n_estimators=1000,
                )
            ),
            ('gnb', GaussianNB())
        ]

        self.models = {
            "gboost": GradientBoostingClassifier(
                n_estimators=data["gboost"]["n_estimators"],
                learning_rate=data["gboost"]["learning_rate"],
                max_depth=data["gboost"]["max_depth"],
            ),
            "rforest": RandomForestClassifier(
                n_estimators=data["rforest"]["n_estimators"]
            ),
            "gnb": GaussianNB(),
            "ada": AdaBoostClassifier(
                n_estimators=data["ada"]["n_estimators"],
                learning_rate=data["ada"]["learning_rate"]
            ),
            "stack": StackingClassifier(
                estimators=estimators,
                final_estimator=AdaBoostClassifier(
                    n_estimators=data["ada"]["n_estimators"],
                    learning_rate=data["ada"]["learning_rate"]
                )
            ),
            "xgboost": XGBClassifier(
                learning_rate=data["xgboost"]["learning_rate"],
                n_estimators=data["xgboost"]["n_estimators"],
                gamma=data["xgboost"]["gamma"],
                subsample=data["xgboost"]["subsample"],
                colsample_bytree=data["xgboost"]["colsample_bytree"],
                objective=data["xgboost"]["objective"],
                nthread=data["xgboost"]["nthread"],
                scale_pos_weight=data["xgboost"]["scale_pos_weight"],
                max_depth=data["xgboost"]["max_depth"],
                min_child_weight=data["xgboost"]["min_child_weight"]
            ),
            "knn": neighbors.KNeighborsClassifier(15, weights="uniform"),
        }

    def kfold_testing(
        self,
        train: pd.DataFrame,
        train_y: pd.Series,
        test: pd.DataFrame,
        nmodel: str = "rforest"
    ) -> Tuple[float, np.array]:
        """
        This function run a 10-fold over a train set to measure
        the robustness of the model.

        Input:
        - train: Dataframe with the training variables for the model
        - train_y: Training labels for the model
        - test: Dataframe with the testing variables for the model
        - nmodel: name of the model to use

        Output:
        - ROC curve usin the 10-fold
        - Predictions over the test set
        """

        if nmodel not in list(self.models.keys()):
            raise ValueError("Model not implemented")
        folds = StratifiedKFold(n_splits=10, shuffle=False)
        oof = np.zeros(len(train))
        predictions = np.zeros(len(test))

        for fold_, (trn_idx, val_idx) in enumerate(
            folds.split(train.values, train_y.values)
        ):
            print(f"Fold {fold_}")
            model = self.models[nmodel]
            clf = model.fit(
                train.iloc[trn_idx, :][self.features],
                train_y.iloc[trn_idx]
            )
            if nmodel != "knn":
                oof[val_idx] = [
                    i[1]
                    for i in clf.predict_proba(
                        train.iloc[val_idx][self.features]
                    )
                ]
            else:
                oof[val_idx] = clf.predict(train.iloc[val_idx][self.features])

            predictions += clf.predict(test[self.features]) / folds.n_splits

        return roc_auc_score(train_y, oof), predictions

    def train_predict_model(
        self,
        train: pd.DataFrame,
        train_y: pd.Series,
        test: pd.DataFrame,
        nmodel: str = "rforest"
    ) -> Tuple[np.array, np.array]:
        """
        Function to train the model over a train dataset and predict
        the results over a test dataset.

         Parameters
        ----------
        - train: Dataframe with the training variables for the model
        - train_y: Training labels for the model
        - test: Dataframe with the testing variables for the model
        - nmodel: Name of the model to use

        Returns
        -------
        - ROC curve usin the 10-fold
        - Predictions over the test set
        """
        if nmodel not in list(self.models.keys()):
            raise ValueError("Model not implemented")

        model = self.models[nmodel]
        clf = model.fit(
            train.loc[:, self.features],
            train_y
        )
        predictions = clf.predict(test.loc[:, self.features])
        proba = clf.predict_proba(test.loc[:, self.features])

        return predictions, proba, clf

    def report_metrics(
        self,
        test: pd.DataFrame,
        test_y: pd.Series,
        models: list,
    ) -> dict:
        """_summary_

        Parameters
        ----------
        test : pd.DataFrame
            Dataset with the features to use in the model
        test_y : pd.Series
            Golden standar in the test set
        models : list
            List of trained models to compute the metrics

        Returns
        -------
        dict
            _description_
        """
        results = {
            "f1": [],
            "recall": [],
            "auc": []
        }
        for model in models:
            preds_prob = model.predict_proba(test)
            prob = [1 if i[1] >= 0.70 else 0 for i in preds_prob]
            results["f1"].append(f1_score(test_y, prob, average='weighted'))
            results["recall"].append(
                recall_score(
                    test_y,
                    prob,
                    average='weighted'
                )
            )
            fpr, tpr, _ = metrics.roc_curve(
                test_y.values, [i[0] for i in preds_prob],
                pos_label=0
            )
            results["auc"].append(metrics.auc(fpr, tpr))

        return results

    @staticmethod
    def find_opt_cutoff(target, predicted):
        fpr, tpr, threshold = roc_curve(target, predicted)
        i = np.arange(len(tpr))
        roc = pd.DataFrame(
            {
                'tf': pd.Series(tpr-(1-fpr), index=i),
                'threshold': pd.Series(threshold, index=i)
            }
        )
        roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

        return list(roc_t['threshold'])


if __name__ == "__main__":
    customers = pd.read_csv("/workspaces/new_bpp/data/customer_data.csv")
    trans = pd.read_csv("/workspaces/new_bpp/data/transactions_data.csv")
    import datetime
    trans["Date"] = [
        datetime.datetime.strptime(date_, '%Y-%m-%d')
        for date_ in trans["Date"]
    ]
    complete_df = trans.merge(
        customers,
        on="Customer ID"
    ).drop("Loyalty Points", axis=1)

    vh = FeatureHandler(complete_df)
    vh.run_feat_buider()
    vh.categorical_to_numerical()

    mh = ModelHandler("/workspaces/new_bpp/config/config_trans.yaml")
    train_df, test_df, target, target_test = train_test_split(
        vh.df.loc[:, mh.features],
        vh.df["Incomplete Transaction"],
        test_size=0.20,
        random_state=77
    )

    a, b = mh.kfold_testing(train_df, target, test_df, target_test, "ada")
    print(a)
    _, _, c1 = mh.train_predict_model(train_df, target, test_df, "xgboost")
    _, _, c2 = mh.train_predict_model(train_df, target, test_df, "ada")
    _, _, c3 = mh.train_predict_model(train_df, target, test_df, "rforest")
    _, _, c4 = mh.train_predict_model(train_df, target, test_df, "gboost")

    print(mh.report_metrics(test_df, target_test, [c1, c2, c3, c4]))
