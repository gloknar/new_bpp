import pandas as pd
from feat.feature_builder import FeatureHandler
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from models.model_handler import ModelHandler

customers = pd.read_csv("/workspaces/new_bpp/data/customer_data.csv")
trans = pd.read_csv("/workspaces/new_bpp/data/transactions_data.csv")
trans["Date"] = [
    datetime.datetime.strptime(date_, '%Y-%m-%d')
    for date_ in trans["Date"]
]
complete_df = trans.merge(
        customers,
        on="Customer ID"
    ).drop("Loyalty Points", axis=1)

scaler = StandardScaler()


vh = FeatureHandler(complete_df)
vh.run_feat_buider()
vh.categorical_to_numerical()

mh = ModelHandler("/workspaces/new_bpp/config/config_all.yaml")
vh.df[mh.features] = scaler.fit_transform(vh.df[mh.features])

train_df, test_df, target, target_test = train_test_split(
    vh.df.loc[:, mh.features],
    vh.df["Incomplete Transaction"],
    test_size=0.20,
    random_state=77
)

xgboost, xgboost_pred = mh.kfold_testing(train_df, target, test_df, "xgboost")
ada, ada_pred = mh.kfold_testing(train_df, target, test_df, "ada")
rforest, rforest_pred = mh.kfold_testing(train_df, target, test_df, "rforest")
gboost, gboost_pred = mh.kfold_testing(train_df, target, test_df, "gboost")
knn, knn_pred = mh.kfold_testing(train_df, target, test_df, "knn")

_, _, c1 = mh.train_predict_model(train_df, target, test_df, "xgboost")
_, _, c2 = mh.train_predict_model(train_df, target, test_df, "ada")
_, _, c3 = mh.train_predict_model(train_df, target, test_df, "rforest")
_, _, c4 = mh.train_predict_model(train_df, target, test_df, "gboost")
_, _, c5 = mh.train_predict_model(train_df, target, test_df, "knn")
results = pd.DataFrame(
    mh.report_metrics(test_df, target_test, [c1, c2, c3, c4, c5])
)

results.to_csv("/workspaces/new_bpp/results/kfold_results.csv")
