import pandas as pd
from sklearn.preprocessing import LabelEncoder
import datetime


class FeatureHandler:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.lbl_enc = LabelEncoder()

    def discount_pct(self, dfset: pd.DataFrame) -> pd.Series:
        """
        Function to compute the percentaje of discount in the basket
        Parameters
        ----------
        dfset : pd.DataFrame
            dataframe with the features of the dataset

        Returns
        -------
        pd.Series
            New feature
        """

        dscnt = dfset["Discounted Sales"]/dfset["Total Sales"]

        return dscnt

    def time_between_clicks(self,  dfset: pd.DataFrame) -> pd.Series:
        """
        Function to compute the mean time between clicks in the browsing
        session

        Parameters
        ----------
        dfset : pd.DataFrame
            dataframe with the features of the dataset

        Returns
        -------
        pd.Series
            New feature
        """
        pct_time = (
            dfset["Browsing Duration (minutes)"]/dfset["Number of Clicks"]
        )

        return pct_time

    def discount_to_time(self,  dfset: pd.DataFrame) -> pd.Series:
        """
        Function to compute the time used to search for the discount

        Parameters
        ----------
        dfset : pd.DataFrame
            dataframe with the features of the dataset

        Returns
        -------
        pd.Series
            New feature
        """

        dscnt = dfset["Discounted Sales"]/dfset["Total Sales"]
        discount_ratio_time = dscnt/dfset["Browsing Duration (minutes)"]

        return discount_ratio_time

    def week_day(self,  dfset: pd.DataFrame) -> pd.Series:
        """
        Function to compute the day of the week

        Parameters
        ----------
        dfset : pd.DataFrame
            dataframe with the features of the dataset

        Returns
        -------
        pd.Series
            New feature
        """

        weekday = [day.isoweekday() for day in dfset["Date"]]

        return weekday

    def month_feat(self,  dfset: pd.DataFrame) -> pd.Series:
        """
        Funtion to compute the month of the year

        Parameters
        ----------
        dfset : pd.DataFrame
            dataframe with the features of the dataset

        Returns
        -------
        pd.Series
            New feature
        """

        month = [day.month for day in dfset["Date"]]

        return month

    def run_feat_buider(self) -> None:
        """
        Function to compute all the new features in the
        object self.df
        """
        self.df["discount_pct"] = self.discount_pct(self.df)
        self.df["week_day"] = self.week_day(self.df)
        self.df["month"] = self.month_feat(self.df)
        self.df["time_between_clicks"] = self.time_between_clicks(self.df)
        self.df["discount_to_time"] = self.discount_to_time(self.df)

    def categorical_to_numerical(self) -> None:
        """
        Function to transoform the catecorical variables into numerical ones
        """

        for i, _ in self.df.dtypes[self.df.dtypes == "object"].iteritems():
            self.df.loc[:, i] = self.lbl_enc.fit_transform(self.df.loc[:, i])


if __name__ == "__main__":
    customers = pd.read_csv("/workspaces/new_bpp/data/customer_data.csv")
    trans = pd.read_csv("/workspaces/new_bpp/data/transactions_data.csv")
    trans["Date"] = [
        datetime.datetime.strptime(date_, '%Y-%m-%d')
        for date_ in trans["Date"]
    ]
    complete_df = trans.merge(customers, on="Customer ID")

    vh = FeatureHandler(complete_df)
    vh.run_feat_buider()
    vh.categorical_to_numerical()
    pd.set_option('display.max_columns', None)
    vh.df.describe()
