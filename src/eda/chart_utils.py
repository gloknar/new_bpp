import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_categorical_variable(
    variable: np.array,
    index_values: np.array,
    violin: bool = True
) -> None:
    """
    Function to plot categorical variables as
    a count of values or relation with the index.

    Parameters:
    -----------
        variable: Variable to plot.
        index_values: index of a dataframe or a Serie
        violin: boolean variable to plot or not the violin chart
    """
    if violin:
        _, ax = plt.subplots(1, 2, figsize=(20, 5))
        sns.countplot(variable, ax=ax[0], palette="husl")
        sns.violinplot(x=variable, y=index_values, ax=ax[1], palette="husl")
        sns.stripplot(
            x=variable,
            y=index_values,
            jitter=True,
            ax=ax[1],
            color="black",
            size=0.5,
            alpha=0.5
        )
        ax[1].set_xlabel("Target")
        ax[1].set_ylabel("Index")
        ax[0].set_xlabel("Target")
        ax[0].set_ylabel("Counts")
    else:
        sns.set(rc={'figure.figsize': (20, 10)})
        sns.countplot(variable, palette="husl")
        plt.xticks(rotation=90)
