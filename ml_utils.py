import pandas as pd
import numpy as np
import ipywidgets as widgets
import math
from IPython.display import display

##Seaborn for fancy plots. 
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (8,8)

class edaDF:
    """
    A class used to perform common EDA tasks

    ...

    Attributes
    ----------
    data : dataframe
        a dataframe on which the EDA will be performed
    target : str
        the name of the target column
    cat : list
        a list of the names of the categorical columns
    num : list
        a list of the names of the numerical columns

    Methods
    -------
    setCat(catList)
        sets the cat variable listing the categorical column names to the list provided in the argument catList
        
        Parameters
        ----------
        catlist : list
            The list of column names that are categorical

    setNum(numList)
        sets the cat variable listing the categorical column names to the list provided in the argument catList
        
        Parameters
        ----------
        numlist : list
            The list of column names that are numerical

    countPlots(self, splitTarg=False, show=True)
        generates countplots for the categorical variables in the dataset 

        Parameters
        ----------
        splitTarg : bool
            If true, use the hue function in the countplot to split the data by the target value
        show : bool
            If true, display the graphs when the function is called. Otherwise the figure is returned.
    
    histPlots(self, splitTarg=False, show=True)
        generates countplots for the categorical variables in the dataset 

        Parameters
        ----------
        splitTarg : bool
            If true, use the hue function in the countplot to split the data by the target value
        show : bool
            If true, display the graphs when the function is called. Otherwise the figure is returned. 

    fullEDA()
        Displays the full EDA process. 
    """
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.cat = []
        self.num = []

    def info(self):
        return self.data.info()

    def giveTarget(self):
        return self.target
        
    def setCat(self, catList):
        self.cat = catList
    
    def setNum(self, numList):
        self.num = numList

    def countPlots(self, splitTarg=False, show=True):
        n = len(self.cat)
        cols = 2
        figure, ax = plt.subplots(math.ceil(n/cols), cols)
        r = 0
        c = 0
        for col in self.cat:
            if splitTarg == False:
                sns.countplot(data=self.data, x=col, ax=ax[r][c])
            if splitTarg == True:
                sns.countplot(data=self.data, x=col, hue=self.target, ax=ax[r][c])
            c += 1
            if c == cols:
                r += 1
                c = 0
        if show == True:
            figure.show()
        return figure

    def histPlots(self, kde=True, splitTarg=False, show=True):
        n = len(self.num)
        cols = 2
        figure, ax = plt.subplots(math.ceil(n/cols), cols)
        r = 0
        c = 0
        for col in self.num:
            #print("r:",r,"c:",c)
            if splitTarg == False:
                sns.histplot(data=self.data, x=col, kde=kde, ax=ax[r][c])
            if splitTarg == True:
                sns.histplot(data=self.data, x=col, hue=self.target, kde=kde, ax=ax[r][c])
            c += 1
            if c == cols:
                r += 1
                c = 0
        if show == True:
            figure.show()
        return figure

    def fullEDA(self):
        out1 = widgets.Output()
        out2 = widgets.Output()
        out3 = widgets.Output()
        out4 = widgets.Output()

        tab = widgets.Tab(children = [out1, out2, out3])
        tab.set_title(0, 'Info')
        tab.set_title(1, 'Categorical')
        tab.set_title(2, 'Numerical')
        display(tab)

        with out1:
            self.info()

        with out2:
            fig2 = self.countPlots(splitTarg=True, show=False)
            plt.show(fig2)
        
        with out3:
            fig3 = self.histPlots(kde=True, show=False)
            plt.show(fig3)

## My version of the EDA class
class v_edaDF:
    
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.cat = []
        self.num = []

    def info(self):
        return self.data.info(verbose=True)
    
    def desc(self):
        return self.data.describe()
    
    def first(self):
        return self.data.head()
    
    def last(self):
        return self.data.tail()
    
    def random_sample(self, x):
        return self.data.sample(x)
    
    def nulls_dupes(self):
        null_count = self.data.isna().sum().sum()
        dupe_count = self.data.duplicated().sum()
        return null_count, dupe_count
    
    def target_corr(self):
        df_corr = self.data._get_numeric_data()
        return df_corr.corrwith(df_corr[self.target]).sort_values(ascending=True)

    def giveTarget(self):
        return self.target
        
    def setCat(self, catList):
        self.cat = catList
    
    def setNum(self, numList):
        self.num = numList

    def countPlots(self, splitTarg=False, show=True):
        figure = 0
        if len(self.cat) > 0:
            n_rows = 5
            n_cols = 5

            figure, axes = plt.subplots(n_rows, n_cols, figsize=(15,15))
            axes = axes.flatten()

            for i, col in enumerate(self.cat):
                if i < len(axes):
                    if splitTarg == False:
                        sns.countplot(data=self.data, x=col, ax=axes[i])
                        axes[i].set_title(col)
                        # axes[i].set_xlabel("")
                        # axes[i].set_ylabel("")
                    if splitTarg == True:
                        sns.countplot(data=self.data, x=col, hue=self.target, ax=axes[i])
                        axes[i].set_title(col)
                        # axes[i].set_xlabel("")
                        # axes[i].set_ylabel("")
            if show == True:
                figure.show()

        return figure

    def histPlots(self, kde=True, splitTarg=False, show=True):
        figure = None
        if len(self.num) > 0:
            n_rows = 12
            n_cols = 12

            figure, axes = plt.subplots(n_rows, n_cols, figsize=(15,15))
            axes = axes.flatten()

            for i, col in enumerate(self.num):
                if i < len(axes):
                    if splitTarg == False:
                        sns.histplot(data=self.data.sample(100), x=col, kde=kde, ax=axes[i])
                        axes[i].set_title(col)
                        axes[i].set_xlabel("")
                        axes[i].set_ylabel("")
                    if splitTarg == True:
                        sns.histplot(data=self.data.sample(100), x=col, hue=self.target, kde=kde, ax=axes[i])
                        axes[i].set_title(col)
                        axes[i].set_xlabel("")
                        axes[i].set_ylabel("")
            if show == True:
                figure.show()
        
        return figure

    
    def pairPlot(self, splitTarg=False, show=True):
        if splitTarg == False:
            figure = sns.pairplot(data=self.data.sample(10))
        if splitTarg == True:
            figure = sns.pairplot(data=self.data.sample(10), hue=self.target)
        if show == True:
            figure.show()

        return figure

    def fullEDA(self, pair_plot=False, hist_plot=False, count_plot=False, sample=5):
        out1 = widgets.Output()
        out2 = widgets.Output()
        out3 = widgets.Output()
        out4 = widgets.Output()
        out5 = widgets.Output()
        out6 = widgets.Output()
        out7 = widgets.Output()
        out8 = widgets.Output()
        out9 = widgets.Output()
        out10 = widgets.Output()

        
        tab = widgets.Tab(children = [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10])
        tab.set_title(0, "Info")
        tab.set_title(1, "Describe")
        tab.set_title(2, "Head")
        tab.set_title(3, "Tail")
        tab.set_title(4, "Sample")
        tab.set_title(5, "Null_Dupes_Count")
        tab.set_title(6, "Correlations")
        tab.set_title(7, "Categorical")
        tab.set_title(8, "Numerical")
        tab.set_title(9, "Sample_Pairplot")
        display(tab)

        with out1:
            self.info()

        with out2:
            display(self.desc())

        with out3:
            display(self.first())

        with out4:
            display(self.last())

        with out5:
            display(self.random_sample(sample))

        with out6:
            print("Null values:", self.nulls_dupes()[0])
            print("Duplicate rows:", self.nulls_dupes()[1])

        with out7:
            print(self.target_corr())

        if count_plot == True:
            with out8:
                fig2 = self.countPlots(splitTarg=True, show=False)
                plt.tight_layout
                plt.show(fig2)
        
        if hist_plot == True:
            with out9:
                fig3 = self.histPlots(kde=True, show=False)
                plt.tight_layout
                plt.show(fig3)

        if pair_plot == True:
            with out10:
                fig4 = self.pairPlot(splitTarg=True, show=False)
                plt.tight_layout
                plt.show(fig4)

## For visualizaing classifier performance
def plot_roc_curve(fper, tper):
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.show()


## Dropping Low Frequency Categories
def replace_low_freq(d, col, threshold=10, replacement='Other'):
    value_counts = d[col].value_counts() # Specific column 
    to_remove = value_counts[value_counts <= threshold].index
    tmp = d[col].replace(to_replace=to_remove, value=replacement)
    return tmp