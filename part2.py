import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import plotly.express as px
import timeit

df = pd.read_csv("https://raw.githubusercontent.com/abigel87/project_work/main/Healthcare%20Providers.csv")

df.head()
df.isnull().sum()

df['First Name of the Provider'].fillna(df['First Name of the Provider'].mode()[0], inplace=True)
df['Middle Initial of the Provider'].fillna(df['Middle Initial of the Provider'].mode()[0], inplace=True)
df['Credentials of the Provider'].fillna(df['Credentials of the Provider'].mode()[0], inplace=True)
df['Gender of the Provider'].fillna(df['Gender of the Provider'].mode()[0], inplace=True)
df['Street Address 2 of the Provider'].fillna(df['Street Address 2 of the Provider'].mode()[0], inplace=True)

df.isnull().sum()

df.dtypes

def replace_comma(x):
    return x.replace(",", "")

df["Number of Services"] = pd.to_numeric(df["Number of Services"].apply(lambda x: replace_comma(x)))
df["Number of Medicare Beneficiaries"] = pd.to_numeric(df["Number of Medicare Beneficiaries"].apply(lambda x: replace_comma(x)))
df["Number of Distinct Medicare Beneficiary/Per Day Services"] = pd.to_numeric(df["Number of Distinct Medicare Beneficiary/Per Day Services"].apply(lambda x: replace_comma(x)))

df["Average Medicare Allowed Amount"] = pd.to_numeric(df["Average Medicare Allowed Amount"].apply(lambda x: replace_comma(x)))
df["Average Submitted Charge Amount"] = pd.to_numeric(df["Average Submitted Charge Amount"].apply(lambda x: replace_comma(x)))
df["Average Medicare Payment Amount"] = pd.to_numeric(df["Average Medicare Payment Amount"].apply(lambda x: replace_comma(x)))
df["Average Medicare Standardized Amount"] = pd.to_numeric(df["Average Medicare Standardized Amount"].apply(lambda x: replace_comma(x)))

df.dtypes

df= df.astype({'Number of Services':'int','Number of Medicare Beneficiaries': 'int', 'Number of Distinct Medicare Beneficiary/Per Day Services': 'int', 'Average Medicare Allowed Amount' : 'int', 'Average Submitted Charge Amount' : 'int',  'Average Medicare Payment Amount': 'int',  'Average Medicare Standardized Amount': 'int'})


list_str_obj_cols = df.columns[df.dtypes == "object"].tolist()
for str_obj_col in list_str_obj_cols:
    df[str_obj_col] = df[str_obj_col].astype("category")

df.dtypes
df_train = df.copy()


corr = df_train.corr()
corr_abs = corr.abs()
upper_tri = corr_abs.where(np.triu(np.ones(corr_abs.shape),k=1).astype(bool))
print(upper_tri)

#upper_tri.to_excel(r"C:\Users\gpocs001\Desktop\ELTE_POSTGRAD\Project_work\test.xlsx")

to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
print(to_drop)
to_drop.remove("National Provider Identifier")
print(to_drop)

df_train = df_train.drop(df_train[to_drop], axis=1)
df_train.shape

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df_train_num =df_train.select_dtypes(include=numerics)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_train[df_train_num.columns] = scaler.fit_transform(df_train_num)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
list_cat_cols = df_train.columns[df_train.dtypes == "category"].tolist()
list_cat_cols
for str_cat_col in list_cat_cols:
   df_train[str_cat_col] = encoder.fit_transform(df_train[str_cat_col])

df_train.head()

from pyod.models.abod import ABOD
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.cblof import CBLOF
from pyod.models.copod import COPOD
from pyod.models.lof import LOF
from pyod.models.pca import PCA
from pyod.models.hbos import HBOS
from pyod.models.sos import SOS
from pyod.models.ecod import ECOD

outlier_percent = 0.08

model_dict = {
    "ABOD" : ABOD(contamination=outlier_percent),
    #"Fast-ABOD" : ABOD(contamination=outlier_percent, method='fast', n_neighbors=10), -ez nem segít
    "KNN" : KNN(contamination=outlier_percent),
    "Isolation Forest" : IForest(contamination=outlier_percent),
    "LOF" : LOF(contamination=outlier_percent),
    #"CBLOF" : CBLOF(contamination=outlier_percent),
    "COPOD" : COPOD(contamination=outlier_percent),
    "PCA" : PCA(contamination=outlier_percent),
    "HBOS" : HBOS(contamination=outlier_percent),
    "SOS"  : SOS(contamination=outlier_percent),
    "ECOD" : ECOD(contamination=outlier_percent)
}
result_df = pd.DataFrame({"Package Name":[],"Model Name":[],"Time Taken in Seconds":[],"No.Of Data Points":[],"No.Of Outliers":[],"Percentage":[]})
package_name = "PyOD"

result_noniqr = pd.DataFrame({"Package Name":[],"Model Name":[],"Outliers Predicted":[],"Non-Outliers Predicted":[]})

outlier_df = pd.DataFrame()
n_points = df_train.shape[0]


def makeDataFrameNonIQR(clf_name, pred):
    global result_noniqr
    predicted = df_train[pred == 1]
    predicted_no = df_train[pred == 0]
    outliers_predicted = sum(pred == 1)
    nonoutliers_predicted = sum(pred == 0)

    cur = {"Package Name": package_name, "Model Name": clf_name, "Outliers Predicted": outliers_predicted,
           "Non-Outliers Predicted": nonoutliers_predicted}
    result_noniqr = result_noniqr.append(cur, ignore_index=True)


clf_name = "ABOD"
clf = ABOD(contamination=outlier_percent)

#for clf_name, clf in model_dict.items():
# Starting Timer
start_time = timeit.default_timer()
clf.fit(df_train)
y_decision_scores = clf.decision_scores_
# Stopping Timer
stop_time = timeit.default_timer()

makeDataFrameNonIQR(clf_name, clf.labels_)

diff = np.round(stop_time - start_time, 2)
labels = clf.labels_
n_outliers = np.count_nonzero(labels)
predicted_outlier_percent = np.round((n_outliers / n_points) * 100, 0)

fig = plt.figure()
plt.title(clf_name)
plt.hist(labels)
plt.show()

outlier_df[clf_name + " Outliers"] = labels
outlier_df[clf_name + " Distance"] = np.round(y_decision_scores, 2)

cur_model = {"Package Name": package_name, "Model Name": clf_name, "Time Taken in Seconds": diff,
             "No.Of Data Points": n_points, "No.Of Outliers": n_outliers, "Percentage": predicted_outlier_percent}
result_df = result_df.append(cur_model, ignore_index=True)


##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################

clf_name = "KNN"
clf = KNN(contamination=outlier_percent)

#for clf_name, clf in model_dict.items():
# Starting Timer
start_time = timeit.default_timer()
clf.fit(df_train)
y_decision_scores = clf.decision_scores_
# Stopping Timer
stop_time = timeit.default_timer()

makeDataFrameNonIQR(clf_name, clf.labels_)

diff = np.round(stop_time - start_time, 2)
labels = clf.labels_
n_outliers = np.count_nonzero(labels)
predicted_outlier_percent = np.round((n_outliers / n_points) * 100, 0)

fig = plt.figure()
plt.title(clf_name)
plt.hist(labels)
plt.show()

outlier_df[clf_name + " Outliers"] = labels
outlier_df[clf_name + " Distance"] = np.round(y_decision_scores, 2)

cur_model = {"Package Name": package_name, "Model Name": clf_name, "Time Taken in Seconds": diff,
             "No.Of Data Points": n_points, "No.Of Outliers": n_outliers, "Percentage": predicted_outlier_percent}
result_df = result_df.append(cur_model, ignore_index=True)

##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################

clf_name = "Isolation Forest"
clf = IForest(contamination=outlier_percent)

#for clf_name, clf in model_dict.items():
# Starting Timer
start_time = timeit.default_timer()
clf.fit(df_train)
y_decision_scores = clf.decision_scores_
# Stopping Timer
stop_time = timeit.default_timer()

makeDataFrameNonIQR(clf_name, clf.labels_)

diff = np.round(stop_time - start_time, 2)
labels = clf.labels_
n_outliers = np.count_nonzero(labels)
predicted_outlier_percent = np.round((n_outliers / n_points) * 100, 0)

fig = plt.figure()
plt.title(clf_name)
plt.hist(labels)
plt.show()

outlier_df[clf_name + " Outliers"] = labels
outlier_df[clf_name + " Distance"] = np.round(y_decision_scores, 2)

cur_model = {"Package Name": package_name, "Model Name": clf_name, "Time Taken in Seconds": diff,
             "No.Of Data Points": n_points, "No.Of Outliers": n_outliers, "Percentage": predicted_outlier_percent}
result_df = result_df.append(cur_model, ignore_index=True)

##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################

clf_name = "LOF"
clf = LOF(contamination=outlier_percent)

#for clf_name, clf in model_dict.items():
# Starting Timer
start_time = timeit.default_timer()
clf.fit(df_train)
y_decision_scores = clf.decision_scores_
# Stopping Timer
stop_time = timeit.default_timer()

makeDataFrameNonIQR(clf_name, clf.labels_)

diff = np.round(stop_time - start_time, 2)
labels = clf.labels_
n_outliers = np.count_nonzero(labels)
predicted_outlier_percent = np.round((n_outliers / n_points) * 100, 0)

fig = plt.figure()
plt.title(clf_name)
plt.hist(labels)
plt.show()

outlier_df[clf_name + " Outliers"] = labels
outlier_df[clf_name + " Distance"] = np.round(y_decision_scores, 2)

cur_model = {"Package Name": package_name, "Model Name": clf_name, "Time Taken in Seconds": diff,
             "No.Of Data Points": n_points, "No.Of Outliers": n_outliers, "Percentage": predicted_outlier_percent}
result_df = result_df.append(cur_model, ignore_index=True)

##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################

clf_name = "COPOD"
clf = COPOD(contamination=outlier_percent)

#for clf_name, clf in model_dict.items():
# Starting Timer
start_time = timeit.default_timer()
clf.fit(df_train)
y_decision_scores = clf.decision_scores_
# Stopping Timer
stop_time = timeit.default_timer()

makeDataFrameNonIQR(clf_name, clf.labels_)

diff = np.round(stop_time - start_time, 2)
labels = clf.labels_
n_outliers = np.count_nonzero(labels)
predicted_outlier_percent = np.round((n_outliers / n_points) * 100, 0)

fig = plt.figure()
plt.title(clf_name)
plt.hist(labels)
plt.show()

outlier_df[clf_name + " Outliers"] = labels
outlier_df[clf_name + " Distance"] = np.round(y_decision_scores, 2)

cur_model = {"Package Name": package_name, "Model Name": clf_name, "Time Taken in Seconds": diff,
             "No.Of Data Points": n_points, "No.Of Outliers": n_outliers, "Percentage": predicted_outlier_percent}
result_df = result_df.append(cur_model, ignore_index=True)

##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################

clf_name = "PCA"
clf = PCA(contamination=outlier_percent)

#for clf_name, clf in model_dict.items():
# Starting Timer
start_time = timeit.default_timer()
clf.fit(df_train)
y_decision_scores = clf.decision_scores_
# Stopping Timer
stop_time = timeit.default_timer()

makeDataFrameNonIQR(clf_name, clf.labels_)

diff = np.round(stop_time - start_time, 2)
labels = clf.labels_
n_outliers = np.count_nonzero(labels)
predicted_outlier_percent = np.round((n_outliers / n_points) * 100, 0)

fig = plt.figure()
plt.title(clf_name)
plt.hist(labels)
plt.show()

outlier_df[clf_name + " Outliers"] = labels
outlier_df[clf_name + " Distance"] = np.round(y_decision_scores, 2)

cur_model = {"Package Name": package_name, "Model Name": clf_name, "Time Taken in Seconds": diff,
             "No.Of Data Points": n_points, "No.Of Outliers": n_outliers, "Percentage": predicted_outlier_percent}
result_df = result_df.append(cur_model, ignore_index=True)

##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################

clf_name = "HBOS"
clf = HBOS(contamination=outlier_percent)

#for clf_name, clf in model_dict.items():
# Starting Timer
start_time = timeit.default_timer()
clf.fit(df_train)
y_decision_scores = clf.decision_scores_
# Stopping Timer
stop_time = timeit.default_timer()

makeDataFrameNonIQR(clf_name, clf.labels_)

diff = np.round(stop_time - start_time, 2)
labels = clf.labels_
n_outliers = np.count_nonzero(labels)
predicted_outlier_percent = np.round((n_outliers / n_points) * 100, 0)

fig = plt.figure()
plt.title(clf_name)
plt.hist(labels)
plt.show()

outlier_df[clf_name + " Outliers"] = labels
outlier_df[clf_name + " Distance"] = np.round(y_decision_scores, 2)

cur_model = {"Package Name": package_name, "Model Name": clf_name, "Time Taken in Seconds": diff,
             "No.Of Data Points": n_points, "No.Of Outliers": n_outliers, "Percentage": predicted_outlier_percent}
result_df = result_df.append(cur_model, ignore_index=True)

##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################

clf_name = "ECOD"
clf = ECOD(contamination=outlier_percent)

#for clf_name, clf in model_dict.items():
# Starting Timer
start_time = timeit.default_timer()
clf.fit(df_train)
y_decision_scores = clf.decision_scores_
# Stopping Timer
stop_time = timeit.default_timer()

makeDataFrameNonIQR(clf_name, clf.labels_)

diff = np.round(stop_time - start_time, 2)
labels = clf.labels_
n_outliers = np.count_nonzero(labels)
predicted_outlier_percent = np.round((n_outliers / n_points) * 100, 0)

fig = plt.figure()
plt.title(clf_name)
plt.hist(labels)
plt.show()

outlier_df[clf_name + " Outliers"] = labels
outlier_df[clf_name + " Distance"] = np.round(y_decision_scores, 2)

cur_model = {"Package Name": package_name, "Model Name": clf_name, "Time Taken in Seconds": diff,
             "No.Of Data Points": n_points, "No.Of Outliers": n_outliers, "Percentage": predicted_outlier_percent}
result_df = result_df.append(cur_model, ignore_index=True)

##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################

from matplotlib.patches import Polygon

outlier_flag_col = []
for i in outlier_df.columns:
    if "Outliers" in i:
        outlier_flag_col.append(i)

print(outlier_flag_col)

outlier_df["nr_of_match"] = outlier_df[outlier_flag_col].sum(axis=1)
outlier_df["nr_of_match"].max()

outlier_df_filt = outlier_df[outlier_df["nr_of_match"]>7]
outlier_df_filt.shape

outlier_df_filt.to_excel(r"C:\Users\gpocs001\Desktop\ELTE_POSTGRAD\Project_work\test.xlsx")







def create_boxplot_chart(row_id, boxplot_cols):
    hcpcs = df[df.index == row_id]["HCPCS Description"].values[0]
    prov_type = df[df.index == row_id]["Provider Type"].values[0]

    df[df["HCPCS Description"]==hcpcs]
    df[df["Provider Type"] == prov_type]

    data = df[df["Provider Type"] == prov_type][boxplot_cols]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.canvas.manager.set_window_title('Selected line item')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    bp = ax1.boxplot(data, notch=False, sym='+', vert=True, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')

    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)

    ax1.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title='Relative position of the outlier to similar type of services',
        ylabel='Value',
    )


    box_colors = ['aqua', 'royalblue', 'blue', 'darkorchid', 'purple', 'deeppink']
    num_boxes = len(data.columns)
    medians = np.empty(num_boxes)

    for box_id, col_names in enumerate(data.columns):
        box = bp['boxes'][box_id]
        box_x = []
        box_y = []
        for j in range(5):
            box_x.append(box.get_xdata()[j])
            box_y.append(box.get_ydata()[j])
        box_coords = np.column_stack([box_x, box_y])
        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[box_id]))

        print(box_id)
        print(data[data.index == row_id][col_names].values[0])


        ax1.plot(box_id+1, data[data.index == row_id][col_names].values[0], color='w', marker='*', markeredgecolor='red', markersize=12)

    ax1.set_xlim(0.5, num_boxes + 0.5)
    top = np.percentile(data[data.idxmax(axis=1).values[0]],95)
    bottom = -5
    ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(boxplot_cols, rotation=0, fontsize=8, wrap=True)

    row_id_values = data[data.index==row_id].values[0].tolist()
    pos = np.arange(num_boxes) + 1
    upper_labels = [str(round(s, 2)) for s in row_id_values]

    for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
        ax1.text(pos[tick], .95, upper_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='x-small',
                 weight='bold', color='black')
    plt.show()


row_id = 95243

boxplot_cols = ['Average Medicare Allowed Amount', 'Average Submitted Charge Amount',
       'Average Medicare Payment Amount',
       'Average Medicare Standardized Amount']

boxplot_cols2 = ['Number of Services', 'Number of Medicare Beneficiaries',
       'Number of Distinct Medicare Beneficiary/Per Day Services',]

create_boxplot_chart(95243, boxplot_cols)
create_boxplot_chart(95243, boxplot_cols2)


##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################


def create_boxplot_chart2(outlier_list, boxplot_cols):

    data = df[boxplot_cols]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.canvas.manager.set_window_title('Selected line item')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    bp = ax1.boxplot(data, notch=False, sym='+', vert=True, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')

    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)

    ax1.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title='Relative position of the outlier to similar type of services',
        ylabel='Value',
    )


    box_colors = ['aqua', 'royalblue', 'blue', 'darkorchid', 'purple', 'deeppink']
    num_boxes = len(data.columns)
    medians = np.empty(num_boxes)

    for box_id, col_names in enumerate(data.columns):
        box = bp['boxes'][box_id]
        box_x = []
        box_y = []
        for j in range(5):
            box_x.append(box.get_xdata()[j])
            box_y.append(box.get_ydata()[j])
        box_coords = np.column_stack([box_x, box_y])
        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[box_id]))

        print(box_id)
        print(data[data.index == row_id][col_names].values[0])

        for o in outlier_list:
            ax1.plot(box_id+1, data[data.index == o][col_names].values[0], color='w', marker='*', markeredgecolor='red', markersize=12)

    ax1.set_xlim(0.5, num_boxes + 0.5)
    top = np.percentile(data[data.idxmax(axis=1).values[0]],95)
    bottom = -5
    ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(boxplot_cols, rotation=0, fontsize=8, wrap=True)


'''    row_id_values = data[data.index==row_id].values[0].tolist()
    pos = np.arange(num_boxes) + 1
    upper_labels = [str(round(s, 2)) for s in row_id_values]

    for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
        ax1.text(pos[tick], .95, upper_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='x-small',
                 weight='bold', color='black')'''
    plt.show()


outlier_list = outlier_df_filt.index.tolist()
create_boxplot_chart2(outlier_list, boxplot_cols)
