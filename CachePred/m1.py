# Databricks notebook source
import dask.dataframe as dd
import pandas as pd
import plotly.express as px
from plotly.offline import plot
import numpy as np

# COMMAND ----------

s3_df = dd.read_csv("s3://search-curated-sec/ChangeDateTime/withoutMulti/BOM-PAT/6E/complete/*.csv", names=["sector", "fno", "depdate", "deptime", "bookdate", "booktime", "dur", "htg", "fare","changeDateTime_1", "changeDateTime_2", "changeDateTime_3", "changeDateTime_4", "changeDateTime_5", "changeDateTime_6", "changeDateTime_7", "changeDateTime_8", "changeDateTime_9", "changeDateTime_10",  "posNeg"], dtype={'changeDateTime_1': 'str', 'changeDateTime_2': 'str', 'changeDateTime_3': 'str', 'changeDateTime_4': 'str', 'changeDateTime_5': 'str', 'changeDateTime_6': 'str', 'changeDateTime_7': 'str', 'changeDateTime_8': 'str', 'changeDateTime_9': 'str', 'changeDateTime_10': 'str'})

# COMMAND ----------

pd.set_option('display.max_columns', 80)

# COMMAND ----------

s3_df.isna().sum().compute()

# COMMAND ----------

s3_df = s3_df.dropna(subset=["changeDateTime_4"]).reset_index()

# COMMAND ----------

#s3_df["depDateTime"] = s3_df.depdate * 10000 + s3_df.deptime

# COMMAND ----------

s3_df.describe().compute()

# COMMAND ----------

# computing ttl
#s3_df["ttl"] = s3_df
s3_df["changeDateTime_4"].dtype

# COMMAND ----------

t = pd.DataFrame()
t["time"] = pd.to_datetime(s3_df.changeDateTime_4, format="%Y%m%d%H")- pd.to_datetime(s3_df.booktime, format="%Y%m%d%H%M")
#s3_df["time_min"] = s3_df['time']/np.timedelta64(1,'m') 
#s3_df["time_hr"] = s3_df['time']//np.timedelta64(1,'h')
#s3_df = s3_df.drop(column=[time])

# COMMAND ----------

t["time"].head()

# COMMAND ----------

s3_df["time"] = np.nan

# COMMAND ----------

df = s3_df.compute()

# COMMAND ----------

df["time"] = t["time"]

# COMMAND ----------

df["time_min"] = df['time']/np.timedelta64(1,'m') 
df["time_hr"] = df['time']//np.timedelta64(1,'h')
df.drop(columns=["time"], inplace=True)

# COMMAND ----------

# remove all changeDateTime except changeDateTime4
remove = []
for i in range(1,11):
  remove.append("changeDateTime_{}".format(i))
df.drop(columns=remove+["posNeg"], inplace=True)
print(df)

# COMMAND ----------

df.describe()

# COMMAND ----------

import pickle as pkl

# Save INS and OOS data
data_path_ins = "/dbfs/sud/data/m1"
with open(data_path_ins, 'wb') as fi:
  pkl.dump(df,fi)
  fi.close()


# COMMAND ----------

# MAGIC %fs ls /sud/data/

# COMMAND ----------

import pickle as pkl
with open("/dbfs/sud/data/m1", "rb") as f:
  df = pkl.load(f)
  f.close()
print(df.head(), type(df))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic DNN

# COMMAND ----------

import tensorflow as tf
from tensorflow import keras

# COMMAND ----------

df.head()

# COMMAND ----------

df.sort_values("booktime").reset_index(drop=True, inplace=True)

# COMMAND ----------

df.head()

# COMMAND ----------

#Y = df[["time_min", "time_hr"]]
Y = df[["time_hr"]]

# COMMAND ----------

Y.head()

# COMMAND ----------

df.columns

# COMMAND ----------

X = df[['fno', 'depdate', 'deptime', 'bookdate', 'booktime',
       'dur', 'htg', 'fare']]

# COMMAND ----------

print(X.shape, Y.shape)

# COMMAND ----------

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
X["fno"] =le.fit_transform(X.fno)


# COMMAND ----------

dfH = pd.read_csv("s3://search-curated-sec/HolidayCalendar.csv")

# COMMAND ----------

dfH["date"] =dfH.searchdate.str.replace("-","").astype(int)
dfH.drop(columns=["searchdate"], inplace=True)

# COMMAND ----------

dfH.head()

# COMMAND ----------

X = pd.merge(X, dfH, how='left', left_on = 'depdate', right_on = 'date')

# COMMAND ----------

X.drop(columns=["date"], inplace=True)

# COMMAND ----------

X.head()

# COMMAND ----------

X.isnull().any()

# COMMAND ----------

X.fillna(0, inplace=True)

# COMMAND ----------

X.loc[X.depdate==20180815]

# COMMAND ----------

Y.head()

# COMMAND ----------

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=101)

# COMMAND ----------

print("shapes", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# COMMAND ----------

scaler = preprocessing.MinMaxScaler()
scaler.fit(X_train)


# COMMAND ----------

X_train =pd.DataFrame(scaler.transform(X_train),
                          columns = X_train.columns,
                          index = X_train.index)

# COMMAND ----------

X_test =pd.DataFrame(scaler.transform(X_test),
                          columns = X_test.columns,
                          index = X_test.index)

# COMMAND ----------

X_train.info()

# COMMAND ----------

X_test.describe()

# COMMAND ----------

len(X_train.fno.unique())

# COMMAND ----------

dnn = keras.Sequential()

# COMMAND ----------

dnn.add(keras.layers.Dense(X_train.shape[1], input_shape= (X_train.shape[1],), activation="relu"))
dnn.output_shape

# COMMAND ----------

dnn.add(keras.layers.Dense(64, activation="relu"))
dnn.output_shape

# COMMAND ----------

dnn.add(keras.layers.Dropout(0.2))
dnn.output_shape

# COMMAND ----------

dnn.add(keras.layers.Dense(128, activation="relu"))
dnn.output_shape

# COMMAND ----------

dnn.add(keras.layers.Dense(y_train.shape[1], activation="relu"))
dnn.output_shape

# COMMAND ----------

# logdir = "logs/sud/"
# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# COMMAND ----------

def lr_schedule(epoch):
  """
  Returns a custom learning rate that decreases as epochs progress.
  """
  learning_rate = 0.2
  if epoch > 10:
    learning_rate = 0.02
  if epoch > 20:
    learning_rate = 0.01
  if epoch > 50:
    learning_rate = 0.005
  return learning_rate

# COMMAND ----------

lr_callback = keras.callbacks.LearningRateScheduler(lr_schedule)

# COMMAND ----------

dnn.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])
#dnn.compile(loss='mse', # keras.losses.mean_squared_error
#            optimizer=keras.optimizers.SGD(lr=0.2),
#            metrics=["accuracy"])

# COMMAND ----------

dnn.fit(x=X_train, 
        y= y_train, 
        validation_data=(X_test, y_test), 
        callbacks=[lr_callback],
        batch_size=520,
        epochs=10)

# COMMAND ----------

dnn.evaluate(x=X_test, y=y_test, batch_size=300)

# COMMAND ----------

print(dnn.predict(x=X_train[:1]),"\n true", y_train[:1] )
#X_test[:1]

# COMMAND ----------

help(keras.Sequential.fit)

# COMMAND ----------

