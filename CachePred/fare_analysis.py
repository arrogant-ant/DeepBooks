# Databricks notebook source
# MAGIC %md
# MAGIC ###**Tries to see factors that affecting fares**

# COMMAND ----------

import s3fs
import s3transfer
import pandas as pd
import plotly.express as px
from plotly.offline import plot

# COMMAND ----------

import dask.dataframe as dd

# COMMAND ----------

#pd.set_option('display.max_columns', 20)

# COMMAND ----------

                     s3://search-curated-sec/changeWithPosNeg/correct_posNeg/BOM-PAT/part-00000-9839df01-0985-4cc1-9698-592ad0d43d12-c000.csv   
s3_df = dd.read_csv("s3://search-curated-sec/changeWithPosNeg/correct_posNeg/BOM-PAT/6E/complete/part-00000-36beedc3-50e1-4da0-8d33-5040c7025698-c000.csv", names=['fno', 'fare', 'depdate', 'deptime', 'searchdate', 'searchdatetime', 'htg', 'uid', 'changeDateTime1', 'changeDateTime2', 'changeDateTime3', 'changeDateTime4', 'changeDateTime5', 'changeDateTime6', 'changeDateTime7', 'changeDateTime8', 'changeDateTime9', 'changeDateTime10', 'magnitude'], dtype={'changeDateTime1': 'str', 'changeDateTime2': 'str', 'changeDateTime3': 'str', 'changeDateTime4': 'str', 'changeDateTime5': 'str', 'changeDateTime6': 'str', 'changeDateTime7': 'str', 'changeDateTime8': 'str', 'changeDateTime9': 'str', 'changeDateTime10': 'str'})

s3_df2 = dd.read_csv("s3://search-curated-sec/changeWithPosNeg/correct_posNeg/BOM-PAT/6E/complete/p*.csv", names=['fno', 'fare', 'depdate', 'deptime', 'searchdate', 'searchdatetime', 'htg', 'uid', 'changeDateTime1', 'changeDateTime2', 'changeDateTime3', 'changeDateTime4', 'changeDateTime5', 'changeDateTime6', 'changeDateTime7', 'changeDateTime8', 'changeDateTime9', 'changeDateTime10', 'magnitude'], dtype={'changeDateTime1': 'str', 'changeDateTime2': 'str', 'changeDateTime3': 'str', 'changeDateTime4': 'str', 'changeDateTime5': 'str', 'changeDateTime6': 'str', 'changeDateTime7': 'str', 'changeDateTime8': 'str', 'changeDateTime9': 'str', 'changeDateTime10': 'str'})

s3_df3 = dd.read_csv("s3://search-curated-sec/changeWithPosNeg/correct_posNeg/BOM-PAT/6E/complete/*.csv")

# COMMAND ----------

s3_df.compute().shape

# COMMAND ----------

s3_df.tail()

# COMMAND ----------

s3_df2.tail()

# COMMAND ----------

s3_df2.compute().shape

# COMMAND ----------

s3_df3.compute().shape

# COMMAND ----------

sample = s3_df.loc[s3_df["searchdate"] > 20190515][['fno', 'fare', 'depdate', 'deptime', 'htg']]
sample["depdate"] = sample["depdate"].astype(str)
sample.info()

# COMMAND ----------

del s3_df

# COMMAND ----------

#sample_grp = sample.groupby(["fno","depdate", "deptime", "htg"])
sample_grp = s3_df.groupby(["fno","depdate", "deptime", "htg"]).fare.mean().reset_index()
sample_grp["depdate"] = sample_grp["depdate"].astype(str)

# COMMAND ----------

sample_grp.head()

# COMMAND ----------

sample_grp.tail()

# COMMAND ----------

sample.drop_duplicates(inplace=True)
sample.reset_index(inplace=True, drop=True)
sample.head()

# COMMAND ----------

sample_grp.groupby(["fno"]).head()

# COMMAND ----------

# MAGIC %md
# MAGIC ###fare with htg 
# MAGIC *(flight-wise)*

# COMMAND ----------

fno_df = sample_grp.loc[sample_grp["fno"].isin(["6E298|6E581", "6E154|6E494", "6E168|6E2039"])]
fno_fig = plot(px.scatter(fno_df.compute().sort_values("htg"), x="htg", y="fare",color="fno"), output_type="div")
displayHTML(fno_fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Fixed dept date

# COMMAND ----------

help(px.scatter)

# COMMAND ----------

dep_df = sample_grp.loc[sample_grp["depdate"].isin(["20190702", "20190701", "20190815", "20190704", "20190705"])]\
  .loc[sample_grp["fno"].isin(["6E298|6E581", "6E154|6E494", "6E787"])]

dep_fig = plot(px.scatter(dep_df.compute(), x="htg", y="fare",color="depdate", trendline="lowess",facet_col= "depdate"), output_type="div")
displayHTML(dep_fig)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Everything against Everything

# COMMAND ----------

all_fig = plot(px.scatter_matrix(sample_grp.sample(10000).sort_values("htg"),dimensions=["depdate","fare","htg"], color="fno"), output_type = "div")
displayHTML(all_fig)

# COMMAND ----------

