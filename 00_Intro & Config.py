# Databricks notebook source
# MAGIC %md The purpose of this notebook is provide an introduction to the Nixtla intermittent demand forecasting solution accelerator and to provide access to the configuration values that support it. You may also find this accelerator notebook at https://github.com/databricks-industry-solutions/intermittent-forecasting.git.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC 
# MAGIC In order to capture localized patterns of demand, more and more organizations are generating forecasts at the store-item level.  Working at this level of granularity not only explodes the number of forecasts we need to produce, it frequently presents us with timeseries containing numerous periods of zero-unit demand.  Commonly used models are simply inappropriate for such timeseries, requiring us to turn to specialized algorithms built for intermittent (sparse) timeseries.
# MAGIC 
# MAGIC In this solution accelerator, we'll leverage historical daily data for a large number of store-item combinations within which numerous periods of zero-unit sales are observed in order to explore how we might overcome this challenge.  We will make use of the [Nixtla statsforecast library](https://github.com/Nixtla/statsforecast) which has support for numerous models specialized for intermittent demand forecasting and which has a built-in backend for scaling the training of large numbers of models across a Databricks cluster.

# COMMAND ----------

# MAGIC %md ##Configuration
# MAGIC 
# MAGIC The following values are used to set consistent configurations across the notebooks that make up this accelerator:

# COMMAND ----------

# DBTITLE 1,Initialize Configuration Variable
if 'config' not in locals():
  config = {}

# COMMAND ----------

# DBTITLE 1,Initialize Database
# set database name
config['database name'] = 'nixtla'

# create database to house mappings
_ = spark.sql('CREATE DATABASE IF NOT EXISTS {0}'.format(config['database name']))

# set database as default for queries
_ = spark.catalog.setCurrentDatabase(config['database name'])

# COMMAND ----------

# MAGIC %md The following setting identifies the path to house the sample data. Here we use a `/tmp/...` path in DBFS to minimize dependency. We recommend using a `/mnt/...` path or one that directly connects to your cloud storage for production usage. To learn more about mount points, please review [this document](https://docs.databricks.com/dbfs/mounts.html).  If you would like to use a mount point or a different path, please update the variable below with the appropriate path:

# COMMAND ----------

# DBTITLE 1,Storage Settings
# path where files are stored

config['mount path'] = '/tmp/nixtla'

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | statsforecast| Lightning fast forecasting with statistical and econometric models | Apache 2.0 | https://github.com/Nixtla/statsforecast |
