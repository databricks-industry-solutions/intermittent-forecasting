# Databricks notebook source
# MAGIC %md The purpose of this notebook is prepare the dataset which will be used throughout the remainder of the Nixtla intermittent demand forecasting solution accelerator. You may also find this accelerator notebook at https://github.com/databricks-industry-solutions/intermittent-forecasting.git.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC 
# MAGIC In our examination of the Nixtla forecasting libraries, we will make use of the [M5 competition](https://mofc.unic.ac.cy/m5-competition/) dataset provided by Walmart. This dataset is interesting for its scale but also the fact that it features many timeseries with infrequent occurrences.  Such timeseries are common in retail scenarios and are difficult for traditional timeseries forecasting techniques to address. But before we can make use of this dataset, we need to download it and re-align its structure to what is expected by most of the timeseries libraries available today, including Nixtla. 

# COMMAND ----------

# DBTITLE 1,Get Config Values
# MAGIC %run "./00_Intro & Config"

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import os
import pyspark.sql.functions as fn

# COMMAND ----------

# MAGIC %md ##Step 1: Download the Data Files
# MAGIC 
# MAGIC The original M5 data files are available for download the competition's [Kaggle page](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data), but the full set, including files used in the final evaluation of the competitors, has been made accessible by the folks at Nixtla for easy download.  
# MAGIC 
# MAGIC We'll download a Zip file containing the M5 data from the Nixtla repository and extract the file's contents to a mount point configured for our workspace.  Please see the instructions in the *NX 00* notebook to setup an appropriate mount point in your environment:

# COMMAND ----------

# DBTITLE 1,Set Downloads Path
downloads_path = f"{config['mount path']}/downloads"

# make variable available to shell script
os.environ['downloads_path'] = downloads_path

# COMMAND ----------

# DBTITLE 1,Download the M5 Data Files
# MAGIC %sh 
# MAGIC 
# MAGIC # set downloads path
# MAGIC downloads_path="/dbfs${downloads_path}"
# MAGIC 
# MAGIC # reset and move to downloads path
# MAGIC rm -rf $downloads_path
# MAGIC mkdir -p $downloads_path
# MAGIC cd $downloads_path
# MAGIC 
# MAGIC # download the m5 data
# MAGIC wget -q -O m5.zip https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip
# MAGIC 
# MAGIC # unzip the m5 data
# MAGIC unzip -q m5.zip
# MAGIC rm m5.zip
# MAGIC 
# MAGIC # display the folder contents
# MAGIC pwd
# MAGIC stat -c "%n (%s)" * | ( TAB=$'    ' ; sed "s/^/$TAB/" )

# COMMAND ----------

# MAGIC %md ##Step 2: Read the Data Files
# MAGIC 
# MAGIC The files downloaded by NeuralForecast are as follows:
# MAGIC </p>
# MAGIC 
# MAGIC * **calendar.csv** - contains information about the dates on which the products are sold
# MAGIC * **sales_train_evaluation.csv** -  contains the historical daily unit sales data per product and store \[days 1 - 1941\]
# MAGIC * **sales_test_evaluation.csv** -  contains the historical daily unit sales data per product and store \[days 1942 - 1969\] 
# MAGIC * **sales_train_validation.csv** - contains the historical daily unit sales data per product and store \[days 1 - 1913\]
# MAGIC * **sales_test_validation.csv** - contains the historical daily unit sales data per product and store \[days 1914 - 1941\]
# MAGIC * **sell_prices.csv** - contains information about the price of the products sold per store and week
# MAGIC * **weights_evaluation.csv** - contains the weights used for computing WRMSSE for the validation phase of the competition
# MAGIC * **weights_validation.csv** - contains the weights used for computing WRMSSE for the evaluation phase of the competition
# MAGIC 
# MAGIC Each CSV file can be read to a Spark dataframe leveraging the Spark CSV DataFrameReader's ability to infer a schema as follows:
# MAGIC 
# MAGIC **NOTE** We are using Python's ability to generate a variable from a string name through the *vars* library.  The end result of the following logic will be one dataframe variable for each CSV file named according to that file's base name. 

# COMMAND ----------

# DBTITLE 1,Read Data Files to Dataframes
# identify base name of csv files to process
timeseries_file_names = ['sales_test_evaluation','sales_train_evaluation','sales_test_validation', 'sales_train_validation']
other_file_names = ['calendar', 'weights_evaluation','sell_prices']
csv_file_names = timeseries_file_names + other_file_names

# instantiate vars object
my_vars = vars()

# for each csv file
for csv in csv_file_names:
  
  # create a dataframe with same name as csv base file name
  my_vars[csv] = (
    spark
      .read
      .csv(
        path=downloads_path + '/' + csv + '.csv', # path to csv
        header=True,  # has header
        inferSchema=True # infer schema from data
        )
      )

# COMMAND ----------

# MAGIC %md ##Step 3: Restructure the Datasets
# MAGIC The *sales_train_evaluation* dataframe illustrates the structural challenge with the timeseries files that we now need to address. Notice that values for each day in the dataset are captured in a day-specific field:

# COMMAND ----------

# DBTITLE 1,Examine Sample Data Structure
display(sales_train_evaluation)

# COMMAND ----------

# MAGIC %md In order to pass this data to most forecasting libraries, we need to unpivot the day-specific fields so that a single field captures the date (or day number) and another field captures the value for that day.  In addition, we will want to create a unique identifier for each row by combining the item and store ids in order to align the dataset with the expectations of the Nixtla models:

# COMMAND ----------

# DBTITLE 1,Unpivot Relevant Dataframes
# for each timeseries dataframe
for csv in csv_file_names:
  
  # combine item and store to form a unique identifier for each row
  if ('item_id' in my_vars[csv].columns) and ('store_id' in my_vars[csv].columns):
    my_vars[csv] = (
        my_vars[csv]
          .withColumn('unique_id', fn.expr("concat(item_id,'_',store_id)"))
          .drop('item_id', 'store_id')
      )
     
  # identify the date fields in the dataframe (these will start with 'd_')
  date_columns = [c for c in my_vars[csv].columns if c[:2]=='d_']

  # if date columns present ...
  if len(date_columns) > 0:
    # assemble a stack expression mapping a given field to a value in a column
    stack_expression = ','.join([f"'{d}', {d}" for d in date_columns])

    # perform the unpivot operation, mapping each field to a value in a 
    # column called ds_id and its value to a column called y
    my_vars[csv] = (
      my_vars[csv]
        .selectExpr(
          'unique_id',
          f"stack({len(date_columns)}, {stack_expression}) as (ds_id, y)" # this is the unpivot expression
          )
      )

# COMMAND ----------

# DBTITLE 1,Examine Unpivoted Sample Data Structure
display(sales_train_evaluation)

# COMMAND ----------

# MAGIC %md The day numbers now present in our ds_id field are valued from 1 to 1969.  These correspond to the ordinal values associated with the dates in the calendar dataset. To translate these numbers to actual dates, we first need to generate the day number for each date in the calendar dataset in the format currently used by the timeseries datasets:

# COMMAND ----------

# DBTITLE 1,Generate Day Number for Dates
calendar = (
  calendar
    .withColumn('ds_id', fn.expr('row_number() over(order by date)')) # generate ordinal number
    .withColumn('ds_id', fn.expr("concat('d_', ds_id)")) # format day numbers using 'd_' prefix
    .select('ds_id', *calendar.columns)
    )

display(calendar)

# COMMAND ----------

# MAGIC %md We can now convert the day numbers in our timeseries dataframes to actual dates:

# COMMAND ----------

# DBTITLE 1,Convert Day Numbers to Dates
# for each timeseries dataframe
for csv in timeseries_file_names:
  
  # if dataframe has a ds_id field
  if 'ds_id' in my_vars[csv].columns:
  
    # join with calendar to get date value
    my_vars[csv] = (
        my_vars[csv]
          .join(calendar, on='ds_id')
          .selectExpr(
            'unique_id',
            'date as ds',
            'y'
            )
        )

# COMMAND ----------

# MAGIC %md We can now examine the revised structure of these datasets:

# COMMAND ----------

# DBTITLE 1,Examine Restructured Sample Data
display(sales_train_evaluation)

# COMMAND ----------

# MAGIC %md ##Step 4: Persist the Datasets
# MAGIC 
# MAGIC With our datasets properly structured, we can persist them as follows:

# COMMAND ----------

# DBTITLE 1,Reinitiate the Database
_ = spark.sql('DROP DATABASE IF EXISTS {0} CASCADE'.format(config['database name']))
_ = spark.sql('CREATE DATABASE {0}'.format(config['database name']))

# COMMAND ----------

# DBTITLE 1,Persist Data for Reuse
# for each dataset
for csv in csv_file_names:
  
  # write it to a table named for the base file name 
  _ = (
    my_vars[csv]
      .write
      .format('delta')
      .mode('overwrite')
      .option('overwriteSchema','true')
      .saveAsTable(csv)
    )

# COMMAND ----------

# DBTITLE 1,Show Tables in Database
# MAGIC %sql show tables;

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | statsforecast| Lightning fast forecasting with statistical and econometric models | Apache 2.0 | https://github.com/Nixtla/statsforecast |
