# Databricks notebook source
# MAGIC %md The purpose of this notebook is generate store-item level forecasts as part of the Nixtla intermittent forecasting solution accelerator. You may also find this accelerator notebook at https://github.com/databricks-industry-solutions/intermittent-forecasting.git.

# COMMAND ----------

# MAGIC %md ## Introduction
# MAGIC 
# MAGIC In this notebook, we want to walk through the typical steps organizations employ in developing forecasts at the store-item level. The dataset we are using features data for products sold across several store locations at a daily level for a roughly 5 year period.  A key feature of this set is that many of the items have no sales on numerous days, a common characteristics of such data when observed at a low level of granularity.  In order to build forecasts at the store-item, we will need to employ techniques specifically designed to deal with intermittent (sporadic) values.

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install "statsforecast[fugue]==1.4.0"

# COMMAND ----------

# DBTITLE 1,Get Config Values
# MAGIC %run "./00_Intro & Config"

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import pyspark.sql.functions as fn

from statsforecast import StatsForecast
from statsforecast.distributed.fugue import FugueBackend
from statsforecast.models import *

import numpy as np
import pandas as pd

# COMMAND ----------

# MAGIC %md ##Step 1: Explore the Timeseries Data
# MAGIC 
# MAGIC To get started, let's examine the structure of the dataset.  As mentioned above, this data contains a wide number of products in different store locations.  We have combined the store and item identifiers into a single field named *unique_id* which we can break down as follows to identify not just the store and item but the state within which a store resides and the category and department associated with a given product:

# COMMAND ----------

# DBTITLE 1,Recreate Item-Store Parts
store_items = (
  spark
    .table('sales_train_evaluation')
    .select('unique_id')
    .distinct()
    .withColumn('parts', fn.expr("split(unique_id, '_')"))
    .withColumn('category', fn.expr('parts[0]'))
    .withColumn('department', fn.expr("array_join(slice(parts,1,2), '_')"))
    .withColumn('item', fn.expr("array_join(slice(parts,1,3), '_')"))
    .withColumn('state', fn.expr('parts[3]'))
    .withColumn('store', fn.expr("array_join(slice(parts,4,2), '_')"))
    .drop('parts')
  ).cache()

display(store_items.orderBy('unique_id'))

# COMMAND ----------

# DBTITLE 1,Determine Unique Values for Each Part
results = []

# for each column in the dataset
for c in store_items.columns:
  
  # count the distinct values
  uniq_count = store_items.select(c).distinct().count()
  
  # append counts to results set
  results += [(c, uniq_count)]
  
# display results
display(
  spark.createDataFrame(results, schema='column string, count int')
  )

# COMMAND ----------

# MAGIC %md From our analysis, we can see we have 3,049 unique products across 10 store locations which will require us to generate 30,490 store-item level forecasts.  The products are aligned across 3 categories and 7 departments, and the stores are found in 3 US states.
# MAGIC 
# MAGIC Examining one of the timeseries in the dataset, *i.e.* *FOODS_1_001_TX_1* or product 1 in department 1 within the foods category residing in store 1 in the state of Texas, we can see an example of the intermittent sales mentioned above.  While we are only observing one of the store-item combinations with this query, this pattern of days with zero sales is frequently observed across all store-item combinations in this dataset:

# COMMAND ----------

# DBTITLE 1,Show Timeseries Values for One Store-Item Combination
# MAGIC %sql
# MAGIC 
# MAGIC SELECT ds, y
# MAGIC FROM sales_train_evaluation
# MAGIC WHERE unique_id='FOODS_1_001_TX_1'
# MAGIC ORDER BY ds
# MAGIC LIMIT 1000

# COMMAND ----------

# MAGIC %md Using a visualization, it can be difficult to appreciate the number of missing observations in the dataset.  Here we calculate the number of dates for which a zero-units sales value is observed for each store-item combination and calculate a ratio of zero-value observations to total observations.  For the timeseries above, 67% of the observations are zero-values which is pretty close to the average of 68% across all timeseries in the dataset: 

# COMMAND ----------

# DBTITLE 1,Calculate Ratio of Zero-Value Observations
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   unique_id,
# MAGIC   COUNT(*) as obs,
# MAGIC   COUNT_IF(y=0) as zero_obs,
# MAGIC   FORMAT_NUMBER( COUNT_IF(y=0)/COUNT(*), '#.###') as zero_obs_ratio
# MAGIC FROM sales_train_evaluation
# MAGIC GROUP BY unique_id WITH ROLLUP
# MAGIC ORDER BY unique_id

# COMMAND ----------

# MAGIC %md ##Step 2: Evaluate Baseline Model
# MAGIC 
# MAGIC To establish a baseline for evaluating our forecasts, we will generate a simple, naive model.  We will do this for one product in one location to get oriented to how Nixtla generates forecasts before then scaling forecast generation across all 30K+ store-item combinations in the dataset:

# COMMAND ----------

# DBTITLE 1,Get History for One Store-Item Combination
y_hist = (
  spark
    .table('sales_train_evaluation')
    .filter(fn.expr("unique_id='FOODS_1_001_TX_1'")) # focus on one store-item combination
    .orderBy('ds') # sort on date
    .selectExpr('cast(y as float) as y') # extract historical values
  ).toPandas()['y'].values # extract historical values as a numpy array

# display historical values
y_hist

# COMMAND ----------

# DBTITLE 1,Generate Naïve Forecast (One Store-Item)
# instantiate model
model = Naive()

# fit model to historical data
model = model.fit(y_hist)

# generate 28 day forecast
y_pred = model.predict(h=28)

# display forecast results
y_pred

# COMMAND ----------

# MAGIC %md The [naive model](https://nixtla.github.io/statsforecast/models.html#naive) simply repeats the last value in the historical dataset as the forecasted value.  While not a particularly robust forecast, it will provide us a nice baseline against which our other models can be compared.
# MAGIC 
# MAGIC To evaluate our forecast, we need to grab the actual values for the 28 day period over which we are forecasting:

# COMMAND ----------

# DBTITLE 1,Get Actuals for Forecast Period (One Store-Item)
y_true = (
  spark
    .table('sales_test_evaluation')
    .filter(fn.expr("unique_id='FOODS_1_001_TX_1'"))
    .orderBy('ds')
    .selectExpr('cast(y as float) as y')
  ).toPandas()['y'].values

y_true

# COMMAND ----------

# MAGIC %md To evaluate our model, we can generate a set of commonly employed evaluation metrics:

# COMMAND ----------

# DBTITLE 1,Evaluate Naïve Forecast (One Store-Item)
# calculate forecast error
err = y_pred['mean'] - y_true

# calculate metrics
rmse = np.sqrt( np.mean( err**2 ) )
bias = np.mean( err )
smape = np.mean( np.abs(err) / ((y_pred['mean']+y_true)/2) )
mae = np.mean( np.abs(err) )
mase = np.mean( np.abs(err) / np.mean(np.abs(np.diff(y_true))) ) # non-seasonal mase


# display results
display(
  pd.DataFrame(
    [('RMSE', rmse), ('Bias', bias), ('sMAPE', smape), ('MAE', mae), ('MASE', mase)],
    columns=['Metric','Value']
    )
  )

# COMMAND ----------

# MAGIC %md With the mechanics of producing a single forecast under our belts, let's use the Nixtla Fugue backend to generate a naive forecast for all store-item combinations in our dataset.  To do this, we'll first grab all the historical data to a Spark dataframe:
# MAGIC 
# MAGIC **NOTE** The dataset should have three fields, one of which identifies the timeseries, another which provides the date and a final field which provides the observed value on that date.

# COMMAND ----------

# DBTITLE 1,Get Historical Data
y_hist = (
  spark
    .table('sales_train_evaluation')
  )

display(y_hist)

# COMMAND ----------

# MAGIC %md Next, we instantiate the Fugue backend and point it to the Spark engine in the Databricks cluster on which this notebook is running. The *use_pandas_udf* setting instructs the backend to scale by distributing the forecasting workload across the Spark cluster leveraging a pandas UDF pattern.  You can see how this pattern is manually applied in order to scale a fine-grained forecasting workload by reviewing the (Python) notebook associated with [this blog](https://www.databricks.com/blog/2021/04/06/fine-grained-time-series-forecasting-at-scale-with-facebook-prophet-and-apache-spark-updated-for-spark-3.html).  It's nice that this pattern is automated with a simple configuration setting: 

# COMMAND ----------

# DBTITLE 1,Instantiate Fugue Backend
backend = FugueBackend(spark, {'fugue.spark.use_pandas_udf':True})

# COMMAND ----------

# MAGIC %md Now we can fit all 30K+ models and generate forecasts with a simple call to the backend:
# MAGIC 
# MAGIC **NOTE** The amount of time it takes to complete a scaled-out forecast such as this depends on the number of processors available across the worker nodes in your cluster.  Scale up and down the number of worker nodes and the number of processors per node to lower or increase the amount of time it takes to complete a scaled-out forecast.

# COMMAND ----------

# DBTITLE 1,Generate Forecasts
# generate forecasts
sf = StatsForecast(
   models = [ Naive() ], # models to employ
   freq = 'D', # frequency per https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
   backend = backend # specify backend, if you want to run your pipeline locally, just remove it
)
y_pred = sf.forecast(
    df = y_hist, 
    h = 28
  ).cache()

# display results
display(
  y_pred.orderBy('unique_id','ds')
  )

# COMMAND ----------

# MAGIC %md To evaluate these forecasts, we'll again need to retrieve the actuals for the forecast period:

# COMMAND ----------

# DBTITLE 1,Get Actuals for Forecast Period
y_true = (
  spark
    .table('sales_test_evaluation')
  )

display(y_true)

# COMMAND ----------

# MAGIC %md We can then define a function to help us evaluate our forecasts.  Because we will eventually employ this logic to evaluate multiple forecasts for each timeseries, we'll write the function in a manner that dynamically inspects the forecast dataframe and applies the various evaluation metrics to each: 

# COMMAND ----------

# DBTITLE 1,Define Forecast Evaluation Function
def evaluate_forecasts(y_true, y_pred):
  
  # check that correct df was sent to arguments
  if 'y' not in y_true.columns:
    raise Exception("A column named 'y' was not found in the y_true dataframe. Please make sure you submitted your dataframe arguments in the correct order.")
  
  # determine which models were used to produce the forecasts
  # (simply identify columns that are not the unique id or date)
  models = [c for c in y_pred.columns if c not in ['unique_id','ds']]

  # join actuals and predictions and calculate some values needed from across the dataset
  eval = (
    y_true
      .join(y_pred, on=['unique_id','ds'])
      .withColumn('y_previous', fn.expr('LAG(y, 1) OVER(PARTITION BY unique_id ORDER BY ds)'))
      .withColumn('y_absdiff', fn.expr("case when y_previous is not null then ABS(y - y_previous) else 0 end")) # needed for mase
      .withColumn('obs_count', fn.expr("COUNT(*) OVER(PARTITION BY unique_id)")) # needed for mase
      .withColumn('avg_y_absdiff', fn.expr('SUM(y_absdiff) OVER(PARTITION BY unique_id)/(obs_count-1)')) # needed for mase
      .drop('y_absdiff','obs_count') # no longer needed for mase
    )

  # define evaluation metric logic for each model
  aggs = []
  for m  in models:
    err_col = f"{m}_err"
    eval = eval.withColumn(err_col, fn.expr(f"{m}-y")) # error calculation
    aggs += [fn.sqrt( fn.avg( fn.pow(err_col,2) ) ).alias(f"{m}_rmse")] #rmse
    aggs += [fn.avg( err_col ).alias(f"{m}_bias")] # bias
    aggs += [fn.avg( fn.expr(f"abs({err_col})/(({m}+y)/2)") ).alias(f"{m}_smape")] #smape
    aggs += [fn.avg( fn.abs(err_col) ).alias(f"{m}_mae")] # mae
    aggs += [fn.avg( fn.abs(err_col) / fn.col('avg_y_absdiff') ).alias(f"{m}_mase")] # mase, non-seasonal    
    
  results = (
    eval
      .groupBy('unique_id')
        .agg( *aggs )
    )  
  
  return results

# COMMAND ----------

# MAGIC %md The function above deserves a bit of an explanation. Upon receiving actuals and forecasts Spark dataframes, the two are joined on store-item and date, and for each forecast (model) column found in the forecasts dataframe, the error between the forecast and actuals is calculated.  
# MAGIC 
# MAGIC The routine then generates the definitions for each of the evaluation metrics to be derived for each forecast.  These definitions, all of which employ aggregations, are captured in a list.  When we then group our data by store-item, *e.g.* *unique_id*, these can be applied by submitting them to the aggregation method as a list to be unpacked (per the asterisk preceding the list argument).
# MAGIC 
# MAGIC With that, let's now take a look at the evaluation results:

# COMMAND ----------

# DBTITLE 1,Display Evaluation Metrics
evaluation_metrics = evaluate_forecasts(y_true, y_pred)

display( 
  evaluation_metrics.orderBy('unique_id') 
  )

# COMMAND ----------

# MAGIC %md In scenarios such as this when we are generating a large number of forecasts, we will occasionally want to summarize the metrics to get a sense for how each model performs on average.  As each store-item forecast is based on the same number of input dates, we can simply apply an average to each metric:

# COMMAND ----------

# DBTITLE 1,Define Evaluation Metrics Summarization Function
def summarize_evaluation_metrics(evaluation_results):
  
  # identify metrics in the results set
  metrics = [c for c in evaluation_results.columns if c not in ['unique_id']]
  
  # assemble metrics aggregation logic
  aggs = []
  stack_expression = []
  for m in metrics:
    stack_expression += [f"'{m}'", m]
    aggs += [fn.avg(f"{m}").alias(f"{m}")]
  
  results = (
    evaluation_results
      .groupBy() # aggregate evaluation metrics
        .agg(*aggs)
      .selectExpr( # unpivot results
        f"stack( {len(metrics)}, {','.join(stack_expression)}) as (name, value)"
        )
      .withColumn('name', fn.expr("split(name,'_')"))
      .withColumn('model', fn.expr("array_join(slice(name,-2, size(name)-1),'_')")) # extract model from metric name
      .withColumn('metric', fn.expr("slice(name,-1, 1)[0]")) # extract metric from metric name
      .groupBy('model') # pivot data to put model on rows and metrics across columns
        .pivot('metric')
          .agg(fn.first('value'))
      )
  
  return results

# COMMAND ----------

# DBTITLE 1,Display Evaluation Metrics Summary
# display aggregate metrics
display(
  summarize_evaluation_metrics(evaluation_metrics)
  )

# COMMAND ----------

# MAGIC %md ##Step 3: Evaluate Intermittent Models
# MAGIC 
# MAGIC Having established a baseline, we now turn our attention to constructing a proper forecast. The statsforecast library makes available a wide range of model types, [many of which](https://nixtla.github.io/statsforecast/models.html#sparse-or-intermittent) are equipped to handle the intermittent values issues associated with our dataset.  Some models we might consider are:
# MAGIC </p>
# MAGIC 
# MAGIC * [ADIDA](https://link.springer.com/article/10.1057/jors.2010.32) - Aggregate-Disaggregate Intermittent Demand Approach to forecasting
# MAGIC * [IMAPA](https://kourentzes.com/forecasting/2014/04/19/multiple-aggregation-prediction-algorithm-mapa/) - Intermittent Multiple Aggregation Prediction Algorithm
# MAGIC * [CrostonClassic](https://www.jstor.org/stable/3007885?origin=crossref) - Classic Croston forecasting
# MAGIC * CrostonOptimized - Croston forecasting with optimized smoothing parameter selection
# MAGIC * [TSB](https://www.sciencedirect.com/science/article/abs/pii/S0377221711004437) - Teunter-Syntetos-Babai modification of Croston’s method that replaces the inter-demand intervals with demand probability
# MAGIC </p>
# MAGIC 
# MAGIC To employ these different models, we simply specify a list of model types in our call to the Fugue backend.  Where we wish to specify configuration parameters, we can do so with each model instance.  In addition, we can specify a fallback model to provide values should we encounter a problem with any one of these models.
# MAGIC 
# MAGIC **NOTE** The statsforecast library comes equipped with many more [models](](https://nixtla.github.io/statsforecast/models.html) than the ones enumerated here, some of which may be appropriate in other scenarios.
# MAGIC 
# MAGIC As we specify different models to try, its important to keep in mind that each model will be required to generate 30K+ forecasts.  During the development cycle, we may wish to explore a wide variety of models but as we move towards a production deployment which will run frequently, often within tight processing windows, we will typically narrow our focus to the most promising algorithms:

# COMMAND ----------

# DBTITLE 1,Generate Forecasts using Specialized Models
sf = StatsForecast(
    models=[ 
      Naive(), # include Naive to make the comparison easier
      CrostonClassic(),
      IMAPA(),
      ADIDA(), # you can include more models such as AutoARIMA
     ], 
    fallback_model=Naive(),
    freq='D', 
    backend=backend
)
y_pred = sf.forecast(
    df=y_hist,
    h=28
  )

display(y_pred.orderBy('unique_id','ds'))

# COMMAND ----------

# MAGIC %md As before, we can calculate various evaluation metrics using our function:

# COMMAND ----------

# DBTITLE 1,Evaluate Individual Forecasts
evaluation_metrics = evaluate_forecasts(y_true, y_pred)

display(evaluation_metrics)

# COMMAND ----------

# DBTITLE 1,Evaluate Forecasts in Summary
summary_results = summarize_evaluation_metrics(evaluation_metrics)

display(summary_results)

# COMMAND ----------

# MAGIC %md ##Step 4: Perform Cross Validation
# MAGIC 
# MAGIC In previous steps, we've taken all our historical data and used it to predict a fixed future period.  In some ways, this mirrors a traditional train-test split with our historical data serving as the training set and our 28-day actuals serving as the testing set.  Much like in a traditional model training scenario, we want to make sure our model performs well across a range of inputs and isn't just performing well in the presence of a lucky split.  A cross-validation is often used for this where we shuffle our data across a number of folds and perform repeated evaluations where one fold serves as the test set and the remaining is used for training until each fold has an opportunity to serve as the test set.
# MAGIC 
# MAGIC With timeseries data, we can't just shuffle our records the same way, so instead, we often implement cross-validation by defining a sliding window across the historical data and predict the period following it. This form of cross-validation allows us to arrive at a better estimation of our model's predictive abilities across a wider range of instances while also keeping the data in the training set contiguous as is required by our models:
# MAGIC </p>
# MAGIC 
# MAGIC ![Alt Text](https://raw.githubusercontent.com/Nixtla/statsforecast/main/nbs/imgs/ChainedWindows.gif)
# MAGIC 
# MAGIC Cross-validation of timeseries models is considered a best practice but most implementations are very slow.  The statsforecast library implements cross-validation as a distributed operation, making the process less time consuming to perform:

# COMMAND ----------

# MAGIC %md
# MAGIC In this case, we want to evaluate the performance of each model for the last 3 months (`n_windows=3`), forecasting each month (`step_size=28`).

# COMMAND ----------

# DBTITLE 1,Test models using Cross Validation
sf = StatsForecast(
    models=[ 
      Naive(), # include Naive to make the comparison easier
      CrostonClassic(),
      IMAPA(),
      ADIDA(), # you can include more models such as AutoARIMA
     ], 
    fallback_model=Naive(),
    freq='D', 
    backend=backend
)
y_pred_cv = sf.cross_validation(
    df=y_hist,
    h=28,
    step_size=28,
    n_windows=4
  )

display(y_pred_cv.orderBy('unique_id','ds'))

# COMMAND ----------

# MAGIC %md  The cross-validation method call is very similar to the call used to generate our original forecasts.  We are again forecasting over a 28 day horizon as indicated by the *h* and *freq* parameters, respectively.  We are defining our training window to be equal to 4 steps, as indicated by the *n_windows* parameter.  Each step is defined as 28 days in length (*step_size*).  With each iteration of the cross-validation, the window is moved forward 1 step until the dataset is exhausted.
# MAGIC 
# MAGIC Comparing the model predictions to actuals for each window, we can generate evaluation metrics as follows:

# COMMAND ----------

# DBTITLE 1,Evaluate Individual Cross Validation Forecasts
evaluation_metrics_cv = evaluate_forecasts(
  y_pred_cv.select(fn.col('unique_id'), fn.col('ds'), fn.col('y')), # true values for each window
  y_pred_cv.drop('y', 'cutoff') # forecast of each model
)
 
display( 
  evaluation_metrics_cv.orderBy('unique_id') 
 )

# COMMAND ----------

# DBTITLE 1,Evaluate Cross Validation Forecasts in Summary
summary_results_cv = summarize_evaluation_metrics(evaluation_metrics_cv)

display(summary_results_cv)

# COMMAND ----------

# MAGIC %md ## Step 5: Select Best Model
# MAGIC 
# MAGIC We've now explored multiple models using multiple evaluation metrics leveraging multiple evaluation techniques. Now we need to choose the *best* model for a given store-item combination.  To do this, we need to decide on which metric we will use to define *best* and examine our evaluation results for that metric:

# COMMAND ----------

# DBTITLE 1,Define Function to get the Best Model per Store-Item
def get_best_model_per_series(evaluation_results, metric):
    
    # identify relevant metrics in the results set (which will be in the form model_metric)
    metrics = [c for c in evaluation_results.columns if (metric in c)]
    
    # assemble metrics aggregation logic
    stack_expression = [] 
    for m in metrics:
      stack_expression += [f"'{m.replace(f'_{metric}', '')}'", m] #extract model from metric name
      
    results = (
      evaluation_results
        .select( # pivot results to be unique_id, model, score
          'unique_id', 
          fn.expr(f"stack( {len(metrics)}, {','.join(stack_expression)}) as (model, score)")
          )
        .withColumn( # sequence models for each unique_id (ignoring any score collisions)
          'rank', 
          fn.expr('row_number() over (partition by unique_id order by score, model)') # the sort on model is arbitrary should there be a score collision
          )
      .where(fn.expr('rank = 1')) # get best model
      .select('unique_id', 'model')
      )
    
    return results

# COMMAND ----------

# DBTITLE 1,Get Best Model per Store-Item using RMSE
best_model_per_series = get_best_model_per_series(evaluation_metrics_cv, 'rmse')

display(best_model_per_series)

# COMMAND ----------

# MAGIC %md We now have identified the best model for each store-item combination.  In effect, we've automated a *bake-off* between models which can allow us to find the best approach for a given scenario.  But out of curiosity, were there any standouts in terms of model performance across these entries?:

# COMMAND ----------

# DBTITLE 1,Plot Model Selections
display(
  best_model_per_series
    .groupBy('model')
    .agg(
      fn.count('*').alias('instances')
      )
    .orderBy('instances', ascending=False)
  )

# COMMAND ----------

# MAGIC %md For these data, ADIDA was the most frequently selected *best* model with *IMAPA* in a respectable second-place position meaning that there's no clear *one approach* to generating these forecasts that will work well across all store-item combinations.  Interestingly, the Naive model did pretty well for quite a few products.  It's likely for these that there are sizeable gaps in purchases and/or no clear pattern behind purchase events to indicate a more sophisticated approach.

# COMMAND ----------

# MAGIC %md ##Step 6: Generate Final Forecast
# MAGIC 
# MAGIC With our best model selected for each store-item, we can now generate our final forecast.  We've been using a train-test split of our data in our evaluation, but the reality is that we always are forecasting over periods for which we don't have actuals.  To simulate this, let's combine the training and testing datasets to bring us up to the end of the period for which we have known values:

# COMMAND ----------

# DBTITLE 1,Assemble Full Historical Dataset
# assemble full set of historical data
y_hist_full = (
  spark
    .table('sales_train_evaluation')
    .unionAll(
      spark.table('sales_test_evaluation')
      )
    .orderBy('unique_id','ds')
  )

display(y_hist_full)

# COMMAND ----------

# MAGIC %md  Using our full historical dataset, we can now generate the various forecasts:

# COMMAND ----------

# DBTITLE 1,Generate Forecasts for all Time Series and all Models
sf = StatsForecast(
    models=[ 
      Naive(), # include Naive to make the comparison easier
      CrostonClassic(),
      IMAPA(),
      ADIDA(), # you can include more models such as AutoARIMA
     ], 
    fallback_model=Naive(),
    freq='D', 
    backend=backend
)
forecasts = sf.forecast(
    df=y_hist,
    h=28
)

display(forecasts.orderBy('unique_id','ds'))

# COMMAND ----------

# MAGIC %md And we can choose our best forecast with the definition of another function:

# COMMAND ----------

# DBTITLE 1,Define Function to Select Best Model for each Time Series
def get_best_model_forecasts(y_pred, best_models):
  
    # identify models in the results set
    models = [c for c in y_pred.columns if c not in ['unique_id', 'ds']]
    
    # assemble models for pivot
    stack_expression = []
    for m in models:
      stack_expression += [f"'{m}'", m]
      
    results = (
      y_pred
        .select( # pivot model from field to row
          'unique_id', 
          'ds',
          fn.expr(f"stack( {len(models)}, {','.join(stack_expression)}) as (model, forecast)")
          )
        .join( best_models, on=['unique_id','model']) # join to "best model" selection 
        .select(
          'unique_id', 
          'ds',
          'forecast'
          )
        )
    
    return results

# COMMAND ----------

# DBTITLE 1,Retrieve Best Model Forecast for each Time Series
best_model_per_series_forecast = get_best_model_forecasts(forecasts, best_model_per_series)

_ = (
  best_model_per_series_forecast
    .withColumn('training_date', fn.current_date())
    .write
    .format('delta')
    .mode('append')
    .saveAsTable('forecasts')
  )

display( spark.table('forecasts') )

# COMMAND ----------

# MAGIC %md #BONUS: How Frequently Should You Generate Forecasts?
# MAGIC 
# MAGIC A common question that comes up in forecasting is how often should we generate new forecasts.  Many organizations take advantage of the latest data and the speed and scalability offered by the cloud to regenerate forecasts on a daily basis or even more often. But is that necessary? The answer to that question depends on the speed at which the performance of you forecasts degrade over time.  
# MAGIC 
# MAGIC The further out any forecast goes, the more inaccurate that forecast tends to be. Some forecasts degrade in accuracy slowly and some degrade much more rapidly.  To evaluate the pace at which our store-item forecasts degrade, we can repeat our cross-validation work across multiple horizons and step-sizes and then use the results to observe how accuracy degrades over time.
# MAGIC 
# MAGIC Please note the *test_size* argument in the command below specifies that forecasts will be created at the specified frequency until the forecasting period (specified by the *freq* and *h* arguments) is exhausted. For example, if a 28-day forecast is called for, a test_size of 7 will trigger a new 7 day forecast every 7-days until the 28-day period is completed:
# MAGIC 
# MAGIC **NOTE** These next steps will take a while to complete given the large number for computational cycles involved.  This is a step you should perform periodically but not as part of a day-to-day forecasting cycle.

# COMMAND ----------

# DBTITLE 1,Define Function to Evaluate Forecast at Differing Frequencies
def get_pipeline_generation_evaluation(forecast_frequency, test_size, models):
  
    # generate cross-validation 
    sf = StatsForecast(models=models, fallback_model=Naive(), freq='D', backend=backend)
    y_pred_cv = sf.cross_validation(
      df=y_hist, 
      h=forecast_frequency,
      step_size=forecast_frequency,
      test_size=test_size,
      n_windows=None
    )
    
    # calculate evaluation metrics
    evaluation_metrics_cv = evaluate_forecasts(
      y_pred_cv.select(fn.col('unique_id'), fn.col('ds'), fn.col('y')), # true values for each window
      y_pred_cv.drop('y', 'cutoff') # forecast of each model
      )
    
    # get summary metrics
    summary_results = summarize_evaluation_metrics(evaluation_metrics_cv).withColumn('forecast_frequency', fn.lit(forecast_frequency))
    
    return summary_results

# COMMAND ----------

# DBTITLE 1,Evaluate Forecasts Over Time
# configure evaluation metrics
frequencies = [1, 7, 28] # Runs scenarios where we run the forecasting process daily, weekly, monthly
test_size = 28 # test the run frequencies for a month
models = [ADIDA(), IMAPA()] # models to evaluate

# generate cross validation results
results = None
for frequency in frequencies:

  # get cross validation output
  df = get_pipeline_generation_evaluation(frequency, test_size, models)
  
  # union output with prior output
  if results is None:
    results = df
  else:
    results = results.unionAll(df)


display(results)

# COMMAND ----------

# MAGIC %md From the results, we can see that on average the RMSE associated with our forecasts rise a bit between the 1 day and 7 day forecast interval.  The pace of degradation then appears to slow as we look further out up to the 28 day interval.  The decision as to whether to regenerate forecasts daily, weekly or monthly is therefore more a matter of business tolerance of slightly higher inaccuracy relative to the cost of running more frequent forecast generation cycles.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | statsforecast| Lightning fast forecasting with statistical and econometric models | Apache 2.0 | https://github.com/Nixtla/statsforecast |
