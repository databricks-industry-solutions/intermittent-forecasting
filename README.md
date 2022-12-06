## Intermittent Forecasting Solution Accelerator

In order to capture localized patterns of demand, more and more organizations are generating forecasts at the store-item level.  Working at this level of granularity not only explodes the number of forecasts we need to produce, it frequently presents us with timeseries containing numerous periods of zero-unit demand.  Commonly used models are simply inappropriate for such timeseries, requiring us to turn to specialized algorithms built for intermittent (sparse) timeseries.

In this solution accelerator, we'll leverage historical daily data for a large number of store-item combinations within which numerous periods of zero-unit sales are observed in order to explore how we might overcome this challenge.  We will make use of the [Nixtla statsforecast library](https://github.com/Nixtla/statsforecast) which has support for numerous models specialized for intermittent demand forecasting and which has a built-in backend for scaling the training of large numbers of models across a Databricks cluster.

___

&copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.

| library                                | description             | license    | source                                              |
|----------------------------------------|-------------------------|------------|-----------------------------------------------------|
| statsforecast| Lightning fast forecasting with statistical and econometric models | Apache 2.0 | https://github.com/Nixtla/statsforecast |

## Instruction

To run this accelerator, clone this repo into a Databricks workspace. Attach the `RUNME` notebook to any cluster and execute the notebook via Run-All. A multi-step-job describing the accelerator pipeline will be created, and the link will be provided. Execute the multi-step-job to see how the pipeline runs. The job configuration is written in the RUNME notebook in json format. The cost associated with running the accelerator is the user's responsibility.
