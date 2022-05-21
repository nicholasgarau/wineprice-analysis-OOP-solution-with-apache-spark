from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import RandomForestRegressor as RF
from pyspark.sql.functions import round
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorSlicer
import matplotlib.pyplot as plt
import pandas as pd
from itertools import chain


def linear_regressor(dataframe):
    training_set, test_set = dataframe.randomSplit([0.8, 0.2])
    regressor = LinearRegression(featuresCol='features', labelCol='WinePrice', predictionCol='prediction')
    model_lm = regressor.fit(training_set)
    performance_lm = model_lm.evaluate(test_set)
    print(f"R2 measure: \t {performance_lm.r2}")
    print(f"Mean Absolute Error: \t {performance_lm.meanAbsoluteError}")
    print(f"Root Mean Squared Error: \t {performance_lm.rootMeanSquaredError}")
    predictions_lm = model_lm.transform(test_set)
    predictions_lm.select('WinePrice', round('prediction', 2).alias('prediction')).show(20)


def extract_feature_imp(featureImp, dataset, featuresCol):
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return (varlist.sort_values('score', ascending=False))


def random_forest_regressor(dataframe):
    training_set, test_set = dataframe.randomSplit([0.8, 0.2])
    regressor_rf = RF(featuresCol='features', labelCol='WinePrice', numTrees=10, maxDepth=5, seed=1000)
    model_rf = regressor_rf.fit(training_set)
    predictions_rf = model_rf.transform(test_set)
    evaluator = RegressionEvaluator(labelCol="WinePrice", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions_rf)
    rf_pred = model_rf.transform(dataframe)
    rf_result = rf_pred.toPandas()
    plt.plot(rf_result.WinePrice, rf_result.prediction, 'bo')
    plt.xlabel('Price')
    plt.ylabel('Prediction')
    plt.suptitle("Model Performance RMSE: %f" % rmse)
    plt.show()
    print(f'Feature importance values:{model_rf.featureImportances}')
    print(f'FI values (descent order): \t  {sorted(model_rf.featureImportances.values, reverse=True)[:10]}')
    # attrs = sorted(
    #     (attr["idx"], attr["name"]) for attr in (chain(*dataframe
    #                                                    .schema["features"]
    #                                                    .metadata["ml_attr"]["attrs"].values())))
    # feature_importance_meaning = [(name, model_rf.featureImportances[idx]) for idx, name in attrs if
    #                               model_rf.featureImportances[idx]]
    # print(f'FEATURE IMPORTANCE: \t {feature_importance_meaning}')
    return model_rf


def subset_selection_rf(model, dataframe):
    varlist = extract_feature_imp(model.featureImportances, dataframe, "features")
    print(f'FEATURE IMPORTANCE extract post function: \n {varlist}')
    varidx = [x for x in varlist['idx'][0:20]]
    slicer = VectorSlicer(inputCol="features", outputCol="features_sub", indices=varidx)
    df_subset = slicer.transform(dataframe)
    df_subset = df_subset.drop('rawPrediction', 'probability', 'prediction')
    regressor_rf_sub = RF(labelCol='WinePrice', featuresCol='features', numTrees=10, maxDepth=5, seed=1000)
    model_rf_sub = regressor_rf_sub.fit(df_subset)
    predictions_rf = model_rf_sub.transform(df_subset)
    evaluator = RegressionEvaluator(labelCol="WinePrice", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions_rf)
    rf_pred = model_rf_sub.transform(dataframe)
    rf_result = rf_pred.toPandas()
    plt.plot(rf_result.WinePrice, rf_result.prediction, 'bo')
    plt.xlabel('Price')
    plt.ylabel('Prediction')
    plt.suptitle("Model Performance (subset sel) RMSE: %f" % rmse)
    plt.show()
