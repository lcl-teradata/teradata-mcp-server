import argparse
import os
from urllib.parse import urlparse

from teradataml import *


def main():
    parser = argparse.ArgumentParser(description="Teradata MCP Server")
    parser.add_argument('--database_uri', type=str, required=False, help='Database URI to connect to: teradata://username:password@host:1025/schemaname')
    parser.add_argument('--action', type=str, choices=['setup', 'cleanup'], required=True, help='Action to perform: setup, test or cleanup')
    # Extract known arguments and load them into the environment if provided
    args, unknown = parser.parse_known_args()

    connection_url = args.database_uri or os.getenv("DATABASE_URI")

    eng = None
    if args.action in ['setup', 'cleanup']:
        if not connection_url:
            raise ValueError("DATABASE_URI must be provided either as an argument or as an environment variable.")

        parsed_url = urlparse(connection_url)
        user = parsed_url.username
        password = parsed_url.password
        host = parsed_url.hostname
        database = user

        eng = create_context(host=host, username=user, password=password)

    if args.action=='setup':
        # Set up the analytic functions test data.

        # Setup for ANOVA.
        load_example_data("teradataml", ["insect_sprays"])

        # Setup for Antiselect.
        load_example_data("dataframe", ["sales"])

        # Setup for Apriori.
        load_example_data("apriori", ["trans_dense", "trans_sparse"])

        # Setup for BincodeFit & Transform.
        load_example_data("teradataml", ["titanic", "bin_fit_ip"])
        titanic = DataFrame.from_table("titanic")
        bin_code_2 = BincodeFit(data=titanic,
                                target_columns='age',
                                method_type='Equal-Width',
                                nbins=2,
                                label_prefix='label_prefix'
                                )
        bin_code_2.output.to_sql(table_name="bin_fit_op", if_exists="replace")

        # Setup for CFilter
        load_example_data("dataframe", ["grocery_transaction"])

        # Setup for CategoricalSummary
        load_example_data("teradataml", ["titanic"])

        # Setup for ChiSq
        load_example_data("teradataml", "chi_sq")

        # Setup for ClassificationEvaluator
        load_example_data("teradataml", ["titanic"])
        df = DataFrame("titanic")
        ndf = df.assign(pcol=df.survived)
        ndf.to_sql(table_name="CVTable", if_exists="replace")

        # Setup for ColumnSummary
        load_example_data("teradataml", ["titanic"])

        # Setup for ColumnTransformer
        titanic = DataFrame.from_table("titanic")
        bin_code = BincodeFit(data=titanic,
                              target_columns='age',
                              method_type='Equal-Width',
                              nbins=2,
                              label_prefix='label_prefix')
        bin_code.output.to_sql(table_name="bin_fit_op", if_exists="replace")

        # Setup for DecisionForest & Predict.
        load_example_data("decisionforest", ["boston"])
        boston = DataFrame.from_table("boston")

        DecisionForest_out = DecisionForest(data=boston,
                                            input_columns=['crim', 'zn', 'indus', 'chas', 'nox', 'rm',
                                                           'age', 'dis', 'rad', 'tax', 'ptratio', 'black',
                                                           'lstat'],
                                            response_column='medv',
                                            max_depth=12,
                                            num_trees=4,
                                            min_node_size=1,
                                            mtry=3,
                                            mtry_seed=1,
                                            seed=1,
                                            tree_type='REGRESSION')
        DecisionForest_out.result.to_sql(table_name="decision_forest_op", if_exists="replace")

        # Setup for OneHotEncodingFit & Transform
        one_hot_encoding = OneHotEncodingFit(data=titanic,
                                             is_input_dense=True,
                                             target_column="sex",
                                             categorical_values=["male", "female"],
                                             other_column="other")
        one_hot_encoding.result.to_sql(table_name="one_hot_op", if_exists="replace")

        # Setup for GetFutileColumns.
        load_example_data("teradataml", ["titanic"])
        CategoricalSummary_out = CategoricalSummary(data=titanic,
                                                    target_columns=["cabin", "sex", "ticket"])
        CategoricalSummary_out.result.to_sql(table_name="cat_summary_op", if_exists="replace")

        # Setup for Fit.
        load_example_data("teradataml", ["iris_input", "transformation_table"])

        # Setup for KMeans.
        load_example_data("kmeans", "computers_train1")
        load_example_data("kmeans",'kmeans_table')
        computers_train1 = DataFrame.from_table("computers_train1")
        KMeans_out = KMeans(id_column="id",
                            target_columns=['price', 'speed'],
                            data=computers_train1,
                            num_clusters=2)
        KMeans_out.result.to_sql(table_name="kmeans_op", if_exists="replace")

        # Setup for KNN.
        load_example_data("knn", ["computers_train1_clustered", "computers_test1"])

        # Create teradataml DataFrame objects.
        computers_test1 = DataFrame.from_table("computers_test1")
        computers_train1_clustered = DataFrame.from_table("computers_train1_clustered")

        # Generate fit object for column "computer_category".
        fit_obj = OneHotEncodingFit(data=computers_train1_clustered,
                                    is_input_dense=True,
                                    target_column="computer_category",
                                    categorical_values=["ultra", "special"],
                                    other_column="other")

        # Encode "ultra" and "special" values of column "computer_category".
        computers_train1_encoded = OneHotEncodingTransform(data=computers_train1_clustered,
                                                           object=fit_obj.result,
                                                           is_input_dense=True)
        computers_train1_encoded.result.to_sql(table_name="knn_OHE_op", if_exists="replace")

        # Setup for MovingAverage.
        load_example_data("movavg", ["ibm_stock"])

        # Setup for NerExtractor.
        load_example_data("tdnerextractor", ["ner_input_eng", "ner_dict", "ner_rule"])

        # Setup for NGramSplitter.
        load_example_data("ngrams", ["paragraphs_input"])

        # Setup for NaiveBayesTextClassifierPredict & Trainer.
        load_example_data("NaiveBayesTextClassifierPredict", ["complaints_tokens_test", "token_table"])

        token_table = DataFrame("token_table")

        # Create a model which is output of NaiveBayesTextClassifierTrainer.
        nbt_out = NaiveBayesTextClassifierTrainer(data=token_table,
                                                  token_column='token',
                                                  doc_id_column='doc_id',
                                                  doc_category_column='category',
                                                  model_type="Bernoulli")
        nbt_out.result.to_sql(table_name="nbt_op", if_exists="replace")

        # Setup for NonLinearCombineFit & Transform
        Fit_out = NonLinearCombineFit(data=titanic,
                                      target_columns=["sibsp", "parch", "fare"],
                                      formula="Y=(X0+X1+1)*X2",
                                      result_column="total_cost")
        Fit_out.result.to_sql(table_name="non_linear_fit_op", if_exists="replace")

        # Setup for NumApply
        load_example_data("teradataml", ["numerics"])

        # setup for OneClassSVM & Predict
        load_example_data("larpredict", ["diabetes"])
        load_example_data("teradataml", ["cal_housing_ex_raw"])

        # Create teradataml DataFrame objects.
        data_input = DataFrame.from_table("cal_housing_ex_raw")

        # Scale "target_columns" with respect to 'STD' value of the column.
        fit_obj = ScaleFit(data=data_input,
                           target_columns=['MedInc', 'HouseAge', 'AveRooms',
                                           'AveBedrms', 'Population', 'AveOccup',
                                           'Latitude', 'Longitude'],
                           scale_method="STD")

        # Transform the data.
        transform_obj = ScaleTransform(data=data_input,
                                       object=fit_obj.output,
                                       accumulate=["id", "MedHouseVal"])
        transform_obj.result.to_sql(table_name="scale_transform_op", if_exists="replace")
        # Train the input data by OneClassSVM which helps model
        # to find anomalies in transformed data.
        one_class_svm = OneClassSVM(data=transform_obj.result,
                                   input_columns=['MedInc', 'HouseAge', 'AveRooms',
                                                  'AveBedrms', 'Population', 'AveOccup',
                                                  'Latitude', 'Longitude'],
                                   local_sgd_iterations=537,
                                   batch_size=1,
                                   learning_rate='constant',
                                   initial_eta=0.01,
                                   lambda1=0.1,
                                   alpha=0.0,
                                   momentum=0.0,
                                   iter_max=1
                                   )
        one_class_svm.result.to_sql(table_name="one_class_svm_op", if_exists="replace")

        # Setup for OneHotEncodingFit & Transform
        load_example_data("teradataml", ["titanic"])
        fit_obj = OneHotEncodingFit(data=titanic,
                                    is_input_dense=True,
                                    target_column="sex",
                                    categorical_values=["male", "female"],
                                    other_column="other")
        fit_obj.result.to_sql(table_name="one_hot_fit_op", if_exists="replace")

        # Setup for OrdinalEncodingFit & Transform
        load_example_data("teradataml", ["titanic"])
        ordinal_encodingfit_res_2 = OrdinalEncodingFit(target_column='sex',
                                                       approach='LIST',
                                                       categories=['category0', 'category1'],
                                                       ordinal_values=[1, 2],
                                                       start_value=0,
                                                       default_value=-1,
                                                       data=titanic)
        ordinal_encodingfit_res_2.result.to_sql(table_name="ordinal_fit_op", if_exists="replace")

        # Setup for OutlierFilterFit & Transform
        fit_obj = OutlierFilterFit(data=titanic,
                                   target_columns="fare",
                                   lower_percentile=0.1,
                                   upper_percentile=0.9,
                                   outlier_method="PERCENTILE",
                                   replacement_value="MEDIAN",
                                   percentile_method="PERCENTILECONT")
        fit_obj.result.to_sql(table_name="outlier_fit_op", if_exists="replace")

        # Setup for Pack.
        load_example_data("pack", ["ville_temperature"])

        # Setup for Pivoting & Unpivoting.
        load_example_data('unpivot', 'titanic_dataset_unpivoted')

        # Setup for PolynomialFeaturesFit & Transform
        load_example_data("teradataml", ["numerics"])
        numerics = DataFrame.from_table("numerics")
        fit_obj = PolynomialFeaturesFit(data=numerics,
                                        target_columns=["integer_col", "smallint_col"],
                                        degree=2)
        fit_obj.output.to_sql(table_name="poly_fit_op", if_exists="replace")

        # Setup for QQNorm
        load_example_data("teradataml", ["rank_table"])

        # Setup for ROC
        load_example_data("roc", ["roc_input"])

        # Setup for RandomProjectionFit, RandomProjectionMinComponents & Transform
        load_example_data("teradataml", "stock_movement")
        stock_movement = DataFrame.from_table("stock_movement")
        fit_obj = RandomProjectionFit(data=stock_movement,
                                      target_columns="1:",
                                      epsilon=0.9,
                                      num_components=343)
        fit_obj.result.to_sql(table_name="random_proj_fit_op", if_exists="replace")

        # Setup for RowNormalizeFit & Transform
        load_example_data("teradataml", ["numerics"])
        numerics = DataFrame.from_table("numerics")
        fit_obj = RowNormalizeFit(data=numerics,
                                  target_columns=["integer_col", "smallint_col"],
                                  approach="INDEX",
                                  base_column="integer_col",
                                  base_value=100.0)
        fit_obj.output.to_sql(table_name="row_normalize_fit_op", if_exists="replace")

        # Setup for SMOTE
        load_example_data("dataframe", "iris_test")

        # Setup for SVM, SVMPredict & SVMSparsePredict

        # Load the example data.
        load_example_data("teradataml", ["cal_housing_ex_raw"])

        # Create teradataml DataFrame objects.
        data_input = DataFrame.from_table("cal_housing_ex_raw")

        # Scale "target_columns" with respect to 'STD' value of the column.
        fit_obj = ScaleFit(data=data_input,
                           target_columns=['MedInc', 'HouseAge', 'AveRooms',
                                           'AveBedrms', 'Population', 'AveOccup',
                                           'Latitude', 'Longitude'],
                           scale_method="STD")

        # Transform the data.
        transform_obj = ScaleTransform(data=data_input,
                                       object=fit_obj.output,
                                       accumulate=["id", "MedHouseVal"])
        transform_obj.result.to_sql(table_name="scale_transform_op2", if_exists="replace")

        # Train the transformed data using SVM() where "model_type" is 'Regression'.
        svm_obj1 = SVM(data=transform_obj.result,
                      input_columns=['MedInc', 'HouseAge', 'AveRooms',
                                     'AveBedrms', 'Population', 'AveOccup',
                                     'Latitude', 'Longitude'],
                      response_column="MedHouseVal",
                      model_type="Regression"
                      )
        svm_obj1.result.to_sql(table_name="svm_op", if_exists="replace")

        # Setup for ScaleFit & Transform
        load_example_data("teradataml", ["scale_housing"])

        scaling_house = DataFrame.from_table("scale_housing")

        fit_obj = ScaleFit(data=scaling_house,
                           target_columns="lotsize",
                           scale_method="MEAN",
                           miss_value="KEEP",
                           global_scale=False,
                           multiplier="1",
                           intercept="0")
        fit_obj.output.to_sql(table_name="scale_fit_op", if_exists="replace")

        # Setup for SentimentExtractor
        load_example_data("sentimentextractor", ["sentiment_extract_input"])

        # Setup for Shap
        load_example_data("byom", "iris_input")
        load_example_data("teradataml", ["cal_housing_ex_raw"])
        iris_input = DataFrame("iris_input")

        XGBoost_out = XGBoost(data=iris_input,
                              input_columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
                              response_column='species',
                              model_type='Classification',
                              iter_num=25)
        XGBoost_out.result.to_sql(table_name="xgboost_op", if_exists="replace")

        # Setup for Silhouette
        load_example_data("teradataml", ["mobile_data"])

        # Setup for SimpleImputeFit & Transform
        fit_obj = SimpleImputeFit(data=titanic,
                                  stats_columns="age",
                                  literals_columns="cabin",
                                  stats="median",
                                  literals="General")
        fit_obj.output.to_sql(table_name="simple_impute_fit_op", if_exists="replace")

        # Setup for StrApply

        # Setup for StringSimilarity
        load_example_data("stringsimilarity", ["strsimilarity_input"])

        # Setup for TDNaiveBayesPredict

        # Load the example data.
        load_example_data("decisionforestpredict", ["housing_train", "housing_test"])

        # Create teradataml DataFrame objects.
        housing_train = DataFrame.from_table("housing_train")

        # Import function  TDNaiveBayesPredict.
        from teradataml import NaiveBayes

        # Example 1: TDNaiveBayesPredict function to predict the classification label using Dense input.
        NaiveBayes_out = NaiveBayes(data=housing_train, response_column='homestyle',
                                    numeric_inputs=['price', 'lotsize', 'bedrooms', 'bathrms', 'stories', 'garagepl'],
                                    categorical_inputs=['driveway', 'recroom', 'fullbase', 'gashw', 'airco',
                                                        'prefarea'])
        NaiveBayes_out.result.to_sql(table_name="naive_bayes_op", if_exists="replace")

        # Setup for TFIDF
        load_example_data('naivebayestextclassifier', "token_table")

        # Setup for TargetEncodingFit & Transform

        # Create teradataml DataFrame objects.
        data_input = DataFrame.from_table("titanic")

        # Find the distinct values and counts for column 'sex' and 'embarked'.
        categorical_summ = CategoricalSummary(data=data_input,
                                              target_columns = ["sex", "embarked"]
                                              )

        # Find the distinct count of 'sex' and 'embarked' in which only 2 column should be present
        #  name 'ColumnName' and 'CategoryCount'.
        category_data=categorical_summ.result.groupby('ColumnName').count()
        category_data = category_data.assign(drop_columns=True,
                                             ColumnName=category_data.ColumnName,
                                             CategoryCount=category_data.count_DistinctValue)
        category_data.to_sql(table_name="category_data_op", if_exists="replace")

        # Generates the required hyperparameters when "encoder_method" is 'CBM_BETA'.
        TargetEncodingFit_out = TargetEncodingFit(data=data_input,
                                                  category_data=category_data,
                                                  encoder_method='CBM_BETA',
                                                  target_columns=['sex', 'embarked'],
                                                  response_column='survived',
                                                  default_values=[-1, -2]
                                                  )
        TargetEncodingFit_out.result.to_sql(table_name="target_encoding_fit_op", if_exists="replace")

        # Setup for TextMorph
        load_example_data("textmorph", ["words_input", "pos_input"])

        # Setup for TextParser
        load_example_data("textparser", ["complaints", "stop_words"])

        # Setup for TrainTestSplit
        # Setup for Transform
        load_example_data("teradataml", ["iris_input", "transformation_table"])
        # Create teradataml DataFrame objects.
        iris_input = DataFrame.from_table("iris_input")
        transformation_df = DataFrame.from_table("transformation_table")
        transformation_df = transformation_df.drop(['id'], axis=0)

        # Example 1: Run Fit() with all arguments and pass the output to Transform().
        fit_obj = Fit(data=iris_input,
                     object=transformation_df,
                     object_order_column='TargetColumn'
                     )
        fit_obj.result.to_sql(table_name="fit_op", if_exists="replace")

        # Run Transform() with persist as True in order to save the result.
        transform_result = Transform(data=iris_input,
                                     data_partition_column='sepal_length',
                                     data_order_column='sepal_length',
                                     object=fit_obj.result,
                                     object_order_column='TargetColumn',
                                     id_columns=['species', 'id'],
                                     persist=True
                                     )

        # Setup for UnivariateStatistics

        # Setup for Unpack
        load_example_data("Unpack", ["ville_tempdata", "ville_tempdata1"])

        # Setup for Unpivoting
        load_example_data('unpivot', 'unpivot_input')

        # Setup for VectorDistance
        load_example_data("vectordistance", ["target_mobile_data_dense", "ref_mobile_data_dense"])

        # Setup for WhichMax
        # Setup for WhichMin
        # Setup for WordEmbeddings
        load_example_data("teradataml", ["word_embed_model", "word_embed_input_table1"])

        # Setup for XGBoost & Predict
        XGBoost_out_1 = XGBoost(data=titanic,
                                input_columns=["age", "survived", "pclass"],
                                response_column='fare',
                                max_depth=3,
                                lambda1=1000.0,
                                model_type='Regression',
                                seed=-1,
                                shrinkage_factor=0.1,
                                iter_num=2)
        XGBoost_out_1.result.to_sql(table_name="xgboost_op2", if_exists="replace")

        # Setup for ZTest

    elif args.action in ('cleanup'):

        # Cleanup for ANOVA.
        db_drop_table(table_name="insect_sprays", suppress_error=True)

        # Cleanup for Antiselect.
        db_drop_table(table_name="antiselect_input", suppress_error=True)

        # Cleanup for ClassificationEvaluator
        db_drop_table(table_name='CVTable', suppress_error=True)

        # Cleanup for ChiSq


        # Cleanup for ConvertTo
        db_drop_table(table_name="convert_to_output_tbl", suppress_error=True)

        print("Or you can run the cleanup action of this script with: `analytic_functions_setup.py --action cleanup`")
    else:
        raise ValueError(f"Unknown action: {args.action}")

    # Drop the context if it was created
    if eng:
        remove_context()


if __name__ == '__main__':
    main()
