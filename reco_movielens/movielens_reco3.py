from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('recommender').getOrCreate()
spark.sparkContext.setLogLevel('WARN')
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator


df = spark.read.csv("datasets/ml-latest-small/ratings.csv", inferSchema=True, header=True)

print(df.printSchema())

print(df.columns)

print(df.head())

print(df.show(3))

print(df.describe().show())

training, test = df.randomSplit([0.8,0.2])

als = ALS(maxIter=5, regParam=0.01, userCol='userId', itemCol='movieId', ratingCol='rating')

model = als.fit(training)

predictions = model.transform(test)

print(predictions.describe().show())


predictions = predictions.na.drop()

print(predictions.describe().show())

evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating')
rmse = evaluator.evaluate(predictions)
print("RMSE is ",rmse)