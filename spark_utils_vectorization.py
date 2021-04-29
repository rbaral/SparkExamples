from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import Normalizer, StandardScaler, MinMaxScaler, MaxAbsScaler
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import PCA


import numpy as np

spark = SparkSession.builder.appName('recommender').getOrCreate()
spark.sparkContext.setLogLevel('WARN')


def count_vectorizer():
    sentenceData = spark.createDataFrame([
        (0, "Python python Spark Spark"),
        (1, "Python SQL")],
     ["document", "sentence"])
    tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
    vectorizer  = CountVectorizer(inputCol="words", outputCol="rawFeatures")
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    pipeline = Pipeline(stages=[tokenizer, vectorizer, idf])
    model = pipeline.fit(sentenceData)
    total_counts = model.transform(sentenceData).select('rawFeatures').rdd.map(lambda row: row['rawFeatures'].toArray()).reduce(lambda x,y: [x[i]+y[i] for i in range(len(y))])
    print("total counts ",total_counts)
    vocabList = model.stages[1].vocabulary
    d = {'vocabList':vocabList,'counts':total_counts}
    print(spark.createDataFrame(np.array(list(d.values())).T.tolist(),list(d.keys())).show())
    counts = model.transform(sentenceData).select('rawFeatures').collect()
    print(counts)

    print(model.transform(sentenceData).show(truncate=False))


def one_hot_encoder():
    df = spark.createDataFrame([
        (0, "a"),
        (1, "b"),
        (2, "c"),
        (3, "a"),
        (4, "a"),
        (5, "c")
    ], ["id", "category"])
    print(df.show())
    stringIndexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
    model = stringIndexer.fit(df)
    indexed = model.transform(df)
    # default setting: dropLast=True
    encoder = OneHotEncoder(inputCol="categoryIndex", outputCol="categoryVec", dropLast = False)
    encoded = encoder.transform(indexed)
    print(encoded.show())


def vector_assembler():
    df = spark.createDataFrame([
        (0, "a"),
        (1, "b"),
        (2, "c"),
        (3, "a"),
        (4, "a"),
        (5, "c")
    ], ["id", "category"])
    categoricalCols = ['category']
    indexers = [StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c)) for c in categoricalCols]
    # default setting: dropLast=True
    encoders = [OneHotEncoder(inputCol=indexer.getOutputCol(),outputCol="{0}_encoded".format(indexer.getOutputCol()), dropLast = False) for indexer in indexers]
    assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders], outputCol="features")
    pipeline = Pipeline(stages=indexers + encoders + [assembler])
    model = pipeline.fit(df)
    data = model.transform(df)
    print(data.show())



def vector_scaler(scaler_type="Normal"):
    df = spark.createDataFrame([
        (0, Vectors.dense([1.0, 0.5, -1.0]),),
        (1, Vectors.dense([2.0, 1.0, 1.0]),),
        (2, Vectors.dense([4.0, 10.0, 2.0]),)
    ], ["id", "features"])
    print(df.show())
    #scaler_type = 'Normal'
    if scaler_type == 'Normal':
        scaler = Normalizer(inputCol="features", outputCol="scaledFeatures", p=1.0)
    elif scaler_type == 'Standard':
        scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                                withStd=True, withMean=False)
    elif scaler_type == 'MinMaxScaler':
        scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
    elif scaler_type == 'MaxAbsScaler':
        scaler = MaxAbsScaler(inputCol="features", outputCol="scaledFeatures")
    pipeline = Pipeline(stages=[scaler])
    model = pipeline.fit(df)
    data = model.transform(df)
    print(data.show())


def vector_normalizer():
    dataFrame = spark.createDataFrame([
        (0, Vectors.dense([1.0, 0.5, -1.0]),),
        (1, Vectors.dense([2.0, 1.0, 1.0]),),
        (2, Vectors.dense([4.0, 10.0, 2.0]),)
    ], ["id", "features"])
    # Normalize each Vector using $L^1$ norm.
    normalizer = Normalizer(inputCol="features", outputCol="normFeatures", p=1.0)
    l1NormData = normalizer.transform(dataFrame)
    print("Normalized using L^1 norm")
    print(l1NormData.show())
    # Normalize each Vector using $L^\infty$ norm.
    lInfNormData = normalizer.transform(dataFrame, {normalizer.p: float("inf")})
    print("Normalized using L^inf norm")
    print(lInfNormData.show())


def do_PCA():
    data = [(Vectors.sparse(5, [(1, 1.0), (3, 7.0)]),),
            (Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),),
            (Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),)]
    df = spark.createDataFrame(data, ["features"])
    pca = PCA(k=3, inputCol="features", outputCol="pcaFeatures")
    model = pca.fit(df)
    result = model.transform(df).select("pcaFeatures")
    print(result.show(truncate=False))


#count_vectorizer()

#one_hot_encoder()

#vector_assembler()

#vector_scaler()

#vector_normalizer()

do_PCA()