'''
some data analysis methods useful with spark
'''

from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import ChiSquareTest


def test_chisquare():
    data = [(0.0, Vectors.dense(0.5, 10.0)),
        (0.0, Vectors.dense(1.5, 20.0)),
        (1.0, Vectors.dense(1.5, 30.0)),
        (0.0, Vectors.dense(3.5, 30.0)),
        (0.0, Vectors.dense(3.5, 40.0)),
        (1.0, Vectors.dense(3.5, 40.0))]
    df = spark.createDataFrame(data, ["label", "features"])
    r = ChiSquareTest.test(df, "features", "label").head()
    print("pValues: " + str(r.pValues))
    print("degreesOfFreedom: " + str(r.degreesOfFreedom))
    print("statistics: " + str(r.statistics))

    #cross table
    df.stat.crosstab("age_class", "Occupation").show()


from pyspark.ml.classification import LogisticRegressionTrainingSummary

