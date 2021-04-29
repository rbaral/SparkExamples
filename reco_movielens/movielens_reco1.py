'''
ref:https://www.codementor.io/@jadianes/building-a-recommender-with-apache-spark-python-example-app-part1-du1083qbw
https://www.codementor.io/@jadianes/building-a-web-service-with-apache-spark-flask-example-app-part2-du1083854
'''

import os
import urllib.request
import zipfile
from time import time
# Setting up spark
import findspark
findspark.init()
from pyspark import SparkConf, SparkContext
#from pyspark.sql import SparkSession
from pyspark.sql import *
#from pyspark.sql.functions import *
from pyspark.mllib.recommendation import ALS
from pyspark.mllib.recommendation import MatrixFactorizationModel
import math
conf = SparkConf().setMaster("local[*]").setAppName("PySpark_feature_engineering")
spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')


complete_dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-latest.zip'
small_dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
datasets_path = os.path.join('/Users/dur-rbaral-m/projects/test_projects/SparkBasics/reco_movielens', 'datasets')
small_ratings_file = os.path.join(datasets_path, 'ml-latest-small', 'ratings.csv')

#not required if run with spark-submit
#sc= SparkContext()

SEED = 5
ITERATIONS = 10
REG_PARAM = 0.1
RANKS = [4, 8, 12]
ERRORS = [0, 0, 0]
ERR = 0
TOLERANCE = 0.02

def download_data():
    """
    download the data if not exist
    :return:
    """
    print("Downloading data")
    if not os.path.exists(small_ratings_file):
        complete_dataset_path = os.path.join(datasets_path, 'ml-latest.zip')
        small_dataset_path = os.path.join(datasets_path, 'ml-latest-small.zip')
        small_f = urllib.request.urlretrieve(small_dataset_url, small_dataset_path)
        complete_f = urllib.request.urlretrieve(complete_dataset_url, complete_dataset_path)

        with zipfile.ZipFile(small_dataset_path, "r") as z:
            z.extractall(datasets_path)

        with zipfile.ZipFile(complete_dataset_path, "r") as z:
            z.extractall(datasets_path)


def get_small_data():
    """
    read data
    :return:
    """
    small_ratings_raw_data = spark.sparkContext.textFile(small_ratings_file)
    small_ratings_raw_data_header = small_ratings_raw_data.take(1)[0]

    small_ratings_data = small_ratings_raw_data.filter(lambda line: line!=small_ratings_raw_data_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1],tokens[2])).cache()

    #check some datasets
    print(small_ratings_data.take(3))

    small_movies_file = os.path.join(datasets_path, 'ml-latest-small', 'movies.csv')

    small_movies_raw_data = spark.sparkContext.textFile(small_movies_file)
    small_movies_raw_data_header = small_movies_raw_data.take(1)[0]

    small_movies_data = small_movies_raw_data.filter(lambda line: line != small_movies_raw_data_header) \
        .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0], tokens[1])).cache()

    print(small_movies_data.take(3))
    print("There are %s recommendations in the small dataset" % (small_movies_data.count()))
    return small_ratings_data


def get_complete_data():
    # Load the complete dataset file
    complete_ratings_file = os.path.join(datasets_path, 'ml-latest', 'ratings.csv')
    complete_ratings_raw_data = spark.sparkContext.textFile(complete_ratings_file)
    complete_ratings_raw_data_header = complete_ratings_raw_data.take(1)[0]

    # Parse
    complete_ratings_data = complete_ratings_raw_data.filter(lambda line: line != complete_ratings_raw_data_header) \
        .map(lambda line: line.split(",")).map(
        lambda tokens: (int(tokens[0]), int(tokens[1]), float(tokens[2]))).cache()

    print("There are %s recommendations in the complete dataset" % (complete_ratings_data.count()))
    return complete_ratings_data


#count the number of ratings per movie
def get_counts_and_averages(ID_and_ratings_tuple):
    nratings = len(ID_and_ratings_tuple[1])
    return ID_and_ratings_tuple[0], (nratings, float(sum(x for x in ID_and_ratings_tuple[1]))/nratings)


def get_complete_data_with_new_ratings():
    # load complete movies
    complete_movies_file = os.path.join(datasets_path, 'ml-latest', 'movies.csv')
    complete_movies_raw_data = spark.sparkContext.textFile(complete_movies_file)
    complete_movies_raw_data_header = complete_movies_raw_data.take(1)[0]

    # Parse
    complete_movies_data = complete_movies_raw_data.filter(lambda line: line != complete_movies_raw_data_header) \
        .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]), tokens[1], tokens[2])).cache()

    complete_movies_titles = complete_movies_data.map(lambda x: (int(x[0]), x[1]))

    print("There are %s movies in the complete dataset" % (complete_movies_titles.count()))
    movie_ID_with_ratings_RDD = (complete_ratings_data.map(lambda x: (x[1], x[2])).groupByKey())
    movie_ID_with_avg_ratings_RDD = movie_ID_with_ratings_RDD.map(get_counts_and_averages)
    movie_rating_counts_RDD = movie_ID_with_avg_ratings_RDD.map(lambda x: (x[0], x[1][0]))

    # add new user ratings
    new_user_ID = 0

    # The format of each line is (userID, movieID, rating)
    new_user_ratings = [
        (0, 260, 4),  # Star Wars (1977)
        (0, 1, 3),  # Toy Story (1995)
        (0, 16, 3),  # Casino (1995)
        (0, 25, 4),  # Leaving Las Vegas (1995)
        (0, 32, 4),  # Twelve Monkeys (a.k.a. 12 Monkeys) (1995)
        (0, 335, 1),  # Flintstones, The (1994)
        (0, 379, 1),  # Timecop (1994)
        (0, 296, 3),  # Pulp Fiction (1994)
        (0, 858, 5),  # Godfather, The (1972)
        (0, 50, 4)  # Usual Suspects, The (1995)
    ]
    new_user_ratings_RDD = spark.sparkContext.parallelize(new_user_ratings)
    print('New user ratings: %s' % new_user_ratings_RDD.take(10))

    # add the datasets to the datasets we use to train our model
    complete_data_with_new_ratings_RDD = complete_ratings_data.union(new_user_ratings_RDD)
    return complete_data_with_new_ratings_RDD, new_user_ratings, complete_movies_data, complete_movies_titles, movie_rating_counts_RDD



#download the data
download_data()

#get the data
small_ratings_data = get_small_data()

#selecting ALS parameters using small dataset
training_RDD, validation_RDD, test_RDD = small_ratings_data.randomSplit([6, 2, 2], seed=0)
validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))


def train_small():
    print("Training with sample datasets")
    min_error = float('inf')
    best_rank = -1
    best_iteration = -1
    ERR = 0
    for rank in RANKS:
        model = ALS.train(training_RDD, rank, seed=SEED, iterations=ITERATIONS,
                          lambda_=REG_PARAM)
        predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
        rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
        error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
        ERRORS[ERR] = error
        ERR+= 1
        print('For rank %s the RMSE is %s' % (rank, error))
        if error < min_error:
            min_error = error
            best_rank = rank

    print('The best model was trained with rank %s' % best_rank)
    # persisting the model
    model_path = os.path.join('..', 'models', 'movie_lens_small_als')

    # Save and load model
    model.save(spark.sparkContext, model_path)
    same_model = MatrixFactorizationModel.load(spark.sparkContext, model_path)

    # Among other things, you will see in your filesystem that there are folder with product and user datasets into Parquet format files.

    #lets check the predictions
    #Basically we have the UserID, the MovieID, and the Rating, as we have in our ratings dataset.
    #In this case the predictions third element, the rating for that movie and user, is the predicted by our ALS model.
    print(predictions.take(3))
    return best_rank


def test_small(best_rank):
    print("testing on sample data")
    # test the selected model
    model = ALS.train(training_RDD, best_rank, seed=SEED, iterations=ITERATIONS, lambda_=REG_PARAM)
    predictions = model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean())
    print('For sample testing datasets the RMSE is %s' % (error))


#work with bigger data set
complete_ratings_data = get_complete_data()
training_RDD, test_RDD = complete_ratings_data.randomSplit([7, 3], seed=0)
best_rank = train_small()
test_small(best_rank)


def train_complete_data():
    print("Training the recommender on complete dataset")

    complete_model = ALS.train(training_RDD, best_rank, seed=SEED,
                               iterations=ITERATIONS, lambda_=REG_PARAM)

    print("Testing the recommender on complete dataset")
    test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))

    predictions = complete_model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean())

    print('For complete testing datasets the RMSE is %s' % (error))


#test with complete data and new ratings
complete_data_with_new_ratings_RDD, new_user_ratings, complete_movies_data, complete_movies_titles, movie_rating_counts_RDD = get_complete_data_with_new_ratings()

def train_complete_data_new_ratings():
    t0 = time()
    new_ratings_model = ALS.train(complete_data_with_new_ratings_RDD, best_rank, seed=SEED,
                                  iterations=ITERATIONS, lambda_=REG_PARAM)
    tt = time() - t0

    print("New model trained in %s seconds" % round(tt, 3))


    #get the recommendation ratings for the items the new user hasn't rated so far
    new_user_ratings_ids = map(lambda x: x[1], new_user_ratings) # get just movie IDs
    # keep just those not on the ID list (thanks Lei Li for spotting the error!)
    new_user_unrated_movies_RDD = (complete_movies_data.filter(lambda x: x[0] not in new_user_ratings_ids).map(lambda x: (new_user_ID, x[0])))

    # Use the input RDD, new_user_unrated_movies_RDD, with new_ratings_model.predictAll() to predict new ratings for the movies
    new_user_recommendations_RDD = new_ratings_model.predictAll(new_user_unrated_movies_RDD)

    # Transform new_user_recommendations_RDD into pairs of the form (Movie ID, Predicted Rating)
    new_user_recommendations_rating_RDD = new_user_recommendations_RDD.map(lambda x: (x.product, x.rating))
    new_user_recommendations_rating_title_and_count_RDD = \
        new_user_recommendations_rating_RDD.join(complete_movies_titles).join(movie_rating_counts_RDD)
    new_user_recommendations_rating_title_and_count_RDD.take(3)

    new_user_recommendations_rating_title_and_count_RDD = \
        new_user_recommendations_rating_title_and_count_RDD.map(lambda r: (r[1][0][1], r[1][0][0], r[1][1]))


    top_movies = new_user_recommendations_rating_title_and_count_RDD.filter(lambda r: r[2]>=25).takeOrdered(25, key=lambda x: -x[1])

    print ('TOP recommended movies (with more than 25 reviews):\n%s' %
            '\n'.join(map(str, top_movies)))


    #getting individual ratings
    my_movie = spark.sparkContext.parallelize([(0, 500)]) # Quiz Show (1994)
    individual_movie_rating_RDD = new_ratings_model.predictAll(new_user_unrated_movies_RDD)
    individual_movie_rating_RDD.take(1)


