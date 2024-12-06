#import necessary packages from python standard library
import json
import sys
from pyspark import SparkContext

def parse_review(line):
    return json.loads(line)
   
def review_2018(review):
    return review['date'].startswith('2018')

def implementation(input_file, output_file):
    #RDD
    sc = SparkContext(appName="data_exploration")
    reviews = sc.textFile(input_file).map(parse_review).filter(lambda x: x is not None)

    #Apply different transformetions and applications to Spark RDD
    #A.Total number of reviews
    total_reviews = reviews.count()
    
    #B.Number of reviews in 2018
    reviews_2018 = reviews.filter(review_2018).count()
    
    #C.Number of distinct users
    distinct_users = reviews.map(lambda review: review['user_id']).distinct().count()
    
    #D.Top 10 users by number of reviews
    top_users = reviews.map(lambda review: (review['user_id'], 1)).reduceByKey(lambda x, y: x + y).takeOrdered(10, key=lambda x: (-x[1], x[0]))

    #E.Number of distinct businesses
    distinct_businesses = reviews.map(lambda review: review['business_id']).distinct().count()
    
    #F.Top 10 businesses by number of reviews
    top_businesses = reviews.map(lambda review: (review['business_id'], 1)).reduceByKey(lambda x, y: x + y).takeOrdered(10, key=lambda x: (-x[1], x[0]))

    #format the outputs specifically according to the pdf
    top10_user_formatted = [["{}: {}".format(user_id, count)] for user_id, count in top_users]
    top10_business_formatted = [["{}: {}".format(business_id, count)] for business_id, count in top_businesses]

    #Output 
    results = {
        "n_review": total_reviews,
        "n_review_2018": reviews_2018,
        "n_user": distinct_users,
        "top10_user": top10_user_formatted,
        "n_business": distinct_businesses,
        "top10_business": top10_business_formatted
    }
    
    with open(output_file, 'w') as outputfile:
        json.dump(results, outputfile, indent=4)
        
    sc.stop()


#get command inputs
input_file = sys.argv[1]
output_file = sys.argv[2]
implementation(input_file, output_file)

