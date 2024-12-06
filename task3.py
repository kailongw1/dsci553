import json
import sys
import time
from pyspark import SparkContext

def average_stars_city(review_rdd, business_rdd):
    #Join reviewRDD and businessRDD on business_id
    joined_rdd = review_rdd.map(lambda review: (review['business_id'], review['stars'])).join(business_rdd.map(lambda business:(business['business_id'], business['city'])))
    
    #Calculate the average stars for each city
    city_stars = joined_rdd.map(lambda x: (x[1][1], (x[1][0], 1))).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])).mapValues(lambda total: total[0] / total[1])
    return city_stars

def sort_cities_in_python(averages):
    collected_avg = averages.collect()
    return sorted(collected_avg, key=lambda x: (-x[1], x[0]))

def sort_cities_in_spark(averages):
    return averages.takeOrdered(10, key=lambda x: (-x[1], x[0]))

def implementation(review_filepath, business_filepath, output_filepath_question_a, output_filepath_question_b):
    sc = SparkContext(appName="multi_datasets_exploration")
    start_load_time = time.time()
    reviews = sc.textFile(review_filepath).map(lambda x: json.loads(x))
    businesses = sc.textFile(business_filepath).map(lambda x: json.loads(x))
    load_time = time.time() - start_load_time

    avg_stars_per_city = average_stars_city(reviews, businesses)

    #Task 3A: Save results as a text file
    sorted_averages = sort_cities_in_spark(avg_stars_per_city)
    with open(output_filepath_question_a, 'w') as output_a:
        output_a.write("city,stars\n")
        for city, stars in sorted_averages:
            output_a.write(f"{city},{stars}\n")

    #Compare execution times for Method1 and Method2
    #Python
    start_m1_time = time.time()
    sorted_averages_m1 = sort_cities_in_python(avg_stars_per_city)
    m1_time = time.time() - start_m1_time + load_time

    #Spark
    start_m2_time = time.time()
    sorted_averages_m2 = sort_cities_in_spark(avg_stars_per_city)
    m2_time = time.time() - start_m2_time + load_time

    results_b = {
        "m1": m1_time,
        "m2": m2_time,
        "reason": "M2 might be faster for larger datasets because of distributed computing. On the other hand, M1 could be faster for small datasets due to lower overhead."
    }
    with open(output_filepath_question_b, 'w') as output_b:
        json.dump(results_b, output_b, indent=4)

    sc.stop()

review_filepath = sys.argv[1]
business_filepath = sys.argv[2]
output_filepath_question_a = sys.argv[3]
output_filepath_question_b = sys.argv[4]
implementation(review_filepath, business_filepath, output_filepath_question_a, output_filepath_question_b)

