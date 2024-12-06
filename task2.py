import json
import sys
import time
from pyspark import SparkContext

def custom_partition(key):
    return hash(key)

def implementation(input_file, output_file, n_partition):
    sc = SparkContext(appName="partition_optimization")
    reviews = sc.textFile(input_file).map(lambda x: json.loads(x))

    #Task1 Q F
    business_reviews = reviews.map(lambda review: (review['business_id'], 1)).reduceByKey(lambda x, y: x + y)
    
    # Default partition
    default_n_partition = business_reviews.getNumPartitions()
    default_items_per_partition = business_reviews.glom().map(len).collect()
    
    # Execution time 
    start_time = time.time()
    top_businesses_default = business_reviews.takeOrdered(10, key=lambda x: (-x[1], x[0]))
    default_time = time.time() - start_time
    
    # Custom partition
    custom_business_reviews = business_reviews.partitionBy(n_partition, custom_partition)
    
    # Details
    custom_n_partition = custom_business_reviews.getNumPartitions()
    custom_items_per_partition = custom_business_reviews.glom().map(len).collect()
    start_time = time.time()
    top_businesses_custom = custom_business_reviews.takeOrdered(10, key=lambda x: (-x[1], x[0]))
    custom_time = time.time() - start_time
    
    # Output
    results = {
        "default": {
            "n_partition": default_n_partition,
            "n_items": default_items_per_partition,
            "exe_time": default_time
        },
        "customized": {
            "n_partition": custom_n_partition,
            "n_items": custom_items_per_partition,
            "exe_time": custom_time
        }
    }
    
    with open(output_file, 'w') as outputfile:
        json.dump(results, outputfile, indent=4)
    sc.stop()


input_file = sys.argv[1]
output_file = sys.argv[2]
n_partition = int(sys.argv[3])
implementation(input_file, output_file, n_partition)

