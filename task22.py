import json
import time
from pyspark import SparkContext

def custom_partition(review_text_length):
    # Partitioning based on the length of the review text
    if review_text_length < 50:
        return 0  # Short reviews
    elif review_text_length < 100:
        return 1  # Medium reviews
    else:
        return 2  # Long reviews

def main(input_file, output_file, n_partition):
    sc = SparkContext(appName="PartitionOptimizationTask2")
    reviews = sc.textFile(input_file).map(lambda x: json.loads(x))

    # Task 1 Question F logic - Top 10 businesses by number of reviews
    business_reviews = reviews.map(lambda review: (review['business_id'], 
1)).reduceByKey(lambda a, b: a + b)
    top_businesses = business_reviews.takeOrdered(10, key=lambda x: (-x[1], x[0]))

    # Default partition details
    default_n_partition = business_reviews.getNumPartitions()
    default_items_per_partition = business_reviews.glom().map(len).collect()

    # Execution time with default partition
    start_time = time.time()
    # Action to trigger the computation for the default partition
    default_top_businesses = business_reviews.takeOrdered(10, key=lambda x: (-x[1], 
x[0]))
    default_exe_time = time.time() - start_time

    # Custom partitioning - partition by the length of the review text
    review_text_lengths = reviews.map(lambda review: len(review['text']))
    custom_reviews = review_text_lengths.map(lambda length: (length, 
1)).partitionBy(n_partition, custom_partition)

    # Custom partition details
    custom_n_partition = custom_reviews.getNumPartitions()
    custom_items_per_partition = custom_reviews.glom().map(len).collect()

    # Execution time with custom partition
    start_time = time.time()
    # Action to trigger the computation for the custom partition
    # Note: Here we need to use the business_reviews RDD with the new partitioning
    custom_business_reviews = business_reviews.partitionBy(n_partition, custom_partition)
    custom_top_businesses = custom_business_reviews.takeOrdered(10, key=lambda x: (-x[1], 
x[0]))
    custom_exe_time = time.time() - start_time

    # Output results
    results = {
        "default": {
            "n_partition": default_n_partition,
            "n_items": default_items_per_partition,
            "exe_time": default_exe_time
        },
        "customized": {
            "n_partition": custom_n_partition,
            "n_items": custom_items_per_partition,
            "exe_time": custom_exe_time
        }
    }

    with open(output_file, 'w') as outfile:
        json.dump(results, outfile, indent=4)

    sc.stop()

if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    n_partition = int(sys.argv[3])  # The number of partitions to be used in the custom partitioning
    main(input_file, output_file, n_partition)

