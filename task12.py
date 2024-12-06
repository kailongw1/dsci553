from pyspark import SparkContext
from itertools import combinations
from collections import defaultdict
import time
import sys

# Hash function for the PCY algorithm
def hash_pair(pair):
    return hash(pair) % num_buckets

# Function for the PCY algorithm's local phase
def pcy_local(iterator, threshold):
    item_counts = defaultdict(int)
    buckets = defaultdict(int)
    
    local_baskets = list(iterator)  # Materialize the iterator into a list to reuse it
    # Generate all combinations up to the size of the largest basket
    max_len = max(len(basket) for basket in local_baskets)
    
    # Count itemsets and use a hash table to count the pairs
    for basket in local_baskets:
        for size in range(1, max_len + 1):
            for itemset in combinations(sorted(set(basket)), size):
                item_counts[itemset] += 1
                if size == 2:  # Only hash pairs for buckets
                    buckets[hash_pair(itemset)] += 1
    
    # Identify frequent buckets and candidate itemsets
    frequent_buckets = {bucket for bucket, count in buckets.items() if count >= threshold}
    candidate_itemsets = {item for item, count in item_counts.items() if count >= threshold}
    
    # Return all candidate itemsets
    return list(candidate_itemsets)


# Function for the PCY algorithm's global phase
def pcy_global(candidate_itemsets, baskets, threshold):
    global_item_counts = defaultdict(int)
    
    # Count all candidate itemsets across all baskets
    for basket in baskets:
        for candidate in candidate_itemsets:
            if set(candidate).issubset(basket):
                global_item_counts[candidate] += 1
    
    # Filter out the itemsets that do not meet the global threshold
    frequent_itemsets = [itemset for itemset, count in global_item_counts.items() if count >= threshold]
    
    return frequent_itemsets

def main():
    sc = SparkContext(appName="PCYSONAlgorithm")

    # Define the number of buckets for hashing
    global num_buckets
    num_buckets = 1000  # Adjust based on your dataset

    # Parse command line arguments
    case_number = int(sys.argv[1])
    support = int(sys.argv[2])
    input_file_path = sys.argv[3]
    output_file_path = sys.argv[4]

    # Start timing the execution
    start_time = time.time()

    # Read the data
    raw_data = sc.textFile(input_file_path)
    header = raw_data.first()
    data = raw_data.filter(lambda row: row != header)

    
    # Depending on the case, transform the data into baskets differently
    if case_number == 1:
        # Case 1: Baskets are users with their reviewed business IDs
        baskets = data.map(lambda x: (x.split(',')[0], x.split(',')[1])) \
                          .distinct() \
                          .groupByKey() \
                          .map(lambda x: list(x[1]))
    elif case_number == 2:
        # Case 2: Baskets are businesses with their reviewing user IDs
        baskets = data.map(lambda x: (x.split(',')[1], x.split(',')[0])) \
                          .distinct() \
                          .groupByKey() \
                          .map(lambda x: list(x[1]))

    # Apply the PCY algorithm using the SON framework
    # Stage 1: Local phase
    local_threshold = support / baskets.getNumPartitions()
    local_candidates = baskets.mapPartitions(lambda partition: pcy_local(partition, local_threshold)) \
                       .flatMap(lambda x: x) \
                       .distinct() \
                       .collect()
    
    # Stage 2: Global phase
    global_frequent_itemsets = pcy_global(local_candidates, baskets.collect(), support)

    # Save the results to the output file
    with open(output_file_path, 'w') as file_out:
        file_out.write("Candidates:\n")
        # Write the candidates from the local phase
        for candidate in sorted(local_candidates):
            file_out.write(str(candidate) + '\n')
        file_out.write("\nFrequent Itemsets:\n")
        # Write the frequent itemsets from the global phase
        for itemset in sorted(global_frequent_itemsets):
            file_out.write(str(itemset) + '\n')

    # Print the execution time
    duration = time.time() - start_time
    print(f"Duration: {duration}")

    sc.stop()

if __name__ == "__main__":
    main()
