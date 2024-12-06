from pyspark import SparkContext
from itertools import combinations
from collections import defaultdict
import time
import sys


def debug_print_partition(iterator):
    items = list(iterator)  # Convert iterator to list to print and reuse it
    for item in items:
        print(item)
    return iter(items)  # Convert the list back to an iterator to return
# Hash function for the PCY algorithm
def hash_pair(pair):
    return hash(pair) % num_buckets

# Function for the PCY algorithm's local phase
def pcy_local(iterator, threshold):
    item_counts = defaultdict(int)
    buckets = defaultdict(int)

    local_baskets = list(iterator)
    max_len = max(len(basket) for basket in local_baskets)

    # Hash pairs into buckets and count single items
    for basket in local_baskets:
        print(f"Current basket: {basket}")
        for item in basket:
            print(f"Type of item: {type(item)}, Item: {item}") 
            item_counts[item] += 1  # Single items as tuples for consistency
        for pair in combinations(sorted(set(basket)), 2):
            buckets[hash_pair(pair)] += 1
    
    # Identify frequent buckets
    frequent_buckets = set()
    for bucket, count in buckets.items():
        if count >= threshold:
            frequent_buckets.add(bucket)

    # Identify frequent singletons and pairs
    frequent_items = set(item for item, count in item_counts.items() if count >= threshold)
    frequent_pairs = set(pair for pair in combinations(frequent_items, 2) if hash_pair(pair) in frequent_buckets)

    # Find all frequent combinations
    all_frequent_combinations = frequent_items.union(frequent_pairs)
    for size in range(3, max_len+1):
        for combo in combinations(frequent_items, size):
            if all(subset in all_frequent_combinations for subset in combinations(combo, size-1)):
                all_frequent_combinations.add(combo)

    # Ensure all keys in item_counts are tuples to be consistent
    all_frequent_combinations = {tuple(sorted(item)) if isinstance(item, list) else item for item in all_frequent_combinations}

    print(f"All frequent combinations from local phase: {all_frequent_combinations}")
    return list(all_frequent_combinations)

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
    # Inside the main function, after the baskets are defined
    if case_number == 1:
        # Case 1: Baskets are users with their reviewed business IDs
        baskets = data.map(lambda x: (x.split(',')[0], x.split(',')[1])) \
                      .distinct() \
                      .groupByKey() \
                      .map(lambda x: list(x[1]))
    # Debug print the first few baskets
        for basket in baskets.take(19):
            print(f"Current basket: {basket}")

    elif case_number == 2:
        # Case 2: Baskets are businesses with their reviewing user IDs
        baskets = data.map(lambda x: (x.split(',')[1], x.split(',')[0])) \
                      .distinct() \
                      .groupByKey() \
                      .map(lambda x: list(x[1])) 
    
    # Apply the PCY algorithm using the SON framework
    # Stage 1: Local phase
    # Apply debug_print_partition function in mapPartitions for debugging
    debug_rdd = baskets.mapPartitions(debug_print_partition)

    # Trigger an action to print the items, such as collect() or take()
    print('here is the debug output:')
    print(debug_rdd.take(10))
 
    local_threshold = support / baskets.getNumPartitions()
    local_candidates = baskets.mapPartitions(lambda partition: pcy_local(partition, local_threshold)) \
                              .flatMap(lambda x: x) \
                              .distinct() \
                              .collect()
    print(local_threshold)
    #print(local_candidates)
    
    # Stage 2: Global phase
    global_frequent_itemsets = pcy_global(local_candidates, baskets.collect(), support)

    # Prepare the candidates and frequent itemsets for output
    local_candidates = sorted(local_candidates, key=lambda x: (len(x), x))
    global_frequent_itemsets = sorted(global_frequent_itemsets, key=lambda x: (len(x), x))

    # Save the results to the output file
    with open(output_file_path, 'w') as file_out:
        file_out.write("Candidates:\n")
        for candidate in local_candidates:
            line = ",".join(str(i) for i in candidate)
            file_out.write(f"({line})\n")
        file_out.write("\nFrequent Itemsets:\n")
        for itemset in global_frequent_itemsets:
            line = ",".join(str(i) for i in itemset)
            file_out.write(f"({line})\n")


    # Print the execution time
    duration = time.time() - start_time
    print(f"Duration: {duration}")

    sc.stop()

if __name__ == "__main__":
    main()

