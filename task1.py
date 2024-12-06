from pyspark import SparkContext
from itertools import combinations
from collections import defaultdict
import time
import sys
def group_and_sort_itemsets(itemsets):
    grouped_itemsets = defaultdict(list)
    
    for itemset in itemsets:
        # Check if the itemset is a singleton 
        if isinstance(itemset, str):
            grouped_itemsets[1].append(itemset)
        else:
            grouped_itemsets[len(itemset)].append(itemset)
    
    # Sort lexicographically
    for size in grouped_itemsets:
        if size == 1:
            grouped_itemsets[size] = sorted(grouped_itemsets[size])
        else:
            
            grouped_itemsets[size] = sorted(grouped_itemsets[size], key=lambda x: tuple(map(str, x)))
    
    return grouped_itemsets

def format_itemset_for_output(itemsets):
    formatted_itemsets = []
    
    # Iterate
    for itemset in itemsets:
        if isinstance(itemset, str): 
            formatted_itemsets.append(f"{{'{itemset}'}}")
        else:  
            formatted = "{" + ",".join(f"'{item}'" for item in itemset) + "}"
            formatted_itemsets.append(formatted)
    
    return ",".join(formatted_itemsets)


# Hash function 
def hash_pair(pair):
    return hash(pair) % num_buckets

# local phase
def pcy_local(iterator, threshold):
    item_counts = defaultdict(int)
    buckets = defaultdict(int)

    local_baskets = list(iterator)
    max_len = max(len(basket) for basket in local_baskets)

    for basket in local_baskets:
        #print(f"Current basket: {basket}")
        for item in basket:
            #print(f"Type of item: {type(item)}, Item: {item}") 
            item_counts[item] += 1  # Single items as tuples for consistency
        for pair in combinations(sorted(set(basket)), 2):
            buckets[hash_pair(pair)] += 1
    
    # Identify frequent buckets
    frequent_buckets = set()
    for bucket, count in buckets.items():
        if count >= threshold:
            frequent_buckets.add(bucket)

    frequent_items = set(item for item, count in item_counts.items() if count >= threshold)
    frequent_pairs = set(pair for pair in combinations(frequent_items, 2) if hash_pair(pair) in frequent_buckets)

    # Find all frequent combinations
    all_frequent_combinations = frequent_items.union(frequent_pairs)
    for size in range(3, max_len+1):
        for combo in combinations(frequent_items, size):
            if all(subset in all_frequent_combinations for subset in combinations(combo, 
size-1)):
                all_frequent_combinations.add(combo)

    # Ensure all keys in item_counts are tuples to be consistent
    all_frequent_combinations = {tuple(sorted(item)) if isinstance(item, list) else item for item in all_frequent_combinations}

    print(f"All frequent combinations from local phase: {all_frequent_combinations}")
    return list(all_frequent_combinations)

def pcy_global(candidate_itemsets, baskets, threshold):
    global_item_counts = defaultdict(int)

    candidate_itemsets = [tuple([item]) if isinstance(item, str) else item for item in candidate_itemsets]
     
    # Count all candidate itemsets across all baskets
    for basket in baskets:
        for candidate in candidate_itemsets:
            if set(candidate).issubset(basket):
                global_item_counts[candidate] += 1
    
    frequent_itemsets = [itemset for itemset, count in global_item_counts.items() if count >= threshold]
    
    return frequent_itemsets

def main():
    sc = SparkContext(appName="PCYSONAlgorithm")
    global num_buckets
    num_buckets = 1000  

    case_number = int(sys.argv[1])
    support = int(sys.argv[2])
    input_file_path = sys.argv[3]
    output_file_path = sys.argv[4]

    start_time = time.time()

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
    
    num_partition = baskets.getNumPartitions()
    local_threshold = support / baskets.getNumPartitions()
    local_candidates = baskets.mapPartitions(lambda partition: pcy_local(partition, local_threshold)) \
                              .distinct() \
                              .collect()

    # Stage 2: Global phase
    global_frequent_itemsets = pcy_global(local_candidates, baskets.collect(), support)
    # Save the results to the output file
    with open(output_file_path, 'w') as file_out:
        file_out.write("Candidates:\n")
        grouped_candidates = group_and_sort_itemsets(local_candidates)
        for size in sorted(grouped_candidates):
            formatted_group = format_itemset_for_output(grouped_candidates[size])
            file_out.write(formatted_group + "\n")       
        file_out.write("\nFrequent Itemsets:\n")
        grouped2_candidates = group_and_sort_itemsets(global_frequent_itemsets)         
        for size in sorted(grouped2_candidates):
            formatted2_group = format_itemset_for_output(grouped2_candidates[size])
            file_out.write(formatted2_group + "\n")    


    sc.stop()

if __name__ == "__main__":
    main()
