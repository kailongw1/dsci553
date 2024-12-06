from pyspark import SparkContext
from itertools import combinations
from collections import defaultdict
import time
import sys

def preprocess_line(line):
    parts = line.split(',')
    date = parts[0].strip('"')
    customer_id = parts[1].strip('"').lstrip('0')  
    product_id = parts[5].strip('"').lstrip('0')  
    product_id_int = int(product_id) if product_id.isdigit() else 0  
    date_customer_id = f"{date}-{customer_id}"  # Combine date and customer ID
    return (date_customer_id, product_id_int)

def group_and_sort_itemsets(itemsets):
    
    grouped_itemsets = defaultdict(list)
    
    for itemset in itemsets:
        if isinstance(itemset, str):
            # Single items
            grouped_itemsets[1].append(itemset)
        else:
            grouped_itemsets[len(itemset)].append(itemset)
    
    for size in grouped_itemsets:
        if size == 1:
            
            grouped_itemsets[size] = sorted(grouped_itemsets[size])
        else:
            grouped_itemsets[size] = sorted(grouped_itemsets[size], key=lambda x: tuple(map(str, x)))
    
    return grouped_itemsets

def format_itemset_for_output(itemsets):
    formatted_itemsets = []
    
    for itemset in itemsets:
        if isinstance(itemset, str): 
            formatted_itemsets.append(f"{{'{itemset}'}}")
        else:  
            formatted = "{" + ",".join(f"'{item}'" for item in itemset) + "}"
            formatted_itemsets.append(formatted)

    return ",".join(formatted_itemsets)


def hash_pair(pair):
    return hash(pair) % num_buckets

#local phase
def pcy_local(iterator, threshold):
    item_counts = defaultdict(int)
    buckets = defaultdict(int)

    local_baskets = list(iterator)
    max_len = max(len(basket) for basket in local_baskets)

    for basket in local_baskets:
        for item in basket:

            item_counts[item] += 1
        for pair in combinations(sorted(set(basket)), 2):
            buckets[hash_pair(pair)] += 1
    
    frequent_buckets = set()
    for bucket, count in buckets.items():
        if count >= threshold:
            frequent_buckets.add(bucket)

    frequent_items = set(item for item, count in item_counts.items() if count >= threshold)
    frequent_pairs = set(pair for pair in combinations(frequent_items, 2) if hash_pair(pair) in frequent_buckets)

    all_frequent_combinations = frequent_items.union(frequent_pairs)
    for size in range(3, max_len+1):
        for combo in combinations(frequent_items, size):
            if all(subset in all_frequent_combinations for subset in combinations(combo, size-1)):
                all_frequent_combinations.add(combo)

    # Ensure all keys in item_counts are tuples to be consistent
    #all_frequent_combinations = {tuple(sorted(item)) if isinstance(item, list) else item for item in all_frequent_combinations}

    print(f"All frequent combinations from local phase: {all_frequent_combinations}")
    return list(all_frequent_combinations)

def pcy_global(candidate_itemsets, baskets, threshold):
    global_item_counts = defaultdict(int)

    # Ensure singletons are treated as tuples for consistent processing
    candidate_itemsets = [tuple([item]) if isinstance(item, str) else item for item in candidate_itemsets]
     
    for basket in baskets:
        for candidate in candidate_itemsets:
            if set(candidate).issubset(basket):
                global_item_counts[candidate] += 1
    
    frequent_itemsets = [itemset for itemset, count in global_item_counts.items() if count >= threshold]
    
    return frequent_itemsets



def main():
    sc = SparkContext(appName="PCYSONAlgorithmTask2")

    global num_buckets
    num_buckets = 1000
    
    filter_threshold = int(sys.argv[1])
    support = int(sys.argv[2])
    input_file_path = sys.argv[3]
    output_file_path = sys.argv[4]

    start_time = time.time()

    raw_data = sc.textFile(input_file_path)
    header = raw_data.first()
    processed_data = raw_data.filter(lambda line: line != header).map(preprocess_line)

    # Group by 'DATE-CUSTOMER_ID'
    baskets = processed_data.groupByKey() \
                            .mapValues(set) \
                            .filter(lambda kv: len(kv[1]) > filter_threshold) \
                            .map(lambda kv: kv[1])  # Keep only the product sets

    # After defining baskets RDD as shown in your snippet
    baskets_collected = baskets.collect()  # Collects the data from RDD to a list

    for basket in baskets_collected:
        print(basket)
        
    num_partitions_desired = 100
    baskets_repartitioned = baskets.repartition(num_partitions_desired)
    local_threshold = support / num_partitions_desired
    local_candidates = baskets_repartitioned.mapPartitions(lambda partition: pcy_local(partition, local_threshold)) \
                              .distinct() \
                              .collect()

    #Global check
    global_frequent_itemsets = pcy_global(local_candidates, baskets.collect(), support)
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
    

    duration = time.time() - start_time
    print(f"Duration: {duration}")

    sc.stop()

if __name__ == "__main__":
    main()

