from pyspark import SparkContext
import sys
import time
import random
from itertools import combinations

def generate_hash_functions(n_functions, n_rows):
    primes = [100019, 100043, 100049, 100057]
    hash_params = [(random.randint(1, 1000), random.randint(0, 1000), random.choice(primes)) for _ in range(n_functions)]
    def hash_family(i):
        a, b, p = hash_params[i]
        return lambda x: (a * x + b) % p % n_rows
    return [hash_family(i) for i in range(n_functions)]

def jaccard_similarity(set1, set2):
    return len(set1.intersection(set2)) / len(set1.union(set2))

if __name__ == "__main__":
    sc = SparkContext.getOrCreate()
    sc.setLogLevel("ERROR")

    start_time = time.time()

    input_file, output_file = sys.argv[1], sys.argv[2]

    # Load data and exclude header
    raw_data = sc.textFile(input_file).filter(lambda x: "user_id" not in x).map(lambda x: x.split(','))
    
    # Create dictionaries mapping users and businesses to indices
    user_ids = raw_data.map(lambda x: x[0]).distinct().collect()
    business_ids = raw_data.map(lambda x: x[1]).distinct().collect()
    
    user_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    business_index = {business_id: idx for idx, business_id in enumerate(business_ids)}

    # Initialize the signature matrix
    num_hash_functions = 50
    signature_matrix = [[float('inf') for _ in business_ids] for _ in range(num_hash_functions)]
    hash_funcs = generate_hash_functions(num_hash_functions, len(user_ids))

    # Apply hash functions to each user for each business
    for row in raw_data.collect():
        user, business = row[0], row[1]
        if user in user_index:  # Check if user exists in user_index to avoid KeyError
            for i, hash_func in enumerate(hash_funcs):
                user_idx = user_index[user]
                business_idx = business_index[business]
                signature_matrix[i][business_idx] = min(signature_matrix[i][business_idx], hash_func(user_idx))

    # Find candidate pairs based on signature matrix bands
    candidate_pairs = set()
    for i in range(num_hash_functions):
        band = sc.parallelize(signature_matrix[i]).zipWithIndex().groupBy(lambda x: x[0] // (num_hash_functions / 10)).filter(lambda x: len(x[1]) > 1)
        for group in band.collect():
            pairs = combinations([business_ids[idx] for _, idx in group[1]], 2)
            for pair in pairs:
                candidate_pairs.add(tuple(sorted(pair)))

    # Compute Jaccard similarity for candidate pairs
    business_user_map = raw_data.map(lambda x: (x[1], set([x[0]]))).reduceByKey(lambda x, y: x.union(y)).collectAsMap()
    results = []
    for b1, b2 in candidate_pairs:
        users1, users2 = business_user_map[b1], business_user_map[b2]
        sim = jaccard_similarity(users1, users2)
        if sim >= 0.5:
            results.append((b1, b2, sim))

    # Write results to file
    with open(output_file, 'w') as f:
        f.write("business_id_1,business_id_2,similarity\n")
        for b1, b2, sim in sorted(results):
            f.write(f"{b1},{b2},{sim}\n")

    print(f"Execution time: {time.time() - start_time}")

