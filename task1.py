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

    raw_data = sc.textFile(input_file).filter(lambda x: "user_id" not in x).map(lambda x: x.split(','))
    user_business = raw_data.map(lambda x: (x[0], x[1])).groupByKey().mapValues(set).collectAsMap()

    businesses = raw_data.map(lambda x: x[1]).distinct().collect()
    business_index = {business: idx for idx, business in enumerate(businesses)}

    users = raw_data.map(lambda x: x[0]).distinct().collect()
    user_index = {user: idx for idx, user in enumerate(users)}

    num_hash_functions = 50
    signature_matrix = [[float('inf')] * len(user_index) for _ in range(num_hash_functions)]
    hash_funcs = generate_hash_functions(num_hash_functions, len(user_index))

    for business, user_set in user_business.items():
        for i, hash_func in enumerate(hash_funcs):
            for user in user_set:
                signature_matrix[i][business_index[business]] = min(signature_matrix[i][business_index[business]], hash_func(user_index[user]))

    candidate_pairs = set()
    for i in range(num_hash_functions):
        band = sc.parallelize(signature_matrix[i]).zipWithIndex().groupBy(lambda x: x[0] // (num_hash_functions / 10)).filter(lambda x: len(x[1]) > 1)
        for group in band.collect():
            pairs = combinations([businesses[idx] for _, idx in group[1]], 2)
            for pair in pairs:
                candidate_pairs.add(tuple(sorted(pair)))

    results = []
    for b1, b2 in candidate_pairs:
        sim = jaccard_similarity(user_business[b1], user_business[b2])
        if sim >= 0.5:
            results.append((b1, b2, sim))

    with open(output_file, 'w') as f:
        f.write("business_id_1,business_id_2,similarity\n")
        for b1, b2, sim in sorted(results):
            f.write(f"{b1},{b2},{sim}\n")

    print(f"Execution time: {time.time() - start_time}")

