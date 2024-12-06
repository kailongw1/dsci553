from pyspark import SparkContext
import sys
import time
import random
from itertools import combinations

def next_prime(n):
    """Return the next prime number after n."""
    def is_prime(k):
        if k < 2:
            return False
        for i in range(2, int(k ** 0.5) + 1):
            if k % i == 0:
                return False
        return True

    while True:
        n += 1
        if is_prime(n):
            return n

def hash_func(num_buckets):
    """Generate a list of hash functions."""
    prime = next_prime(2 * num_buckets)
    a = random.randint(1, prime - 1)
    b = random.randint(0, prime - 1)
    return lambda x: ((a * x + b) % prime) % num_buckets

def calculate_jaccard_similarity(set1, set2):
    """Calculate the Jaccard similarity between two sets."""
    intersection_size = len(set1.intersection(set2))
    union_size = len(set1.union(set2))
    return intersection_size / union_size

def main(input_file, output_file):
    sc = SparkContext.getOrCreate()
    sc.setLogLevel("ERROR")

    data = sc.textFile(input_file).filter(lambda x: "user_id" not in x).map(lambda x: x.split(','))
    users = data.map(lambda x: x[0]).distinct().zipWithIndex().collectAsMap()
    businesses = data.map(lambda x: (x[1], set([x[0]]))).reduceByKey(lambda x, y: x.union(y)).collect()

    num_buckets = len(users)
    num_hash_functions = 50
    hash_functions = [hash_func(num_buckets) for _ in range(num_hash_functions)]

    # Generate signature matrix
    signatures = {}
    for business_id, user_set in businesses:
        business_signature = []
        for hash_function in hash_functions:
            min_hash = min([hash_function(users[user_id]) for user_id in user_set])
            business_signature.append(min_hash)
        signatures[business_id] = business_signature

    # Find candidate pairs
    candidates = set()
    for biz1, sig1 in signatures.items():
        for biz2, sig2 in signatures.items():
            if biz1 < biz2:  # Ensure unique pairs and avoid comparing a business with itself
                similarity_estimate = sum(1 for i, j in zip(sig1, sig2) if i == j) / num_hash_functions
                if similarity_estimate >= 0.5:  # Threshold can be adjusted based on requirements
                    candidates.add((biz1, biz2))

    # Calculate actual Jaccard similarities for candidate pairs
    business_users = dict(businesses)
    results = []
    for biz1, biz2 in candidates:
        actual_similarity = calculate_jaccard_similarity(business_users[biz1], business_users[biz2])
        if actual_similarity >= 0.5:
            results.append((biz1, biz2, actual_similarity))

    # Save the results to file
    with open(output_file, 'w') as f:
        f.write("business_id_1,business_id_2,similarity\n")
        for biz1, biz2, sim in sorted(results):
            f.write(f"{biz1},{biz2},{sim}\n")

    print(f"Completed. Results saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: spark-submit script.py <input_file> <output_file>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])

