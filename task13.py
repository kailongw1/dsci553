from pyspark import SparkContext
import sys
import random
from itertools import combinations
from hashlib import sha256

def create_hash_functions(num_funcs, num_buckets):
    hash_funcs = []
    for _ in range(num_funcs):
        a, b = random.randint(1, 1000), random.randint(0, 500)
        hash_funcs.append(lambda x, a=a, b=b: (a * x + b) % num_buckets)
    return hash_funcs

def jaccard_sim(biz_set1, biz_set2):
    intersection = len(biz_set1.intersection(biz_set2))
    union = len(biz_set1.union(biz_set2))
    return intersection / union if union else 0

def encode_business(business):
    return int(sha256(business.encode('utf-8')).hexdigest(), 16)

if __name__ == "__main__":
    SparkContext.getOrCreate()
    input_path, output_path = sys.argv[1:3]

    data = sc.textFile(input_path)
    header = data.first()
    records = data.filter(lambda line: line != header).map(lambda line: 
line.split(','))

    business_users = records.map(lambda x: (x[1], set([x[0]]))).reduceByKey(lambda 
x, y: x.union(y))
    user_business = records.map(lambda x: (x[0], [x[1]])).reduceByKey(lambda x, y: 
x + y).collectAsMap()

    num_buckets = business_users.count() * 2
    hash_functions = create_hash_functions(20, num_buckets)

    signatures = business_users.map(lambda x: (x[0], 
min([min((func(encode_business(user)) for user in x[1])) for func in 
hash_functions])))

    candidate_pairs = signatures.flatMap(lambda x: [(hash_func(x[1]), x[0]) for 
hash_func in hash_functions])\
                                .groupByKey()\
                                .filter(lambda x: len(x[1]) > 1)\
                                .flatMap(lambda x: combinations(sorted(list(x[1])), 
2))\
                                .distinct()

    similar_pairs = candidate_pairs.map(lambda pair: (pair, 
jaccard_sim(user_business[pair[0]], user_business[pair[1]])))\
                                    .filter(lambda pair_sim: pair_sim[1] >= 0.5)\
                                    .collect()
   

    with open(output_path, 'w') as file:
        file.write("business_id_1,business_id_2,similarity\n")
        for pair, sim in similar_pairs:
            file.write(f"{pair[0]},{pair[1]},{sim}\n")

    print(f"Script executed and output saved to {output_path}.")

