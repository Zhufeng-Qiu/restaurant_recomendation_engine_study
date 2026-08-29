from pyspark import SparkContext, SparkConf
import sys
import json
import time
import itertools

#'''
input_file = "/Users/zephyryau/Documents/study/INF553/2020-summer/Homework/Assignment3/python/data/train_review.json"
output_file = "/Users/zephyryau/Documents/study/INF553/2020-summer/Homework/Assignment3/python/result/task1"
'''
input_file = sys.argv[1]
output_file = sys.argv[2]
'''

start_time = time.time()
conf = SparkConf().setAppName("inf553-HW3-task1").setMaster("local[*]").setAll((("spark.executor.memory", "4g"), ("spark.driver.memory", "4g")))
sc = SparkContext(conf=conf)

input_data = sc.textFile(input_file).map(lambda line:json.loads(line)).map(lambda line: (line["user_id"], line["business_id"])).persist()
user_data = input_data.map(lambda x: x[0]).distinct().collect()
m = len(user_data)
number_hash_func = 100
row = 1
band = int(number_hash_func / row)
#business_data = input_data.map(lambda x: x[1]).distinct()
#business_users = input_data.map(lambda x: (x[1], [x[0]])).reduceByKey(lambda a, b: a + b).persist()

user_idx = {}
i = 0
for user in user_data:
    user_idx[user] = user_idx.get(user, i)
    i += 1

a1 = [5003,5009,5011,5021,5023,5039,5051,5059,5077,5081,5087,5099,5101,5107,5113,5119,5147,5153,5167,5171,5179,5189,5197,5209,5227,5231,5233,5237,5261,5273,5279,5281,5297,5303,5309,5323,5333,5347,5351,5381,5387,5393,5399,5407,5413,5417,5419,5431,5437,5441,5443,5449,5471,5477,5479,5483,5501,5503,5507,5519,5521,5527,5531,5557,5563,5569,5573,5581,5591,5623,5639,5641,5647,5651,5653,5657,5659,5669,5683,5689,5693,5701,5711,5717,5737,5741,5743,5749,5779,5783,5791,5801,5807,5813,5821,5827,5839,5843,5849,5851]
a2 = [9127,9133,9137,9151,9157,9161,9173,9181,9187,9199,9203,9209,9221,9227,9239,9241,9257,9277,9281,9283,9293,9311,9319,9323,9337,9341,9343,9349,9371,9377,9391,9397,9403,9413,9419,9421,9431,9433,9437,9439,9461,9463,9467,9473,9479,9491,9497,9511,9521,9533,9539,9547,9551,9587,9601,9613,9619,9623,9629,9631,9643,9649,9661,9677,9679,9689,9697,9719,9721,9733,9739,9743,9749,9767,9769,9781,9787,9791,9803,9811,9817,9829,9833,9839,9851,9857,9859,9871,9883,9887,9901,9907,9923,9929,9931,9941,9949,9967,9973,10007]
b1 = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313,317,331,337,347,349,353,359,367,373,379,383,389,397,401,409,419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,509,521,523,541]
b2 = [1009,1013,1019,1021,1031,1033,1039,1049,1051,1061,1063,1069,1087,1091,1093,1097,1103,1109,1117,1123,1129,1151,1153,1163,1171,1181,1187,1193,1201,1213,1217,1223,1229,1231,1237,1249,1259,1277,1279,1283,1289,1291,1297,1301,1303,1307,1319,1321,1327,1361,1367,1373,1381,1399,1409,1423,1427,1429,1433,1439,1447,1451,1453,1459,1471,1481,1483,1487,1489,1493,1499,1511,1523,1531,1543,1549,1553,1559,1567,1571,1579,1583,1597,1601,1607,1609,1613,1619,1621,1627,1637,1657,1663,1667,1669,1693,1697,1699,1709,1721]

def h1(x, i):
    global a1
    global b1
    return a1[i] * x + b1[i]

def h2(x, i):
    global a2
    global b2
    return a2[i] * x + b2[i]

def min_hash(x):
    global m
    global number_hash_func
    p = 24251
    min_hash_values = []
    for itr in range(number_hash_func):
        intermediate_values = []
        for i, id in enumerate(x[1]):
            i_i = i + 1
            itr1 = itr+1
            v = ((itr1 * h1(id, itr) + itr1 * h2(id, itr) + itr1 * itr1) % p) % m
            intermediate_values.append(v)
        min_hash_values.append(min(intermediate_values))
    return (x[0], min_hash_values)

def band_hash(v_list):
    sum = 0
    p = 131
    for v in v_list:
            sum += v * p
    return sum & 0x7FFFFFFF

def divide_signature(x):
    global row
    global band
    result = []
    list_idx = 0
    for i in range(0, band, row):
        result.append(((i, band_hash(x[1][list_idx: list_idx + row])), [x[0]]))
        list_idx += row
    return result

def jaccard_similarity(pair):
    global business_dict
    pair_1 = business_dict[pair[0]]
    pair_2 = business_dict[pair[1]]
    return len(pair_1.intersection(pair_2)) / len(pair_1.union(pair_2))

characteristic_matrix = input_data.map(lambda x: (x[1], [user_idx[x[0]]])).reduceByKey(lambda a, b: a + b).mapValues(lambda x: set(x)).persist()
business_dict = characteristic_matrix.collectAsMap()

signature_matrix = characteristic_matrix.map(lambda x: min_hash(x))

candidate_pair = signature_matrix.flatMap(lambda x: divide_signature(x)).reduceByKey(lambda a, b: a + b).filter(lambda x: len(x[1]) > 1).flatMap(lambda x: [cand_pair for cand_pair in itertools.combinations(x[1], 2)]).distinct()

#for i in range(len(candidate_pair.take(20))):
#    print("pair: ", i)
#    print(business_dict[candidate_pair.take(20)[i][0]])
#    print(business_dict[candidate_pair.take(20)[i][1]])
similar_pair = candidate_pair.map(lambda pair: (pair, jaccard_similarity(pair))).filter(lambda x: x[1] >= 0.05).map(lambda pair: {"b1": pair[0][0], "b2": pair[0][1], "sim": pair[1]}).collect()

with open(output_file, "w") as output_f:
    for element in similar_pair:
        json.dump(element, output_f)
        output_f.write('\n')

end_time = time.time()
print("Duration: ", end_time - start_time)
