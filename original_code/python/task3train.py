from pyspark import SparkContext, SparkConf
import sys
import json
import time
import math
import itertools

start_time = time.time()
#'''
#train_file = "/Users/zephyryau/Documents/study/INF553/2020-summer/Homework/Assignment3/python/data/test/test_data.json"
train_file = "/Users/zephyryau/Documents/study/INF553/2020-summer/Homework/Assignment3/python/data/train_review.json"
model_file = "/Users/zephyryau/Documents/study/INF553/2020-summer/Homework/Assignment3/python/result/task2_model_user"
cf_type = "user_based"
#cf_type = "item_based"
'''
train_file = sys.argv[1]
model_file = sys.argv[2]
cf_type = sys.argv[3]
'''

co_rated_threshold = 3
conf = SparkConf().setAppName("inf553-HW3-task3-train").setMaster("local[*]").setAll((("spark.executor.memory", "4g"), ("spark.driver.memory", "4g")))
sc = SparkContext(conf=conf)


input_data = sc.textFile(train_file).map(lambda line: json.loads(line)).map(lambda line: (line["user_id"], line["business_id"], line["stars"])).persist()
user_data = input_data.map(lambda x: x[0]).distinct().collect()
business_data = input_data.map(lambda x: x[1]).distinct().collect()
user_idx = {}
i = 0
for user in user_data:
    user_idx[user] = user_idx.get(user, i)
    i += 1

business_idx = {}
i = 0
for business in business_data:
    business_idx[business] = business_idx.get(business, i)
    i += 1

def list2dict(l):
    l_dict = {}
    for item in l:
        l_dict[item[0]] = item[1]
    return l_dict

def item_pearson_similarity(pair):
    users_i = business_users_dict[pair[0]]
    users_j = business_users_dict[pair[1]]
    co_rated_users = set(users_i.keys()).intersection(set(users_j.keys()))
    business_i = [users_i[user] for user in co_rated_users]
    business_j = [users_j[user] for user in co_rated_users]

    r_i = sum(business_i) / len(business_i)
    r_j = sum(business_j) / len(business_j)

    tmp_1 = [(r_ui - r_i) for r_ui in business_i]
    tmp_2 = [(r_uj - r_j) for r_uj in business_j]
    tmp_3 = [(r_ui - r_i)**2 for r_ui in business_i]
    tmp_4 = [(r_uj - r_j)**2 for r_uj in business_j]

    number = sum([x * y for x, y in zip(tmp_1, tmp_2)])
    denominator = (math.sqrt(sum(tmp_3)) * math.sqrt(sum(tmp_4)))
    if (number == 0) or (denominator == 0):
        return 0
    else:
        similarity = number / denominator
    return similarity

def user_pearson_similarity(pair):
    items_i = user_bussiness_dict[pair[0]]
    items_j = user_bussiness_dict[pair[1]]
    co_rated_bussiness = set(items_i.keys()).intersection(set(items_j.keys()))
    business_i = [items_i[user] for user in co_rated_bussiness]
    business_j = [items_j[user] for user in co_rated_bussiness]

    r_i = sum(business_i) / len(business_i)
    r_j = sum(business_j) / len(business_j)

    tmp_1 = [(r_ui - r_i) for r_ui in business_i]
    tmp_2 = [(r_uj - r_j) for r_uj in business_j]
    tmp_3 = [(r_ui - r_i)**2 for r_ui in business_i]
    tmp_4 = [(r_uj - r_j)**2 for r_uj in business_j]

    number = sum([x * y for x, y in zip(tmp_1, tmp_2)])
    denominator = (math.sqrt(sum(tmp_3)) * math.sqrt(sum(tmp_4)))
    if (number == 0) or (denominator == 0):
        return 0
    else:
        similarity = number / denominator
    return similarity

a1 = [5003,5009,5011,5021,5023,5039,5051,5059,5077,5081,5087,5099,5101,5107,5113,5119,5147,5153,5167,5171,5179,5189,5197,5209,5227,5231,5233,5237,5261,5273,5279,5281,5297,5303,5309,5323,5333,5347,5351,5381,5387,5393,5399,5407,5413,5417,5419,5431,5437,5441]
a2 = [9127,9133,9137,9151,9157,9161,9173,9181,9187,9199,9203,9209,9221,9227,9239,9241,9257,9277,9281,9283,9293,9311,9319,9323,9337,9341,9343,9349,9371,9377,9391,9397,9403,9413,9419,9421,9431,9433,9437,9439,9461,9463,9467,9473,9479,9491,9497,9511,9521,9533]
b1 = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229]
b2 = [1009,1013,1019,1021,1031,1033,1039,1049,1051,1061,1063,1069,1087,1091,1093,1097,1103,1109,1117,1123,1129,1151,1153,1163,1171,1181,1187,1193,1201,1213,1217,1223,1229,1231,1237,1249,1259,1277,1279,1283,1289,1291,1297,1301,1303,1307,1319,1321,1327,1361]

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
    global user_dict
    pair_1 = user_dict[pair[0]]
    pair_2 = user_dict[pair[1]]
    return len(pair_1.intersection(pair_2)) / len(pair_1.union(pair_2))


if cf_type == "item_based":
    business_baskets = input_data.map(lambda x: (x[1], [(x[0], x[2])])).reduceByKey(lambda a, b: a + b).filter(lambda x: len(x[1]) >= co_rated_threshold).mapValues(lambda x: list2dict(x)).persist()
    business_users_dict = business_baskets.collectAsMap()
    intermediate_business_id = business_baskets.map(lambda x: x[0])
    item_based_result = intermediate_business_id.cartesian(intermediate_business_id).filter(lambda pair: pair[0] < pair[1]).filter(lambda pair: len(set(business_users_dict[pair[0]]).intersection(set(business_users_dict[pair[1]]))) >= co_rated_threshold).map(lambda pair: {"b1": pair[0], "b2": pair[1], "sim": item_pearson_similarity(pair)}).filter(lambda x: x["sim"] > 0).collect()
    with open(model_file, "w") as output_f:
        for element in item_based_result:
            json.dump(element, output_f)
            output_f.write('\n')


elif cf_type == "user_based":
    business_data = input_data.map(lambda x: x[1]).distinct().collect()
    m = len(business_data)
    number_hash_func = 50
    row = 1
    band = int(number_hash_func / row)

    characteristic_matrix = input_data.map(lambda x: (x[0], [business_idx[x[1]]])).reduceByKey(lambda a, b: a + b).filter(lambda x: len(x[1]) >= co_rated_threshold).mapValues(lambda x: set(x)).persist()
    user_dict = characteristic_matrix.collectAsMap()
    signature_matrix = characteristic_matrix.map(lambda x: min_hash(x))
    candidate_pair = signature_matrix.flatMap(lambda x: divide_signature(x)).reduceByKey(lambda a, b: a + b).filter(lambda x: len(x[1]) > 1).flatMap(lambda x: [cand_pair for cand_pair in itertools.combinations(x[1], 2)]).distinct()
    similar_user = candidate_pair.map(lambda pair: (pair, jaccard_similarity(pair))).filter(lambda x: x[1] >= 0.01)

    user_baskets = input_data.map(lambda x: (x[0], [(x[1], x[2])])).reduceByKey(lambda a, b: a + b).mapValues(lambda x: list2dict(x)).persist()
    user_bussiness_dict = user_baskets.collectAsMap()
    user_based_result = similar_user.map(lambda x: x[0]).filter(lambda pair: len(set(user_bussiness_dict[pair[0]].keys()).intersection(set(user_bussiness_dict[pair[1]].keys()))) >= co_rated_threshold).map(lambda pair: {"u1": pair[0], "u2": pair[1], "sim": user_pearson_similarity(pair)}).filter(lambda x: x["sim"] > 0).collect()
    with open(model_file, "w") as output_f:
        for element in user_based_result:
            json.dump(element, output_f)
            output_f.write('\n')

end_time = time.time()
print("Duration: ", end_time - start_time)
