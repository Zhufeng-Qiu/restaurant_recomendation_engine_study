from pyspark import SparkContext, SparkConf
import sys
import json
import time
import math

start_time = time.time()
#'''
test_file = "/Users/zephyryau/Documents/study/INF553/2020-summer/Homework/Assignment3/python/data/test_review.json"
model_file = "/Users/zephyryau/Documents/study/INF553/2020-summer/Homework/Assignment3/python/result/task2_model"
output_file = "/Users/zephyryau/Documents/study/INF553/2020-summer/Homework/Assignment3/python/result/task2_predict"
'''
test_file = sys.argv[1]
model_file = sys.argv[2]
output_file = sys.argv[3]
'''

conf = SparkConf().setAppName("inf553-HW3-task2-predict").setMaster("local[*]").setAll((("spark.executor.memory", "4g"), ("spark.driver.memory", "4g")))
sc = SparkContext(conf=conf)

test_data = sc.textFile(test_file).map(lambda line: json.loads(line)).map(lambda x: (x["user_id"], x["business_id"]))
model_data = sc.textFile(model_file).zipWithIndex().persist()
model_data_list = model_data.collectAsMap()
business_profiles = model_data.filter(lambda x: ((x[1] > model_data_list["business profiles:"]) & (x[1] < model_data_list["user profiles:"] - 1))).map(lambda x: x[0]).map(lambda line: json.loads(line)).collectAsMap()
user_profiles_tmp = model_data.filter(lambda x: (x[1] > model_data_list["user profiles:"])).map(lambda x: x[0]).map(lambda line: json.loads(line)).persist()
user_profiles = user_profiles_tmp.collectAsMap()

def cosine_similarity(pair):
    pair_user = set(user_profiles.get(pair[0], []))
    pair_business = set(business_profiles.get(pair[1], []))
    if (len(pair_business) == 0) or (len(pair_user) == 0):
        return 0
    else:
        return len(pair_user.intersection(pair_business)) / (math.sqrt(len(pair_business)) * math.sqrt(len(pair_user)))

result = test_data.map(lambda pair: (pair, cosine_similarity(pair))).filter(lambda x: x[1] >= 0.01).map(lambda pair: {"user_id": pair[0][0], "business_id": pair[0][1], "sim": pair[1]}).collect()
with open(output_file, "w") as output_f:
    for element in result:
        json.dump(element, output_f)
        output_f.write('\n')


end_time = time.time()
print("Duration: ", end_time - start_time)
