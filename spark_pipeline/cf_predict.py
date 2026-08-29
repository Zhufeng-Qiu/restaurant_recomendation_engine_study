from pyspark import SparkContext, SparkConf
import sys
import json
import time
import math

start_time = time.time()
train_file = sys.argv[1]
test_file = sys.argv[2]
model_file = sys.argv[3]
output_file = sys.argv[4]
cf_type = sys.argv[5]
# user_avg.json and business_avg.json are expected next to the train file
f_idx = train_file.rfind("/")
user_avg_file = train_file[:f_idx] + "/user_avg.json"
business_avg_file = train_file[:f_idx] + "/business_avg.json"

conf = SparkConf().setAppName("cf-rating-predict").setMaster("local[*]").setAll((("spark.executor.memory", "4g"), ("spark.driver.memory", "4g")))
sc = SparkContext(conf=conf)

if cf_type == "item_based":
    input_data = sc.textFile(train_file).map(lambda line:json.loads(line)).map(lambda line: (line["user_id"], line["business_id"], line["stars"])).map(lambda x: (x[0], [(x[1], x[2])])).reduceByKey(lambda a, b: a + b).persist()
    model_data = sc.textFile(model_file).map(lambda line:json.loads(line)).map(lambda x: ((x["b1"], x["b2"]), x["sim"])).collectAsMap()
    test_data = sc.textFile(test_file).map(lambda line: json.loads(line)).map(lambda x: (x["user_id"], x["business_id"]))
    business_avg_dict = sc.textFile(business_avg_file).map(lambda line: json.loads(line)).collect()[0]

    def item_predict(x):
        global model_data
        global N
        global business_avg_dict
        res_b_id = x[1][0]
        tmp_b_ids = x[1][1]
        numbers = []
        for b_id in tmp_b_ids:
            w_in = tuple(sorted((b_id[0], res_b_id)))
            numbers.append((b_id[1], model_data.get(w_in, 0)))
        tmp_num = sorted(numbers, key=lambda x: x[1], reverse=True)[:N]
        number = sum([pair[0] * pair[1] for pair in tmp_num])
        denominator = sum([abs(pair[1]) for pair in tmp_num])
        if (number == 0) or (denominator == 0):
            return business_avg_dict.get(res_b_id, 0)
        else:
            return number / denominator
    N = 3
    result = test_data.join(input_data).map(lambda x: {"user_id": x[0], "business_id": x[1][0], "stars": item_predict(x)}).collect()
    with open(output_file, "w") as output_f:
        for element in result:
            json.dump(element, output_f)
            output_f.write('\n')

    end_time = time.time()
    print("Duration: ", end_time - start_time)

elif cf_type == "user_based":
    input_data = sc.textFile(train_file).map(lambda line:json.loads(line)).map(lambda line: (line["user_id"], line["business_id"], line["stars"])).map(lambda x: (x[1], [(x[0], x[2])])).reduceByKey(lambda a, b: a + b).persist()
    model_data = sc.textFile(model_file).map(lambda line:json.loads(line)).map(lambda x: ((x["u1"], x["u2"]), x["sim"])).collectAsMap()
    test_data = sc.textFile(test_file).map(lambda line: json.loads(line)).map(lambda x: (x["business_id"], x["user_id"]))
    user_avg_dict = sc.textFile(user_avg_file).map(lambda line: json.loads(line)).collect()[0]

    def user_predict(x):
        global model_data
        global N
        global user_avg_dict
        res_u_id = x[1][0]
        tmp_u_ids = x[1][1]
        numbers = []
        for u_id in tmp_u_ids:
            w_au = tuple(sorted((u_id[0], res_u_id)))
            numbers.append((u_id[1], user_avg_dict.get(u_id[0], user_avg_dict["UNK"]), model_data.get(w_au, 0)))
        number = sum([(pair[0] - pair[1]) * pair[2] for pair in numbers])
        denominator = sum([abs(pair[1]) for pair in numbers])
        if (number == 0) or (denominator == 0):
            return user_avg_dict.get(res_u_id, user_avg_dict["UNK"])
        else:
            return user_avg_dict.get(res_u_id, user_avg_dict["UNK"]) + number / denominator

    result = test_data.join(input_data).map(lambda x: {"user_id": x[0], "business_id": x[1][0], "stars": user_predict(x)}).collect()
    with open(output_file, "w") as output_f:
        for element in result:
            json.dump(element, output_f)
            output_f.write('\n')

    end_time = time.time()
    print("Duration: ", end_time - start_time)
