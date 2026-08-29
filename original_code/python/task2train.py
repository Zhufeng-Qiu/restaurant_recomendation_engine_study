from pyspark import SparkContext, SparkConf
import sys
import json
import time
import math

start_time = time.time()
#'''
train_file = "/Users/zephyryau/Documents/study/INF553/2020-summer/Homework/Assignment3/python/data/train_review.json"
#train_file = "/Users/zephyryau/Documents/study/INF553/2020-summer/Homework/Assignment3/python/data/test/test_data.json"
model_file = "/Users/zephyryau/Documents/study/INF553/2020-summer/Homework/Assignment3/python/result/task2_model"
stopwords = "/Users/zephyryau/Documents/study/INF553/2020-summer/Homework/Assignment3/python/data/stopwords"

'''
train_file = sys.argv[1]
model_file = sys.argv[2]
stopwords =sys.argv[3]
'''

stop_punctuations = ["(", "[", ",", ".", "!", "?", ":", ";", "]", ")", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "$", "/", '\\', '"', '%', '#']

conf = SparkConf().setAppName("inf553-HW3-task2-train").setMaster("local[*]").setAll((("spark.executor.memory", "4g"), ("spark.driver.memory", "4g")))
sc = SparkContext(conf=conf)

stop_words_list = []
stop_words_file = open(stopwords, 'r')
lines = stop_words_file.readlines()
for line in lines:
    stop_words_list.append(line.strip('\n'))
stop_words_list += ['\n', '\u00a0', '-']
rare_word_threshold = 35072943 * 0.000001

def delete_token(string, stop_punctuations):
    string1 = ""
    for c in string:
        if c not in set(stop_punctuations):
            string1 += c
    return string1

def delete_stop_words(list, stop_words_list):
    new_list = []
    for l in list:
        if l not in set(stop_words_list):
            new_list.append(l)
    return new_list


def tf(basket):
    word_dict = {}
    for word in basket[1]:
        word_dict[(word, basket[0])] = word_dict.get((word, basket[0]), 0) + 1
    max_f = max(word_dict.values())
    return [(key, (value, max_f)) for key, value in word_dict.items()]

def idf(basket):
    global total_documents
    return [((basket[0], b_id), math.log(total_documents / len(basket[1][0]), 2)) for b_id in basket[1][0]]

#data_RDD_E = sc.textFile(train_file).map(lambda line:json.loads(line)).map(lambda line: delete_token(line['text'], stop_punctuations)).flatMap(lambda line: line.split()).map(lambda word: (word.strip().lower(), 1)).filter(lambda word: word[0] not in stop_words).collect()
#print(len(data_RDD_E))

input_data = sc.textFile(train_file).map(lambda line: json.loads(line)).persist()
business_document = input_data.map(lambda line: (line['business_id'], delete_token(line['text'].lower(), stop_punctuations))).map(lambda line: (line[0], delete_stop_words(line[1].split(), stop_words_list))).reduceByKey(lambda a, b: a + b).persist()
business_word_tf_tmp = business_document.flatMap(lambda x: tf(x)).persist()
business_word_tf = business_word_tf_tmp.map(lambda x: (x[0], x[1][0]/x[1][1]))
total_documents = len(business_document.collect())
seqFunc = (lambda x, y: (x[0] + y[0], x[1] + y[1]))
combFunc = (lambda x, y: (x[0] + y[0], x[1] + y[1]))
#business_word_idf = business_word_tf_tmp.map(lambda pair: (pair[0][0], [pair[0][1]])).reduceByKey(lambda a, b: a + b).flatMap(lambda x: idf(x))
#business_word_tf_idf = business_word_tf.join(business_word_idf).map(lambda x: (x[0], x[1][0] * x[1][1])).map(lambda x: (x[0][1], (x[0][0], x[1]))).sortBy(lambda x: x[1][1], False)
business_word_idf = business_word_tf_tmp.map(lambda x: (x[0][0], ([x[0][1]], x[1][0]))).aggregateByKey(([], 0), seqFunc, combFunc).filter(lambda x: x[1][1] > rare_word_threshold).flatMap(lambda x: idf(x))
business_word_tf_idf = business_word_tf.join(business_word_idf).map(lambda x: (x[0], x[1][0] * x[1][1])).sortBy(lambda x: x[1], False)

#word_idx = business_word_tf_idf.map(lambda x: (x[0][0], 1)).reduceByKey(lambda a, b: 1).map(lambda x: x[0]).zipWithIndex().collectAsMap()

business_profiles_tmp = business_word_tf_idf.map(lambda x: (x[0][1], [x[0][0]])).reduceByKey(lambda a, b: a + b).mapValues(lambda x: x[:200])
    #.mapValues(lambda word_list: [word_idx[word] for word in word_list]).persist()
business_profiles = business_profiles_tmp.collect()
#print(len(business_profiles))
business_Map = business_profiles_tmp.collectAsMap()

user_profiles = input_data.map(lambda line: (line['user_id'], [line['business_id']])).reduceByKey(lambda a, b: a + b).mapValues(lambda x: list(set([business_Map[b_id] for b_id in x][0]))).collect()
#print(len(user_profiles))

with open(model_file, "w") as output_f:
    output_f.write("business profiles:")
    output_f.write('\n')
    for business_profile in business_profiles:
        json.dump(business_profile, output_f)
        output_f.write('\n')
    output_f.write('\n')
    output_f.write("user profiles:")
    output_f.write('\n')
    for user_profile in user_profiles:
        json.dump(user_profile, output_f)
        output_f.write('\n')

end_time = time.time()
print("Duration: ", end_time - start_time)

