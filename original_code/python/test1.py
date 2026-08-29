# x1 = ('bPcqucuuClxYrIM8xWoArg', [24267, 13253, 7682, 7802, 2076, 13572, 16896, 8068, 13737, 24652, 11850, 5641, 123, 12130, 11580, 15003, 8160, 23356, 12244, 23544, 23011, 4665, 15663, 2262, 61, 5092, 2696, 17247, 13354, 19763, 13227, 17252, 5327, 2505, 9936, 738, 17088, 22383, 12233, 2057, 9740, 33, 14585, 25815, 21202, 23209, 17880, 20754, 4054, 9109, 9712, 15672, 11695, 25892, 23319, 20616, 543, 26105, 6722, 6052, 15025, 19161, 24335, 17839, 18923, 1486, 61, 25323, 24287, 18727, 13674, 20139, 4611, 25713, 4025, 17156, 1288, 4025, 3897, 14, 16863, 17880, 14753, 8797, 2201, 5681, 3752, 3744, 4130, 19047, 21426, 582, 10228, 22788, 1412, 2628, 2472, 24335, 10759, 4086, 19695, 309, 3765, 16983, 9374, 19178, 473, 2351, 20756, 15182, 18692, 2613, 17471, 21718, 16889, 3028, 20593, 2548, 4320, 24078, 24420, 25411, 7372, 3856, 6741, 8800, 19526, 1772, 11311, 15103, 5846, 7000, 18478, 24453, 15577, 18228, 1948, 19002, 11952, 23209, 1, 15771, 6076, 8765, 61, 7589, 13782, 23356, 1898, 287, 3744, 24453, 22574, 8899, 11655, 22195, 6955, 9691, 9388, 123, 22813, 7682, 6985, 13427, 15091, 11086, 18650, 6064, 5845, 6582, 24040, 16874, 20035, 19617, 626, 19099, 11496, 18732, 5616, 1526, 19991, 5993])
# x2 = ('VfFHPsPtTW4Mgx0eHDyJiQ', [9328, 11752, 2176, 11812, 9399, 7580, 23261, 3884, 6803, 4888, 7498, 8136, 6657, 11543, 15725, 21487, 12299, 11638, 10295, 8310, 17382, 25149, 19629, 24792, 19214, 19357, 11874, 16872, 5633, 16212, 10295, 8392, 2681, 5019, 24871, 10819, 6222, 19084, 16003, 24792, 16, 11464, 19228, 14250, 5848, 14402, 10167, 160, 25876, 17533, 9906, 5677, 8765, 6790, 19808, 7508, 20445, 13295, 24478, 13429, 12165, 4085, 3544, 23635, 20848, 4074, 16896, 7, 3817, 20754, 4759, 24495, 18004, 4143, 5561, 10260, 9500, 15496, 11719, 16270, 13584, 24335, 19594, 8903, 20625, 24333, 14150, 11285, 24271, 1593, 11287, 23607, 9515, 7760, 24306, 242, 17084, 701, 7882, 528, 432, 14272, 9887, 11574, 21611, 12580, 19070, 24701, 13198, 26009, 9433, 16416, 19692, 9348, 1852, 5767, 2709])
# m = 26184
#
# a1 = [5003,5009,5011,5021,5023,5039,5051,5059,5077,5081,5087,5099,5101,5107,5113,5119,5147,5153,5167,5171,5179,5189,5197,5209,5227,5231,5233,5237,5261,5273]
# a2 = [9733,9739,9743,9749,9767,9769,9781,9787,9791,9803,9811,9817,9829,9833,9839,9851,9857,9859,9871,9883,9887,9901,9907,9923,9929,9931,9941,9949,9967,9973]
# b1 = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113]
# b2 = [503,509,521,523,541,547,557,563,569,571,577,587,593,599,601,607,613,617,619,631,641,643,647,653,659,661,673,677,683,691]
#
# def h1(x, i):
#     global a1
#     global b1
#     return a1[i] * x + b1[i]
#
# def h2(x, i):
#     global a2
#     global b2
#     return a2[i] * x + b2[i]
#
# def min_hash(x):
#     p = 24251
#     global m
#     min_hash_values = []
#
#     for itr in range(30):
#         intermediate_values = []
#         for i, id in enumerate(x[1]):
#             i_i = i + 1
#             v = ((i_i * h1(id, itr) + i_i * h2(id, itr) + i_i * i_i) % p) % m
#             intermediate_values.append(v)
#         #print(intermediate_values)
#         min_hash_values.append(min(intermediate_values))
#     return (x[0], min_hash_values)
#
# number_hash_func = 30
# row = 1
# band = int(number_hash_func / row)
#
# x = ('bZMcorDrciRbjdjRyANcjA', [958, 1841, 1027, 3367, 714, 227, 149, 791, 891, 104, 132, 2302, 1661, 236, 1382, 7681, 5223, 242, 15, 2173, 488, 1345, 757, 496, 391, 350, 1275, 1363, 1635, 1013])
# def band_hash(v_list):
#     sum = 0
#     p = 131
#     for v in v_list:
#             sum += v * p
#     return sum & 0x7FFFFFFF
#
# def divide_signature(x):
#     global row
#     global band
#     r = []
#     list_idx = 0
#     for i in range(0, band, row):
#         r.append(((i, band_hash(x[1][list_idx: list_idx + row])), x[0]))
#         list_idx += row
#     return r
#
# print(divide_signature(x))
#
#
# x1 = [10, 958, 1841, 1027, 3367, 714, 227, 149]
# x2 = [958, 1841, 1027, 3367, 714, 227, 149, 791, 891]
#
# def jaccard_similarity(pair):
#     return len(set(pair[0]).intersection(set(pair[1]))) / len(set(pair[0]).union(set(pair[1])))
#
# print(jaccard_similarity((x1, x2)))


# x = set([1, 2, 3])
# for i, id in enumerate(x):
#     print(i, id)

from pyspark import SparkContext, SparkConf
import sys
import json
import time
import math

# train_file = "/Users/zephyryau/Documents/study/INF553/2020-summer/Homework/Assignment3/python/data/train_review.json"
# #train_file = "/Users/zephyryau/Documents/study/INF553/2020-summer/Homework/Assignment3/python/data/test/test_data.json"
# model_file = "/Users/zephyryau/Documents/study/INF553/2020-summer/Homework/Assignment3/python/result/task2_model"
# stopwords = "/Users/zephyryau/Documents/study/INF553/2020-summer/Homework/Assignment3/python/data/stopwords"
# conf = SparkConf().setAppName("inf553-HW3-task2-train").setMaster("local[*]").setAll((("spark.executor.memory", "4g"), ("spark.driver.memory", "4g")))
# sc = SparkContext(conf=conf)
#
# input_data = sc.textFile(train_file).map(lambda line: json.loads(line)).persist()
# user_profiles = input_data.map(lambda line: (line['user_id'], [line['business_id']])).reduceByKey(lambda a, b: a + b).mapValues(lambda x: list(set([business_Map[b_id] for b_id in x][0])))#.collect()
# print(user_profiles.take(5))
# #print(user_profiles)

# train_file = "/Users/zephyryau/Documents/study/INF553/2020-summer/Homework/Assignment3/python/data/train_review.json"
# test_file = "/Users/zephyryau/Documents/study/INF553/2020-summer/Homework/Assignment3/python/data/test_review.json"
# model_file = "/Users/zephyryau/Documents/study/INF553/2020-summer/Homework/Assignment3/python/result/task2_model_user"
# #model_file = "/Users/zephyryau/Documents/study/INF553/2020-summer/Homework/Assignment3/python/result/task2_model_item"
# output_file = "/Users/zephyryau/Documents/study/INF553/2020-summer/Homework/Assignment3/python/result/task3"
# user_avg_file_test = "/Users/zephyryau/Documents/study/INF553/2020-summer/Homework/Assignment3/python/data/user_avg.json"
# business_avg_file_test = "/Users/zephyryau/Documents/study/INF553/2020-summer/Homework/Assignment3/python/data/business_avg.json"
#
# f_idx = train_file.rfind("/")
# user_avg_file = train_file[:f_idx] + "/user_avg.json"
# business_avg_file = train_file[:f_idx] + "/business_avg.json"
#
# print(user_avg_file == user_avg_file_test)
# print(business_avg_file == business_avg_file_test)

import itertools

s = [1,1,2,3]
print(tuple(itertools.combinations(s, 2)))


