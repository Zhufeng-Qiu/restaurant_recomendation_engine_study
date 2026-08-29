import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import java.io._

import scala.io.Source
import org.json4s._
import org.json4s.jackson.JsonMethods._
import java.util.Date

import org.apache.spark.rdd.RDD

import scala.collection.mutable
import scala.util.control.Breaks.{break, breakable}

object task1 {
  def main(args: Array[String]): Unit = {
    val start_time = new Date().getTime


//    val input_file = "/Users/zephyryau/Documents/study/INF553/2020-summer/Homework/Assignment3/python/data/train_review.json"
//    val output_file = "/Users/zephyryau/Documents/study/INF553/2020-summer/Homework/Assignment3/scala/result/task1_scala"

    val input_file = args(0)
    val output_file = args(1)


    val conf = new SparkConf().setAppName("inf553-HW3-task1").setMaster("local[*]").setAll(Array(("spark.executor.memory", "4g"), ("spark.driver.memory", "4g")))
    val sc = new SparkContext(conf)

    val input_data = sc.textFile(input_file).map(string => parse(string).values.asInstanceOf[Map[String,String]]).map(line => (line("user_id"), line("business_id"))).persist()
    val user_data = input_data.map(x => x._1).distinct().collect()
    val m = user_data.size
    val number_hash_func = 100
    val row = 1
    val band = (number_hash_func / row).toInt

    var user_idx = mutable.Map[String, Int]()
    var i = 0
    for (user <- user_data) {
      user_idx(user) = user_idx.getOrElse(user, i)
      i += 1
    }

    val a1 = Array(5003,5009,5011,5021,5023,5039,5051,5059,5077,5081,5087,5099,5101,5107,5113,5119,5147,5153,5167,5171,5179,5189,5197,5209,5227,5231,5233,5237,5261,5273,5279,5281,5297,5303,5309,5323,5333,5347,5351,5381,5387,5393,5399,5407,5413,5417,5419,5431,5437,5441,5443,5449,5471,5477,5479,5483,5501,5503,5507,5519,5521,5527,5531,5557,5563,5569,5573,5581,5591,5623,5639,5641,5647,5651,5653,5657,5659,5669,5683,5689,5693,5701,5711,5717,5737,5741,5743,5749,5779,5783,5791,5801,5807,5813,5821,5827,5839,5843,5849,5851)
    val a2 = Array(9127,9133,9137,9151,9157,9161,9173,9181,9187,9199,9203,9209,9221,9227,9239,9241,9257,9277,9281,9283,9293,9311,9319,9323,9337,9341,9343,9349,9371,9377,9391,9397,9403,9413,9419,9421,9431,9433,9437,9439,9461,9463,9467,9473,9479,9491,9497,9511,9521,9533,9539,9547,9551,9587,9601,9613,9619,9623,9629,9631,9643,9649,9661,9677,9679,9689,9697,9719,9721,9733,9739,9743,9749,9767,9769,9781,9787,9791,9803,9811,9817,9829,9833,9839,9851,9857,9859,9871,9883,9887,9901,9907,9923,9929,9931,9941,9949,9967,9973,10007)
    val b1 = Array(2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313,317,331,337,347,349,353,359,367,373,379,383,389,397,401,409,419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,509,521,523,541)
    val b2 = Array(1009,1013,1019,1021,1031,1033,1039,1049,1051,1061,1063,1069,1087,1091,1093,1097,1103,1109,1117,1123,1129,1151,1153,1163,1171,1181,1187,1193,1201,1213,1217,1223,1229,1231,1237,1249,1259,1277,1279,1283,1289,1291,1297,1301,1303,1307,1319,1321,1327,1361,1367,1373,1381,1399,1409,1423,1427,1429,1433,1439,1447,1451,1453,1459,1471,1481,1483,1487,1489,1493,1499,1511,1523,1531,1543,1549,1553,1559,1567,1571,1579,1583,1597,1601,1607,1609,1613,1619,1621,1627,1637,1657,1663,1667,1669,1693,1697,1699,1709,1721)

    def h1(x: Int, i: Int) = {
      //global a1
      //global b1
      a1(i) * x + b1(i)
    }

    def h2(x: Int, i: Int) = {
      //global a2
      //global b2
      a2(i) * x + b2(i)
    }
    def min_hash(tuple: (String, Array[Int])) = {
      //global m
      //global number_hash_func
      val p = 24251
      var min_hash_values = mutable.ArrayBuffer[Int]()
      for (itr <- 0 until number_hash_func) {
        var intermediate_values = mutable.ArrayBuffer[Int]()
        for ((id, i) <- tuple._2.zipWithIndex) {
          var i_i = i + 1
          var itr1 = itr + 1
          var v = ((itr1 * h1(id, itr) + itr1 * h2(id, itr) + itr1 * itr1) % p) % m
          intermediate_values.append(v)
        }
        min_hash_values.append(intermediate_values.min)
      }
      (tuple._1, min_hash_values.toArray)
    }
    def band_hash(v_list: Array[Int]) = {
      var sum = 0
      val p = 131
      for (v <- v_list) {
        sum += v * p
      }
      sum & 0x7FFFFFFF
    }
    def divide_signature(x: Tuple2[String, Array[Int]]) = {
      //global row
      //global band
      var result = mutable.ArrayBuffer[Tuple2[Tuple2[Int, Int], Array[String]]]()
      var list_idx = 0
      for (i <- 0 until band by row) {
        result.append(((i, band_hash(x._2.slice(list_idx, list_idx + row))), Array(x._1)))
        list_idx += row
      }
      result
    }

    val characteristic_matrix = input_data.map(x => (x._2, Array(user_idx(x._1)))).reduceByKey(_++_).persist()

    val business_dict = characteristic_matrix.collectAsMap()

    def jaccard_similarity(pair: Array[String]) = {
      //global business_dict
      var pair_1 = business_dict(pair(0))
      var pair_2 = business_dict(pair(1))
      pair_1.intersect(pair_2).size.toFloat / pair_1.union(pair_2).size.toFloat
    }

    val signature_matrix = characteristic_matrix.map(x => min_hash(x))

    def build_pair(tuple: ((Int, Int), Array[String])) = {
      for (cand_pair <- tuple._2.combinations(2)) yield cand_pair.toSet
    }

    val candidate_pair = signature_matrix.flatMap(x => divide_signature(x)).reduceByKey(_++_).filter(x => x._2.size > 1).flatMap(x => build_pair(x)).distinct()

    val similar_pair = candidate_pair.map(pair => (pair.toArray, jaccard_similarity(pair.toArray))).filter(x => x._2.toFloat >= 0.05).map(pair => Map("b1" -> pair._1(0), "b2" -> pair._1(1), "sim" -> pair._2)).collect()

    var output_f = new PrintWriter(new File(output_file))
    for (element <- similar_pair){
      var result_str = jackson.Json(DefaultFormats).write(element).toString.replace(":", ": ").replace(",", ", ")
      output_f.write(result_str)
      output_f.write("\n")
    }
    output_f.close()

    val end_time = new Date().getTime
    var duration_time = end_time - start_time
    println("Duration: " + duration_time.toFloat/1000)
  }
}
