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

object task3train {
  def main(args: Array[String]): Unit = {
    val start_time = new Date().getTime

//    val train_file = "/Users/zephyryau/Documents/study/INF553/2020-summer/Homework/Assignment3/python/data/train_review.json"
//    val model_file = "/Users/zephyryau/Documents/study/INF553/2020-summer/Homework/Assignment3/scala/result/task2_model_item_scala"
//    //val cf_type = "user_based"
//    val cf_type = "item_based"

    val train_file = args(0)
    val model_file = args(1)
    val cf_type = args(2)


    val co_rated_threshold = 3
    val conf = new SparkConf().setAppName("inf553-HW3-task3-train").setMaster("local[*]").setAll(Array(("spark.executor.memory", "4g"), ("spark.driver.memory", "4g")))
    val sc = new SparkContext(conf)
    sc.setLogLevel("Error")

    val input_data = sc.textFile(train_file).map(string => parse(string).values.asInstanceOf[Map[String,Any]]).map(line => (line("user_id").toString, line("business_id").toString, line("stars").toString.toDouble)).persist()
    val user_data = input_data.map(x => x._1).distinct().collect()
    val business_data = input_data.map(x => x._2).distinct().collect()

    var user_idx = mutable.HashMap[String, Int]()
    var i = 0
    for (user <- user_data) {
      user_idx(user) = user_idx.getOrElse(user, i)
      i += 1
    }

    var business_idx = mutable.HashMap[String, Int]()
    i = 0
    for (business <- business_data) {
      business_idx(business) = business_idx.getOrElse(business, i)
      i += 1
    }

    val a1 = Array(5003,5009,5011,5021,5023,5039,5051,5059,5077,5081,5087,5099,5101,5107,5113,5119,5147,5153,5167,5171,5179,5189,5197,5209,5227,5231,5233,5237,5261,5273,5279,5281,5297,5303,5309,5323,5333,5347,5351,5381,5387,5393,5399,5407,5413,5417,5419,5431,5437,5441)
    val a2 = Array(9127,9133,9137,9151,9157,9161,9173,9181,9187,9199,9203,9209,9221,9227,9239,9241,9257,9277,9281,9283,9293,9311,9319,9323,9337,9341,9343,9349,9371,9377,9391,9397,9403,9413,9419,9421,9431,9433,9437,9439,9461,9463,9467,9473,9479,9491,9497,9511,9521,9533)
    val b1 = Array(2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229)
    val b2 = Array(1009,1013,1019,1021,1031,1033,1039,1049,1051,1061,1063,1069,1087,1091,1093,1097,1103,1109,1117,1123,1129,1151,1153,1163,1171,1181,1187,1193,1201,1213,1217,1223,1229,1231,1237,1249,1259,1277,1279,1283,1289,1291,1297,1301,1303,1307,1319,1321,1327,1361)

    def list2dict(l: Array[(String, Double)]): mutable.HashMap[String, Double] ={
      val l_dict = mutable.HashMap[String, Double]()
      for (item <- l) {
        l_dict(item._1) = item._2
      }
      l_dict
    }



    if (cf_type == "item_based") {
      val business_baskets = input_data.map(x => (x._2, Array((x._1, x._3)))).reduceByKey(_++_).filter(x => x._2.size >= co_rated_threshold).mapValues(x => list2dict(x)).persist()
      val business_users_dict = business_baskets.collectAsMap()
      val intermediate_business_id = business_baskets.map(x => x._1)

      def item_pearson_similarity(pair: (String, String)): Double ={
        val users_i = business_users_dict(pair._1)
        val users_j = business_users_dict(pair._2)
        val co_rated_users = users_i.keys.toSet.intersect(users_j.keys.toSet)
        var business_i = co_rated_users.map(user => users_i(user))
        var business_j = co_rated_users.map(user => users_j(user))

        val r_i = business_i.sum / business_i.size
        val r_j = business_j.sum / business_j.size

        var tmp_1 = business_i.map(r_ui => (r_ui - r_i))
        var tmp_2 = business_j.map(r_uj => (r_uj - r_j))
        var tmp_3 = business_i.map(r_ui => math.pow(r_ui - r_i, 2))
        var tmp_4 = business_j.map(r_uj => math.pow(r_uj - r_j, 2))

        var number = tmp_1.zip(tmp_2).map(x => x._1 * x._2).sum

        var denominator = Math.sqrt(tmp_3.sum) * Math.sqrt(tmp_4.sum)
        if ((number == 0) || (denominator == 0)) {
          0.0.toDouble
        } else {
          number / denominator
        }
      }

      val item_based_result =  intermediate_business_id.cartesian(intermediate_business_id).filter(pair => pair._1 < pair._2).filter(pair => (business_users_dict(pair._1).toSet.intersect(business_users_dict(pair._2).toSet)).size >= co_rated_threshold).map(pair => Map("b1" -> pair._1, "b2" -> pair._2, "sim" -> item_pearson_similarity(pair))).filter(x => x("sim").toString.toDouble > 0.0).collect()

      var output_f = new PrintWriter(new File(model_file))
      for (element <- item_based_result) {
        var r_str = jackson.Json(DefaultFormats).write(element).toString
        output_f.write(r_str)
        output_f.write("\n")
      }
      output_f.close()

    } else if (cf_type == "user_based") {
      val business_data = input_data.map(x => x._2).distinct().collect()
      val m = business_data.size
      val number_hash_func = 50
      val row = 1
      val band = (number_hash_func / row).toInt

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

      val characteristic_matrix = input_data.map(x => (x._1, Array(business_idx(x._2)))).reduceByKey(_++_).filter(x => x._2.size >= co_rated_threshold)
      val user_dict = characteristic_matrix.collectAsMap()
      val signature_matrix = characteristic_matrix.map(x => min_hash(x))

      def build_pair(tuple: ((Int, Int), Array[String])) = {
        for (cand_pair <- tuple._2.combinations(2)) yield cand_pair.toSet
      }

      val candidate_pair = signature_matrix.flatMap(x => divide_signature(x)).reduceByKey(_++_).filter(x => x._2.size > 1).flatMap(x => build_pair(x)).distinct()

      def jaccard_similarity(pair: Array[String]) = {
        //global business_dict
        var pair_1 = user_dict(pair(0))
        var pair_2 = user_dict(pair(1))
        pair_1.intersect(pair_2).size.toFloat / pair_1.union(pair_2).size.toFloat
      }

      val similar_user = candidate_pair.map(pair => (pair.toArray, jaccard_similarity(pair.toArray))).filter(x => x._2 >= 0.01)
      val user_baskets = input_data.map(x => (x._1, Array((x._2, x._3)))).reduceByKey(_++_).mapValues(x => list2dict(x))
      val user_bussiness_dict = user_baskets.collectAsMap()

      def user_pearson_similarity(pair: Array[String]): Double ={
        val items_i = user_bussiness_dict(pair(0))
        val items_j = user_bussiness_dict(pair(1))
        val co_rated_bussiness = items_i.keys.toSet.intersect(items_j.keys.toSet)
        var business_i = co_rated_bussiness.map(user => items_i(user))
        var business_j = co_rated_bussiness.map(user => items_j(user))

        val r_i = business_i.sum / business_i.size
        val r_j = business_j.sum / business_j.size

        var tmp_1 = business_i.map(r_ui => (r_ui - r_i))
        var tmp_2 = business_j.map(r_uj => (r_uj - r_j))
        var tmp_3 = business_i.map(r_ui => math.pow(r_ui - r_i, 2))
        var tmp_4 = business_j.map(r_uj => math.pow(r_uj - r_j, 2))

        var number = tmp_1.zip(tmp_2).map(x => x._1 * x._2).sum
        var denominator = Math.sqrt(tmp_3.sum) * Math.sqrt(tmp_4.sum)
        if ((number == 0) || (denominator == 0)) {
          0.0.toDouble
        } else {
          number / denominator
        }
      }

      val user_based_result = similar_user.map(x => x._1).filter(pair => ((user_bussiness_dict(pair(0)).keys.toSet).intersect(user_bussiness_dict(pair(1)).keys.toSet)).size >= co_rated_threshold).map(pair => Map("u1" -> pair(0), "u2" -> pair(1), "sim" -> user_pearson_similarity(pair))).filter(x => x("sim").toString.toDouble > 0.0).collect()
      var output_f = new PrintWriter(new File(model_file))
      for (element <- user_based_result) {
        var r_str = jackson.Json(DefaultFormats).write(element).toString
        output_f.write(r_str)
        output_f.write("\n")
      }
      output_f.close()
    }
    val end_time = new Date().getTime
    var duration_time = end_time - start_time
    println("Duration: " + duration_time.toFloat/1000)
  }

}
