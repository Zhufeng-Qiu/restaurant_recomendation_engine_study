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

object task3predict {
  def main(args: Array[String]): Unit = {
    val start_time = new Date().getTime

//    val train_file = "/Users/zephyryau/Documents/study/INF553/2020-summer/Homework/Assignment3/python/data/train_review.json"
//    val test_file = "/Users/zephyryau/Documents/study/INF553/2020-summer/Homework/Assignment3/python/data/test_review.json"
//    val model_file = "/Users/zephyryau/Documents/study/INF553/2020-summer/Homework/Assignment3/python/result/task2_model_user"
//    //val model_file = "/Users/zephyryau/Documents/study/INF553/2020-summer/Homework/Assignment3/python/result/task2_model_item"
//    val output_file = "/Users/zephyryau/Documents/study/INF553/2020-summer/Homework/Assignment3/python/result/task3"
//    var f_idx = train_file.lastIndexOf('/')
//    val user_avg_file = train_file.slice(0, f_idx) + "/user_avg.json"
//    val business_avg_file = train_file.slice(0, f_idx) + "/business_avg.json"
//    val cf_type = "user_based"
    //val cf_type = "item_based"

    val train_file = args(0)
    val test_file = args(1)
    val model_file = args(2)
    val output_file = args(3)
    val cf_type = args(4)
    var f_idx = train_file.lastIndexOf('/')
    val user_avg_file = train_file.slice(0, f_idx) + "/user_avg.json"
    val business_avg_file = train_file.slice(0, f_idx) + "/business_avg.json"


    val conf = new SparkConf().setAppName("inf553-HW3-task3-predict").setMaster("local[*]").setAll(Array(("spark.executor.memory", "4g"), ("spark.driver.memory", "4g")))
    val sc = new SparkContext(conf)

    if (cf_type == "item_based") {
      val input_data = sc.textFile(train_file).map(string => parse(string).values.asInstanceOf[Map[String,Any]]).map(line => (line("user_id").toString, line("business_id").toString, line("stars").toString.toDouble)).map(x => (x._1, Array((x._2, x._3)))).reduceByKey(_++_).persist()
      val model_data = sc.textFile(model_file).map(string => parse(string).values.asInstanceOf[Map[String,Any]]).map(x => ((x("b1").toString, x("b2").toString), x("sim").toString.toDouble)).collectAsMap()
      val test_data = sc.textFile(test_file).map(string => parse(string).values.asInstanceOf[Map[String,String]]).map(x => (x("user_id"), x("business_id")))
      val business_avg_dict = sc.textFile(business_avg_file).map(string => parse(string).values.asInstanceOf[Map[String,Double]]).collect()(0)

      val N = 3

      def item_predict(x: (String, (String, Array[(String, Double)]))): Double ={
        val res_b_id = x._2._1
        val tmp_b_ids = x._2._2
        var numbers = mutable.ArrayBuffer[Tuple2[Double, Double]]()
        for (b_id <- tmp_b_ids) {
          var w_in = (b_id._1, res_b_id)
          numbers.append((b_id._2, model_data.getOrElse(w_in, 0)))
        }

        val tmp_num = numbers.sortBy(x => x._2).reverse.slice(0, N)
        val number = tmp_num.map(x => x._1 * x._2).sum
        val denominator = tmp_num.map(x => math.abs(x._2)).sum

        if ((number == 0) || (denominator == 0)){
          business_avg_dict.getOrElse(res_b_id, 0.0)
        } else {
          number / denominator
        }

      }

      val result = test_data.join(input_data).map(x => Map("user_id" -> x._1, "business_id" -> x._2._1, "stars" -> item_predict(x))).collect()
      var output_f = new PrintWriter(new File(output_file))
      for (element <- result) {
        var r_str = jackson.Json(DefaultFormats).write(element).toString
        output_f.write(r_str)
        output_f.write("\n")
      }
      output_f.close()


    } else if (cf_type == "user_based") {
      val input_data = sc.textFile(train_file).map(string => parse(string).values.asInstanceOf[Map[String,Any]]).map(line => (line("user_id").toString, line("business_id").toString, line("stars").toString.toDouble)).map(x => (x._2, Array((x._1, x._3)))).reduceByKey(_++_).persist()
      val model_data = sc.textFile(model_file).map(string => parse(string).values.asInstanceOf[Map[String,Any]]).map(x => ((x("u1").toString, x("u2").toString), x("sim").toString.toDouble)).collectAsMap()
      val test_data = sc.textFile(test_file).map(string => parse(string).values.asInstanceOf[Map[String,String]]).map(x => (x("business_id"), x("user_id")))
      val user_avg_dict = sc.textFile(user_avg_file).map(string => parse(string).values.asInstanceOf[Map[String,Double]]).collect()(0)

      def user_predict(x: (String, (String, Array[(String, Double)]))): Unit ={
        val res_u_id = x._2._1
        val tmp_u_ids = x._2._2
        var numbers = mutable.ArrayBuffer[Tuple3[Double, Double, Double]]()
        for (u_id <- tmp_u_ids) {
          var w_au = ("", "")
          if (u_id._1 < res_u_id) {
            w_au = (u_id._1, res_u_id)
          } else {
            w_au = (res_u_id, u_id._1)
          }
          numbers.append((u_id._2, user_avg_dict.getOrElse(u_id._1, user_avg_dict("UNK")), model_data.getOrElse(w_au, 0)))
        }

        val number = numbers.map(x => (x._1 - x._2) * x._3).sum
        val denominator = numbers.map(x => math.abs(x._2)).sum

        if ((number == 0) || (denominator == 0)) {
          user_avg_dict.getOrElse(res_u_id, user_avg_dict("UNK"))
        } else {
          return user_avg_dict.getOrElse(res_u_id, user_avg_dict("UNK")) + number / denominator
        }
      }

      val result = test_data.join(input_data).map(x => Map("user_id" -> x._1, "business_id" -> x._2._1, "stars" -> user_predict(x))).collect()
      var output_f = new PrintWriter(new File(output_file))
      for (element <- result) {
        var r_str = jackson.Json(DefaultFormats).write(element).toString
        output_f.write(r_str)
        output_f.write("\n")
      }
      output_f.close()
    }



  }
}
