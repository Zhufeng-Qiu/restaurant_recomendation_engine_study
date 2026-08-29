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

object task2predict {
  def main(args: Array[String]): Unit = {
    val start_time = new Date().getTime

//    val test_file = "/Users/zephyryau/Documents/study/INF553/2020-summer/Homework/Assignment3/python/data/test_review.json"
//    val model_file = "/Users/zephyryau/Documents/study/INF553/2020-summer/Homework/Assignment3/scala/result/task2_model_scala"
//    val output_file = "/Users/zephyryau/Documents/study/INF553/2020-summer/Homework/Assignment3/scala/result/task2_predict_scala"

    val test_file = args(0)
    val model_file = args(1)
    val output_file = args(2)

    val conf = new SparkConf().setAppName("inf553-HW3-task2-predict").setMaster("local[*]").setAll(Array(("spark.executor.memory", "4g"), ("spark.driver.memory", "4g")))
    val sc = new SparkContext(conf)

    val test_data = sc.textFile(test_file).map(string => parse(string).values.asInstanceOf[Map[String,String]]).map(x => (x("user_id"), x("business_id")))
    val model_data = sc.textFile(model_file).zipWithIndex().persist()
    //model_data.take(5).foreach(println)
    val model_data_list = model_data.collectAsMap()
    val business_profiles = model_data.filter(x => ((x._2 > model_data_list("business profiles:")) & (x._2 < model_data_list("user profiles:") - 1))).map(x => x._1).map(string => parse(string).values.asInstanceOf[Map[String,List[String]]]).flatMap(x => x).collectAsMap()
    //business_profiles.take(5).foreach(println)
    val user_profiles_tmp = model_data.filter(x => (x._2 > model_data_list("user profiles:"))).map(x => x._1).map(string => parse(string).values.asInstanceOf[Map[String,List[String]]]).flatMap(x => x).persist()
    //user_profiles_tmp.take(5).foreach(println)
    val user_profiles = user_profiles_tmp.collectAsMap()

    def cosine_similarity(pair: (String, String)): Double = {
      val pair_user = user_profiles.getOrElse(pair._1, List[String]()).toSet
      val pair_business = business_profiles.getOrElse(pair._2, List[String]()).toSet
      if ((pair_business.size == 0) || (pair_user.size == 0)){
        0.0
      } else {
        pair_user.intersect(pair_business).size.toDouble / ((Math.sqrt(pair_business.size)) * (Math.sqrt(pair_user.size))).toDouble
      }
    }

    val result = test_data.map(pair => (pair, cosine_similarity(pair))).filter(x => x._2 >= 0.01).map(pair => Map("user_id" -> pair._1._1, "business_id" -> pair._1._2, "sim" -> pair._2)).collect()
    //println(test_data.map(pair => (pair, cosine_similarity(pair))).take(20).foreach(println))
    //println(result.size)
    var output_f = new PrintWriter(new File(output_file))
    for (element <- result) {
      var r_str = jackson.Json(DefaultFormats).write(element).toString.replace(":", ": ").replace(",", ", ")
      output_f.write(r_str)
      output_f.write("\n")
    }
    output_f.close()

    val end_time = new Date().getTime
    var duration_time = end_time - start_time
    println("Duration: " + duration_time.toFloat/1000)
  }
}
