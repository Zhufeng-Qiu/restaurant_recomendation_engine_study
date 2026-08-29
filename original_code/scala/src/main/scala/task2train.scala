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

object task2train {
  def main(args: Array[String]): Unit = {
    val start_time = new Date().getTime


//    val train_file = "/Users/zephyryau/Documents/study/INF553/2020-summer/Homework/Assignment3/python/data/train_review.json"
//    val model_file = "/Users/zephyryau/Documents/study/INF553/2020-summer/Homework/Assignment3/scala/result/task2_model_scala"
//    val stopwords = "/Users/zephyryau/Documents/study/INF553/2020-summer/Homework/Assignment3/python/data/stopwords"

    val train_file = args(0)
    val model_file = args(1)
    val stopwords = args(2)


    var stop_punctuations = Set("(", "[", ",", ".", "!", "?", ":", ";", "]", ")", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "$", "/", "\\", "\"", "%", "#")

    val conf = new SparkConf().setAppName("inf553-HW3-task2-train").setMaster("local[*]").setAll(Array(("spark.executor.memory", "4g"), ("spark.driver.memory", "4g")))
    val sc = new SparkContext(conf)

    val sw_source = Source.fromFile(stopwords)
    var stop_words_list = sw_source.mkString.split("\n").toSet
    sw_source.close()
    stop_words_list ++= Set("\n", "\u00a0", "-")
    val rare_word_threshold = 35072943 * 0.000001

    def delete_token(string: String, stop_punctuations: Set[String]) = {
      var string1 = ""
      for (c <- string) {
        if (!stop_punctuations.contains(c.toString)) {
          string1 += c
        }
      }
      string1
    }

    def delete_stop_words(list: Array[String], stop_words_list: Set[String]) = {
      var new_list = mutable.ArrayBuffer[String]()
      for (l <- list) {
        if (!stop_words_list.contains(l)) {
          new_list.append(l)
        }
      }
      new_list.toArray
    }

    def tf(basket: (String, Array[String])) = {
      var word_dict = mutable.Map[Tuple2[String, String], Int]()
      for (word <- basket._2) {
        word_dict((word, basket._1)) = word_dict.getOrElse((word, basket._1), 0) + 1
      }
      var max_f = word_dict.values.max
      var result = mutable.ArrayBuffer[Tuple2[Tuple2[String, String], Tuple2[Int, Int]]]()
      for ((key, value) <- word_dict) {
        result.append((key, (value, max_f)))
      }
      result.toArray
    }


    val input_data = sc.textFile(train_file).map(string => parse(string).values.asInstanceOf[Map[String,String]])
    val business_document = input_data.map(line => (line("business_id"), delete_token(line("text").toLowerCase(), stop_punctuations))).map(line => (line._1, delete_stop_words(line._2.split("\\s+"), stop_words_list))).reduceByKey(_++_)
    val business_word_tf_tmp = business_document.flatMap(x => tf(x))
    val business_word_tf = business_word_tf_tmp.map(x => (x._1, x._2._1.toFloat / x._2._2.toFloat))
    val total_documents = business_document.collect().size

    def idf(basket: (String, (Array[String], Int))) = {
      var result = mutable.ArrayBuffer[Tuple2[Tuple2[String, String], Double]]()
      for (b_id <- basket._2._1) {
        result.append(((basket._1, b_id), Math.log10(total_documents / basket._2._1.size) / Math.log10(2.0)))
      }
      result.toArray
    }

    val business_word_idf = business_word_tf_tmp.map(x => (x._1._1, (Array(x._1._2), x._2._1))).aggregateByKey((Array[String](), 0))((x, y) => (x._1 ++ y._1, x._2 + y._2), (x, y) => (x._1 ++ y._1, x._2 + y._2)).filter(x => x._2._2 > rare_word_threshold).flatMap(x => idf(x))
    val business_word_tf_idf = business_word_tf.join(business_word_idf).map(x => (x._1, x._2._1 * x._2._2)).sortBy(x => x._2, false)

    val business_profiles_tmp = business_word_tf_idf.map(x => (x._1._2, Array(x._1._1))).reduceByKey(_++_).mapValues(x => x.slice(0, 200))
    val business_profiles = business_profiles_tmp.collect()
    //println(business_profiles.size)
    val business_Map = business_profiles_tmp.collectAsMap()

    def u_p_calculate(x: Array[String]) = {
      var result = mutable.ArrayBuffer[Array[String]]()
      for (b_id <- x) {
        result.append(business_Map(b_id))
      }
      result(0).toSet.toArray
    }

    val user_profiles = input_data.map(line => (line("user_id"), Array(line("business_id")))).reduceByKey(_++_).mapValues(x => u_p_calculate(x)).collectAsMap()
    print(user_profiles.size)

    var output_f = new PrintWriter(new File(model_file))
    output_f.write("business profiles:")
    output_f.write("\n")
    for (business_profile <- business_profiles) {
      var b_str = jackson.Json(DefaultFormats).write(business_profile).toString
      output_f.write(b_str)
      output_f.write("\n")
    }
    output_f.write("\n")
    output_f.write("user profiles:")
    output_f.write("\n")
    for (user_profile <- user_profiles) {
      var u_str = jackson.Json(DefaultFormats).write(user_profile).toString
      //print(u_str)
      output_f.write(u_str)
      output_f.write("\n")
    }
    output_f.close()

    val end_time = new Date().getTime
    var duration_time = end_time - start_time
    println("Duration: " + duration_time.toFloat/1000)
  }
}
