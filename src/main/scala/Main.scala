import java.io._
import scala.collection.parallel.immutable.ParSeq
import scala.util.Random
import breeze.linalg._
import breeze.numerics._
import breeze.stats._

object Main {
  type Gene = DenseVector[Double]

  case class Data(x: DenseVector[Double], y: Double)

  def main(args: Array[String]) {
    val r = new Random()

    val dataCSV = new File("analyze.csv")
    val coefficientCSV = new File("coefficient.csv")

    val a = new BufferedReader(new FileReader(dataCSV))
    var b = a.readLine()

    while (b != null) {
      if (b.split(",").length != 18) {
        println(b)
      }
      b = a.readLine
    }

    val data: DenseMatrix[Double] = csvread(dataCSV)
    val size = data.rows

    val array = Array.ofDim[DenseVector[Double]](size)
    for (i <- 0 until size) {
      array(i) = data(i, ::).t
    }

    val testData = array.groupBy(_(1)).values.toList.map {
      _.map { d =>
        new Data(DenseVector.vertcat(d(2 until data.cols - 2), DenseVector(d(0))), d(data.cols - 2))
      }.toList.filter {
        case Data(x, _) =>
          x(4) == 1.0 || x(5) == 1.0
      }
    }.filter(_.nonEmpty)

    val raceMap: Map[Double, (Double, Double)] = testData.flatten.groupBy {
      case Data(x, y) =>
        makeRaceIdSoft(x)
    }.map {
      case (idx, arr) =>
        val times = arr.map(_.y)
        idx ->(mean(times), stddev(times))
    }


    val coefficient: Gene = csvread(coefficientCSV)(0, ::).t

    val outFile = new File("std.csv")
    val pw = new PrintWriter(outFile)

    try {
      testData.groupBy { list =>
        val Data(x, _) = list.last
        makeRaceIdSoft(x)
      }.toList.collect {
        case (id, list) if list.count(_.length > 3) > 30 =>
          id -> list.filter(_.length > 3)
      }.sortBy {
        case (id, _) =>
          id
      }.foreach {
        case (id, list) =>
          val errors = list.map { dataList =>
            calcDataListCost(raceMap, dataList, (x, y) => Math.abs(x - y), coefficient)
          }.toArray

          val times = list.map { dataList =>
            dataList.head.y
          }.toArray

          val timeMean = mean(times)
          val errorStd = stddev(errors)

          println(s"$id,$errorStd,$timeMean,${list.length}")
          pw.println(s"$id,$errorStd,$timeMean,${list.length}")
      }
    } catch {
      case ex: Exception =>
        ex.printStackTrace()
    } finally {
      pw.close()
    }
  }

  def findNearest(raceMap: Map[Double, (Double, Double)], vector: DenseVector[Double]): (Double, Double) = {
    val raceId = makeRaceIdSoft(vector)
    raceMap.minBy {
      case (idx, value) =>
        Math.abs(raceId - idx)
    }._2
  }

  def prePredict(raceMap: Map[Double, (Double, Double)], stdScore: Double, vector: DenseVector[Double]): Double = {
    val (m, s) = raceMap.getOrElse(makeRaceIdSoft(vector), findNearest(raceMap, vector))
    stdScore * s + m
  }

  def calcStdScore(raceMap: Map[Double, (Double, Double)], d: Data): Double = {
    val (m, s) = raceMap.getOrElse(makeRaceIdSoft(d.x), findNearest(raceMap, d.x))
    if (s == 0.0) {
      0
    } else {
      (d.y - m) / s
    }
  }

  def predict(prevScores: List[(Double, DenseVector[Double])],
              raceMap: Map[Double, (Double, Double)],
              d: Data,
              gene: Gene): (List[(Double, DenseVector[Double])], Double) = {
    val score = (calcStdScore(raceMap, d), d.x) :: prevScores
    val p = prevScores.foldLeft((0.0, 0.0)) {
      case ((scores, weights), (s, vector)) =>
        val distInv = 1.0 / vectorDistance(d.x, vector, gene)
        (scores + s * distInv, weights + distInv)
    }
    val out = prePredict(raceMap, if (prevScores.isEmpty) 0.0 else p._1 / p._2, d.x)

    (score, out)
  }

  def calcDataListCost(raceMap: Map[Double, (Double, Double)],
                       dataList: List[Data],
                       costFunction: (Double, Double) => Double,
                       gene: Gene) =
    dataList.foldLeft((Nil: List[(Double, DenseVector[Double])], 0.0)) {
      case ((prevScores, _), d) =>
        val (scores, out) = predict(prevScores, raceMap, d, gene)
        val cost = costFunction(d.y, out)
        (scores, cost)
    }._2

  def makeBabaId(vector: DenseVector[Double]): Double =
    vector(vector.length - 1) % 100


  def vectorDistance(
                      vector1: DenseVector[Double],
                      vector2: DenseVector[Double],
                      gene: Gene): Double = {
    100 +
      (if (makeBabaId(vector1) == makeBabaId(vector2)) 0 else 100) +
      Math.abs(vector1(3) - vector2(3)) * gene(0) +
      Math.abs(vector1(0) - vector2(0)) * gene(1) +
      (if (vector1(1) != vector2(1) || vector1(2) != vector2(2)) 1.0 else 0.0) * gene(2)
  }

  def makeRaceIdSoft(vector: DenseVector[Double]): Double =
    makeBabaId(vector) * 100000 + vector(3) * 10 + vector(1)

}