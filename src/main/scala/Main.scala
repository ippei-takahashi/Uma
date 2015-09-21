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

    val dataCSV = new File("data.csv")
    val coefficientCSV = new File("coefficient.csv")

    val data: DenseMatrix[Double] = csvread(dataCSV)
    val size = data.rows

    val array = Array.ofDim[DenseVector[Double]](size)
    for (i <- 0 until size) {
      array(i) = data(i, ::).t
    }

    val testData = Random.shuffle(array.groupBy(_(0)).values.toList.map(_.reverseMap { d =>
      new Data(d(1 until data.cols - 1), d(data.cols - 1))
    }.toList))

    val raceMap: Map[Double, (Double, Double)] = testData.flatten.groupBy {
      case Data(x, y) =>
        makeRaceId(x)
    }.map {
      case (idx, arr) =>
        val times = arr.map(_.y)
        idx ->(mean(times), stddev(times))
    }

    val coefficient: Gene = csvread(coefficientCSV)(0, ::).t


    testData.groupBy { list =>
      val Data(x, _) = list.last
      makeRaceIdSoft(x)
    }.toList.collect {
      case (id, list) if list.count(_.length > 3) > 50 =>
        id -> list.filter(_.length > 3)
    }.sortBy {
      case (id, _) =>
        id
    }.foreach {
      case (id, list) =>
        val errors = list.map { dataList =>
          calcDataListCost(raceMap, dataList, (x, y) => Math.abs(x - y), coefficient)
        }.toArray

        val errorStd = stddev(errors)

        println(s"Id$id:Num = ${list.length}, ErrorStd = $errorStd")
    }
  }

  def findNearest(raceMap: Map[Double, (Double, Double)], vector: DenseVector[Double]): (Double, Double) = {
    val raceId = makeRaceId(vector)
    raceMap.minBy {
      case (idx, value) =>
        Math.abs(raceId - idx)
    }._2
  }

  def prePredict(raceMap: Map[Double, (Double, Double)], stdScore: Double, vector: DenseVector[Double]): Double = {
    val (m, s) = raceMap.getOrElse(makeRaceId(vector), findNearest(raceMap, vector))
    stdScore * s + m
  }

  def calcStdScore(raceMap: Map[Double, (Double, Double)], d: Data): Double = {
    val (m, s) = raceMap.getOrElse(makeRaceId(d.x), findNearest(raceMap, d.x))
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


  def vectorDistance(
                      vector1: DenseVector[Double],
                      vector2: DenseVector[Double],
                      gene: Gene): Double = {
    100 +
      Math.abs(vector1(3) - vector2(3)) * gene(0) +
      Math.abs(vector1(0) - vector2(0)) * gene(1) +
      (if (vector1(1) != vector2(1) || vector1(2) != vector2(2)) 1.0 else 0.0) * gene(2)
  }

  def makeRaceIdSoft(vector: DenseVector[Double]): Double =
    vector(3) * 1000 + vector(1) * 100

  def makeRaceId(vector: DenseVector[Double]): Double =
    vector(3) * 1000 + vector(1) * 100 + vector(4) * 30 + vector(5) * 20 + vector(6) * 10 + vector(8) * 3 + vector(9) * 2 + vector(10)
}