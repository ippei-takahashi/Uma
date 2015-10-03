import java.io._
import scala.collection.parallel.immutable.ParSeq
import scala.util.Random
import breeze.linalg._
import breeze.numerics._
import breeze.stats._

object Main {
  type Gene = DenseVector[Double]

  case class Data(x: DenseVector[Double], y: Double, z: Double)

  def main(args: Array[String]) {
    val r = new Random()

    val dataCSV = new File("data.csv")
    val coefficientCSV = new File("coefficient.csv")
    val raceCSV = new File("race.csv")
    val stdCSV = new File("std.csv")

    val data: DenseMatrix[Double] = csvread(dataCSV)
    val size = data.rows

    val array = Array.ofDim[DenseVector[Double]](size)
    for (i <- 0 until size) {
      array(i) = data(i, ::).t
    }

    val stdData: DenseMatrix[Double] = csvread(stdCSV)
    val stdSize = stdData.rows

    val stdArray = Array.ofDim[DenseVector[Double]](stdSize)
    for (i <- 0 until stdSize) {
      stdArray(i) = stdData(i, ::).t
    }
    val stdMap = stdArray.groupBy(_(0)).map {
      case (id, arr) =>
        id -> arr.head(1)
    }

    val testData = array.groupBy(_(0)).map {
      case (id, arr) => id -> arr.map { d =>
        new Data(d(1 until data.cols - 2), d(data.cols - 2), d(data.cols - 1))
      }.toList.filter {
        case Data(x, _, _) =>
          x(4) == 1.0
      }
    }

    val timeMap: Map[Double, (Double, Double)] = testData.values.flatten.groupBy {
      case Data(x, _, _) =>
        makeRaceIdSoft(x)
    }.map {
      case (idx, arr) =>
        val times = arr.map(_.y)
        idx ->(mean(times), stddev(times))
    }

    val coefficient: Gene = csvread(coefficientCSV)(0, ::).t

    val raceData: DenseMatrix[Double] = csvread(raceCSV)
    val raceSize = raceData.rows

    val raceArray = Array.ofDim[DenseVector[Double]](raceSize)
    for (i <- 0 until raceSize) {
      raceArray(i) = raceData(i, ::).t
    }

    val raceMap = raceArray.groupBy(_(0))

    val outFile = new File("raceWithStd.csv")
    val pw = new PrintWriter(outFile)

    try {
      raceMap.foreach {
        case (raceId, arr) =>
          val arrWithData = arr.map { vec =>
            vec -> testData.get(vec(2))
          }.collect {
            case (vec, Some(dataList)) =>
              vec -> subListBeforeRaceId(raceId, dataList)
          }.filter {
            case (_, list) =>
              list.count{
                case Data(x, _, _) =>
                  stdMap.get(makeRaceIdSoft(x)).isDefined
              } > 1 && stdMap.contains(makeRaceIdSoft(list.head.x))
          }

          if (arr.length == arrWithData.length) {
            arrWithData.foreach {
              case (vec, dataList) =>
                val head :: tail = dataList
                val (scores, count, cost) =
                  calcDataListCost(timeMap, tail.reverse, (x, y, z) => {
                    val std = stdMap.get(makeRaceIdSoft(x))
                    if (std.isEmpty)
                      (0, 0.0)
                    else
                      (1, Math.abs(y - z) / std.get)
                  }, coefficient)
              val std = stdMap(makeRaceIdSoft(head.x)) * (1.0 + (cost / (count * 10.0))) / 1.1
              val predictTime = predict(scores, timeMap, head, coefficient)._2
              val list = List(vec(0), vec(2), vec(1), vec(3), vec(4), vec(5), vec(6), tail.length, predictTime, std)
              pw.println(list.mkString(","))
            }
          }
      }
    } catch {
      case ex: Exception =>
        ex.printStackTrace()
    } finally {
      pw.close()
    }
  }

  def subListBeforeRaceId(raceId: Double, list: List[Data]): List[Data] = list match {
    case x :: xs if x.z == raceId =>
      x :: xs
    case _ :: xs =>
      subListBeforeRaceId(raceId, xs)
    case _ =>
      Nil
  }

  def findNearest(timeMap: Map[Double, (Double, Double)], vector: DenseVector[Double]): (Double, Double) = {
    val raceId = makeRaceIdSoft(vector)
    timeMap.minBy {
      case (idx, value) =>
        Math.abs(raceId - idx)
    }._2
  }

  def prePredict(timeMap: Map[Double, (Double, Double)], stdScore: Double, vector: DenseVector[Double]): Double = {
    val (m, s) = timeMap.getOrElse(makeRaceIdSoft(vector), findNearest(timeMap, vector))
    stdScore * s + m
  }

  def calcStdScore(timeMap: Map[Double, (Double, Double)], d: Data): Double = {
    val (m, s) = timeMap.getOrElse(makeRaceIdSoft(d.x), findNearest(timeMap, d.x))
    if (s == 0.0) {
      0
    } else {
      (d.y - m) / s
    }
  }

  def predict(prevScores: List[(Double, DenseVector[Double])],
              timeMap: Map[Double, (Double, Double)],
              d: Data,
              gene: Gene): (List[(Double, DenseVector[Double])], Double) = {
    val score = (calcStdScore(timeMap, d), d.x) :: prevScores
    val p = prevScores.foldLeft((0.0, 0.0)) {
      case ((scores, weights), (s, vector)) =>
        val distInv = 1.0 / vectorDistance(d.x, vector, gene)
        (scores + s * distInv, weights + distInv)
    }
    val out = prePredict(timeMap, if (prevScores.isEmpty) 0.0 else p._1 / p._2, d.x)

    (score, out)
  }

  def calcDataListCost(timeMap: Map[Double, (Double, Double)],
                       dataList: List[Data],
                       costFunction: (DenseVector[Double], Double, Double) => (Int, Double),
                       gene: Gene) = {
    dataList.foldLeft((Nil: List[(Double, DenseVector[Double])], 0, 0.0)) {
      case ((prevScores, prevCount, prevCost), d) =>
        val (scores, out) = predict(prevScores, timeMap, d, gene)
        val (count, cost) = costFunction(d.x, d.y, out)

        val newCount = prevCount + count
        val newCost = prevCost + cost

        val newScores = if (count > 0) scores else prevScores

        (newScores, newCount, newCost)
    }
  }


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
}