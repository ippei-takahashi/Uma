import java.io._
import scala.collection.parallel.immutable.ParSeq
import scala.util.Random
import breeze.linalg._
import breeze.numerics._
import breeze.stats._

object Main {
  type Gene = DenseVector[Double]

  case class Data(x: DenseVector[Double], time: Double, raceId: Double)

  case class PredictData(id: Double, predictTime: Double, odds: Double)

  def main(args: Array[String]) {
    val r = new Random()

    val dataCSV = new File("20151012.csv")
    val coefficientCSV = new File("coefficient.csv")
    val stdCSV = new File("std.csv")

    val a = new BufferedReader(new FileReader(dataCSV))
    var b = a.readLine()

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
        id ->(arr.head(2), arr.head(1))
    }

    val raceMap = array.groupBy(_(0)).map {
      case (raceId, arr) => raceId -> arr.groupBy(_(1)).map {
        case (umaId, arr2) =>
          umaId ->(arr2.head(data.cols - 1), arr2.filter(x => x(10) == 1.0 || x(11) == 1.0).map { d =>
            new Data(d(2 until data.cols - 2), d(data.cols - 2), d(1))
          }.toList)
      }
    }

    val coefficient: Gene = csvread(coefficientCSV)(0, ::).t

    val outFile = new File("result.csv")
    val pw = new PrintWriter(outFile)

    try {
      raceMap.foreach {
        case (raceId, map) =>
          val validMap = map.filter {
            case (_, (_, seq)) =>
              seq.count {
                case Data(x, _, _) =>
                  stdMap.get(makeRaceIdSoft(x)).isDefined
              } > 1 && stdMap.contains(makeRaceIdSoft(seq.head.x))
          }

          if (map.values.toSeq.length == validMap.values.toSeq.length) {
            val pred = validMap.map {
              case (umaId, (odds, dataList)) =>
                val head :: tail = dataList
                val (scores, count, cost) =
                  calcDataListCost(stdMap, tail.reverse, (x, y, z) => {
                    val std = stdMap.get(makeRaceIdSoft(x))
                    if (std.isEmpty)
                      (0, 0.0)
                    else
                      (1, Math.abs(y - z) / std.get._2)
                  }, coefficient)
                val predictTime = predict(scores, stdMap, head, coefficient)._2
                (umaId, odds, predictTime)
            }.toArray
            val stdSorted = pred.sortBy(_._3)
            val oddsSorted = pred.sortBy(_._2)
            val stdAndOdds = stdSorted.slice(0, if (pred.length <= 12) 4 else 5).sortBy(_._2)

            val oddsHead = oddsSorted.head
            val stdAndOddsHead = stdAndOdds(1)

            val m: Double = mean(pred.map(_._3))
            val s: Double = stddev(pred.map(_._3))

            val oddsScore = (m - oddsHead._3) * 10 / s + 50
            val stdAndOddsScore = (m - stdAndOddsHead._3) * 10 / s + 50

            if (stdAndOddsScore > 65 && oddsScore < 60) {
              printf("%10d\n", raceId.toInt)
              stdAndOdds.foreach {
                x =>
                  printf("%10d, %f, %f\n", x._1.toInt, x._2, (m - x._3) * 10 / s + 50)
              }
              pw.println(raceId, stdAndOddsHead._1, stdAndOddsHead._2, stdAndOddsScore)
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
    case x :: xs if x.raceId == raceId =>
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
      (d.time - m) / s
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
        val (count, cost) = costFunction(d.x, d.time, out)

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