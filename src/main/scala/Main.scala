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
    val stdCSV = new File("std.csv")

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
            new Data(DenseVector.vertcat(d(2 until data.cols - 2), DenseVector(d(data.cols - 1), d(0))), d(data.cols - 2))
          }.toList)
      }
    }


    val coefficient: Gene = csvread(coefficientCSV)(0, ::).t

    val outFile = new File("raceWithStd.csv")
    val pw = new PrintWriter(outFile)

    try {
      raceMap.filter{
        case (_, map) =>
          map.values.toSeq.forall(_._2.nonEmpty) && map.values.toSeq.map(_._2.head.x).sortBy(x => x(x.length - 2)).foldLeft(0.0) {
            (acc, now) =>
              now(now.length - 2) - acc + 1.0
          } == 0.0
      }.foreach {
        case (raceId, map) =>
          val validMap = map.filter {
            case (_, (_, seq)) =>
              seq.count {
                case Data(x, _) =>
                  stdMap.get(makeRaceIdSoft(x)).isDefined
              } > 1 && stdMap.contains(makeRaceIdSoft(seq.head.x))
          }

          if (map.values.toSeq.length == validMap.values.toSeq.length) {
            validMap.toSeq.sortBy(_._2._2.head.y).zipWithIndex.foreach {
              case ((umaId, (odds, dataList)), index) =>
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
                val list = List(raceId, umaId, index + 1, odds, head.x(3), head.x(4), head.x(5), head.x(6), tail.length, predictTime)
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