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

    val raceCSV = new File("raceWithStd.csv")

    val raceData: DenseMatrix[Double] = csvread(raceCSV)
    val raceSize = raceData.rows

    val raceArray = Array.ofDim[DenseVector[Double]](raceSize)
    for (i <- 0 until raceSize) {
      raceArray(i) = raceData(i, ::).t
    }

    val raceMap = raceArray.groupBy(_(0))

    var oddsCount = 0.0
    var raceCount = 0
    var over60Count = 0
    var over60WinCount = 0
    var over60LoseCount = 0

    raceMap.foreach {
      case (raceId, arr) =>
        val stdSorted = arr.sortBy(_(5))
        val oddsSorted = arr.sortBy(_(3))

        val m: Double = mean(stdSorted.map(_(5)))
        val s: Double = stddev(stdSorted.map(_(5)))

        val stdAndOdds = stdSorted.slice(0, if (arr.length <= 12) 4 else 5).sortBy(_(3))
        val stdAndOddsHead = stdAndOdds.head
        val stdAndOddsSecond = stdAndOdds(1)
        val stdAndOddsThird = stdAndOdds(2)


        val stdHead = stdSorted.head
        val oddsHead = oddsSorted.head
        val oddsSecond = oddsSorted(1)

        val stdScore = (m - stdHead(5)) * 10 / s + 50
        val oddsScore = (m - oddsHead(5)) * 10 / s + 50
        val oddsSecondScore = (m - oddsHead(5)) * 10 / s + 50
        val stdAndOddsHeadScore = (m - stdAndOddsHead(5)) * 10 / s + 50
        val stdAndOddsSecondScore = (m - stdAndOddsSecond(5)) * 10 / s + 50
        val stdAndOddsThirdScore = (m - stdAndOddsThird(5)) * 10 / s + 50

        raceCount += 1
        if (stdScore > 65 && oddsScore < 60) {
          over60Count += 1
          if (stdHead(2) == 1.0) {
            oddsCount += stdHead(3)
            over60WinCount += 1
          } else {
            over60LoseCount += 1
          }
        }
    }
    val rtn = oddsCount / over60WinCount * over60WinCount / over60Count
    println(raceCount, oddsCount / over60WinCount, over60Count, over60WinCount, over60LoseCount, rtn)
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