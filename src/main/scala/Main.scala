import java.io._

import breeze.linalg._
import breeze.stats._

object Main {
  type Gene = DenseVector[Double]

  case class Data(x: DenseVector[Double], time: Double, raceId: Long, babaId: Long)

  private[this] val raceIdArray = Array(
    1010000, 2010000, 3010000, 4010000, 5010000, 6010000, 7010000, 8010000, 9010000, 10010000,
    1010001, 2010001, 3010001, 4010001, 5010001, 6010001, 7010001, 8010001, 9010001, 10010001,
    1011500, 2011500, 3011500, 4011500, 5011500, 6011500, 7011500, 8011500, 9011500, 10011500,
    1012000, 2012000, 3012000, 4012000, 5012000, 6012000, 7012000, 8012000, 9012000, 10012000,
    1012001, 2012001, 3012001, 4012001, 5012001, 6012001, 7012001, 8012001, 9012001, 10012001,
    1013000, 2013000, 3013000, 4013000, 5013000, 6013000, 7013000, 8013000, 9013000, 10013000,
    1014000, 2014000, 3014000, 4014000, 5014000, 6014000, 7014000, 8014000, 9014000, 10014000,
    1014001, 2014001, 3014001, 4014001, 5014001, 6014001, 7014001, 8014001, 9014001, 10014001,
    1015000, 2015000, 3015000, 4015000, 5015000, 6015000, 7015000, 8015000, 9015000, 10015000,
    1015001, 2015001, 3015001, 4015001, 5015001, 6015001, 7015001, 8015001, 9015001, 10015001,
    1016000, 2016000, 3016000, 4016000, 5016000, 6016000, 7016000, 8016000, 9016000, 10016000,
    1016001, 2016001, 3016001, 4016001, 5016001, 6016001, 7016001, 8016001, 9016001, 10016001,
    1017000, 2017000, 3017000, 4017000, 5017000, 6017000, 7017000, 8017000, 9017000, 10017000,
    1017001, 2017001, 3017001, 4017001, 5017001, 6017001, 7017001, 8017001, 9017001, 10017001,
    1018000, 2018000, 3018000, 4018000, 5018000, 6018000, 7018000, 8018000, 9018000, 10018000,
    1018001, 2018001, 3018001, 4018001, 5018001, 6018001, 7018001, 8018001, 9018001, 10018001,
    1018700, 2018700, 3018700, 4018700, 5018700, 6018700, 7018700, 8018700, 9018700, 10018700,
    1019000, 2019000, 3019000, 4019000, 5019000, 6019000, 7019000, 8019000, 9019000, 10019000,
    1020000, 2020000, 3020000, 4020000, 5020000, 6020000, 7020000, 8020000, 9020000, 10020000,
    1020001, 2020001, 3020001, 4020001, 5020001, 6020001, 7020001, 8020001, 9020001, 10020001,
    1021000, 2021000, 3021000, 4021000, 5021000, 6021000, 7021000, 8021000, 9021000, 10021000,
    1022001, 2022001, 3022001, 4022001, 5022001, 6022001, 7022001, 8022001, 9022001, 10022001,
    1023000, 2023000, 3023000, 4023000, 5023000, 6023000, 7023000, 8023000, 9023000, 10023000,
    1023001, 2023001, 3023001, 4023001, 5023001, 6023001, 7023001, 8023001, 9023001, 10023001,
    1024000, 2024000, 3024000, 4024000, 5024000, 6024000, 7024000, 8024000, 9024000, 10024000,
    1024001, 2024001, 3024001, 4024001, 5024001, 6024001, 7024001, 8024001, 9024001, 10024001,
    1025000, 2025000, 3025000, 4025000, 5025000, 6025000, 7025000, 8025000, 9025000, 10025000,
    1025001, 2025001, 3025001, 4025001, 5025001, 6025001, 7025001, 8025001, 9025001, 10025001,
    1026001, 2026001, 3026001, 4026001, 5026001, 6026001, 7026001, 8026001, 9026001, 10026001,
    1030001, 2030001, 3030001, 4030001, 5030001, 6030001, 7030001, 8030001, 9030001, 10030001,
    1032001, 2032001, 3032001, 4032001, 5032001, 6032001, 7032001, 8032001, 9032001, 10032001,
    1034001, 2034001, 3034001, 4034001, 5034001, 6034001, 7034001, 8034001, 9034001, 10034001,
    1036001, 2036001, 3036001, 4036001, 5036001, 6036001, 7036001, 8036001, 9036001
  )

  def main(args: Array[String]) {

    val dataCSV = new File("data.csv")
    val coefficientCSV = new File("coefficient.csv")
    val raceCSV = new File("race.csv")

    val data: DenseMatrix[Double] = csvread(dataCSV)
    val size = data.rows

    val array = Array.ofDim[DenseVector[Double]](size)
    for (i <- 0 until size) {
      array(i) = data(i, ::).t
    }

    val testData = array.groupBy(_ (0)).map {
      case (id, arr) => id -> arr.map { d =>
        val raceId = d(data.cols - 1).toLong
        val babaId = (raceId / 100000) % 1000
        new Data(d(1 until data.cols - 2), d(data.cols - 2), raceId, babaId)
      }.toList.filter {
        case Data(x, _, _) =>
          x(4) == 1.0
      }
    }

    var oddsCount = 0.0
    var raceCount = 0
    var over60Count = 0
    var over60WinCount = 0
    var over60LoseCount = 0

    raceMap.foreach {
      case (raceId, arr) =>
        val stdSorted = arr.sortBy(_ (8))
        val oddsSorted = arr.sortBy(_ (3))

        val m: Double = mean(stdSorted.map(_ (8)))
        val s: Double = stddev(stdSorted.map(_ (8)))

        val stdAndOdds = stdSorted.slice(0, if (arr.length <= 12) 4 else 5).sortBy(_ (3))
        val stdAndOddsHead = stdAndOdds.head
        val stdAndOddsSecond = stdAndOdds(1)
        val stdAndOddsThird = stdAndOdds(2)


        val stdHead = stdSorted.head
        val oddsHead = oddsSorted.head
        val oddsSecond = oddsSorted(1)

        val stdScore = (m - stdHead(8)) * 10 / s + 50
        val oddsScore = (m - oddsHead(8)) * 10 / s + 50
        val oddsSecondScore = (m - oddsHead(8)) * 10 / s + 50
        val stdAndOddsHeadScore = (m - stdAndOddsHead(8)) * 10 / s + 50
        val stdAndOddsSecondScore = (m - stdAndOddsSecond(8)) * 10 / s + 50
        val stdAndOddsThirdScore = (m - stdAndOddsThird(8)) * 10 / s + 50

        raceCount += 1
        if (stdScore > 70 && oddsScore < 60 && s != 0 && stdHead(3) > 5) {
          over60Count += 1
          if (stdHead(2) == 1.0) {
            oddsCount += stdHead(3)
            over60WinCount += 1
          } else {
            over60LoseCount += 1
          }
        }
    }
    val rtn = oddsCount / over60Count
    val p = over60WinCount.toDouble / over60Count.toDouble
    val r = oddsCount / over60WinCount - 1.0
    val kf = ((r + 1) * p - 1) / r
    val g = Math.pow(Math.pow(1 + r * kf, p) * Math.pow(1 - kf, 1 - p), over60Count)
    println(raceCount, oddsCount / over60WinCount, over60Count, over60WinCount, over60LoseCount, rtn, kf, g)
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

  def makeRaceId(vector: DenseVector[Double], babaId: Long): Long =
    babaId * 100000 + vector(3).toLong * 10 + vector(1).toLong
}