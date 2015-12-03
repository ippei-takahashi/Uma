import java.io._

import breeze.linalg._
import breeze.stats._

object Main {
  type Gene = DenseVector[Double]

  case class Data(x: DenseVector[Double], time: Double, raceId: Long, babaId: Long)

  case class PredictData(rank: Double, odds: Double, prevDataList: Seq[Data])

  case class CompetitionData(umaData1: CompetitionUmaData, umaData2: CompetitionUmaData)

  case class CompetitionUmaData(no: Int, score: Option[Double])

  private[this] val raceIdArray = Array(
    10000,
    10001,
    11500,
    12000,
    12001,
    13000,
    14000,
    14001,
    15000,
    15001,
    16000,
    16001,
    17000,
    17001,
    18000,
    18001,
    18700,
    19000,
    20000,
    20001,
    21000,
    22001,
    23000,
    23001,
    24000,
    24001,
    25000,
    25001,
    26001,
    30001,
    32001,
    34001,
    36001
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

    val dataMap = array.groupBy(_ (0)).map {
      case (umaId, arr) => umaId -> arr.map { d =>
        val raceId = d(data.cols - 1).toLong
        val babaId = (raceId / 100000) % 1000
        new Data(d(1 until data.cols - 2), d(data.cols - 2), raceId, babaId)
      }.toList.filter {
        case Data(x, _, _, _) =>
          x(4) == 1.0
      }
    }

    val race: DenseMatrix[Double] = csvread(raceCSV)
    val raceSize = race.rows

    val raceArray = Array.ofDim[DenseVector[Double]](raceSize)
    for (i <- 0 until raceSize) {
      raceArray(i) = race(i, ::).t
    }

    val raceMap = raceArray.groupBy(_ (0)).map {
      case (raceId, arr) => raceId -> (arr match {
        case validArray if validArray.forall(vec => dataMap.get(vec(2)) match {
          case Some(races) =>
            subListBeforeRaceId(raceId.toLong, races).nonEmpty
          case _ =>
            false
        }) =>
          validArray.map {
            vec =>
              val races = dataMap(vec(2))
              val tail = subListBeforeRaceId(raceId.toLong, races)
              PredictData(rank = vec(1), odds = vec(3), prevDataList = tail)
          }
        case _ => Array[PredictData]()
      })
    }.filter {
      case (raceId, arr) =>
        arr.nonEmpty
    }

    raceMap

    //    var oddsCount = 0.0
    //    var raceCount = 0
    //    var over60Count = 0
    //    var over60WinCount = 0
    //    var over60LoseCount = 0
    //
    //    raceMap.foreach {
    //      case (raceId, arr) =>
    //        val stdSorted = arr.sortBy(_ (8))
    //        val oddsSorted = arr.sortBy(_ (3))
    //
    //        val m: Double = mean(stdSorted.map(_ (8)))
    //        val s: Double = stddev(stdSorted.map(_ (8)))
    //
    //        val stdAndOdds = stdSorted.slice(0, if (arr.length <= 12) 4 else 5).sortBy(_ (3))
    //        val stdAndOddsHead = stdAndOdds.head
    //        val stdAndOddsSecond = stdAndOdds(1)
    //        val stdAndOddsThird = stdAndOdds(2)
    //
    //
    //        val stdHead = stdSorted.head
    //        val oddsHead = oddsSorted.head
    //        val oddsSecond = oddsSorted(1)
    //
    //        val stdScore = (m - stdHead(8)) * 10 / s + 50
    //        val oddsScore = (m - oddsHead(8)) * 10 / s + 50
    //        val oddsSecondScore = (m - oddsHead(8)) * 10 / s + 50
    //        val stdAndOddsHeadScore = (m - stdAndOddsHead(8)) * 10 / s + 50
    //        val stdAndOddsSecondScore = (m - stdAndOddsSecond(8)) * 10 / s + 50
    //        val stdAndOddsThirdScore = (m - stdAndOddsThird(8)) * 10 / s + 50
    //
    //        raceCount += 1
    //        if (stdScore > 70 && oddsScore < 60 && s != 0 && stdHead(3) > 5) {
    //          over60Count += 1
    //          if (stdHead(2) == 1.0) {
    //            oddsCount += stdHead(3)
    //            over60WinCount += 1
    //          } else {
    //            over60LoseCount += 1
    //          }
    //        }
    //    }
    //    val rtn = oddsCount / over60Count
    //    val p = over60WinCount.toDouble / over60Count.toDouble
    //    val r = oddsCount / over60WinCount - 1.0
    //    val kf = ((r + 1) * p - 1) / r
    //    val g = Math.pow(Math.pow(1 + r * kf, p) * Math.pow(1 - kf, 1 - p), over60Count)
    //    println(raceCount, oddsCount / over60WinCount, over60Count, over60WinCount, over60LoseCount, rtn, kf, g)
  }

  def makeAllCompetition(races: Array[PredictData]): Array[CompetitionData] =
    for {
      raceId <- raceIdArray
      i <- races.indices
      j <- i until races.length
    } yield {
      val umaData1 = i -> races(i)
    }


  def subListBeforeRaceId(raceId: Long, list: List[Data]): List[Data] = list match {
    case x :: xs if x.raceId == raceId =>
      xs
    case _ :: xs =>
      subListBeforeRaceId(raceId, xs)
    case _ =>
      Nil
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