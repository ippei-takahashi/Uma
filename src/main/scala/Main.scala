import java.io._

import breeze.linalg._
import breeze.stats._

object Main {
  type Gene = DenseVector[Double]

  case class Data(x: DenseVector[Double], time: Double, raceId: Long, raceType: Long)

  case class PredictData(horseId: Double, raceType: Long, rank: Double, odds: Double, prevDataList: Seq[Data])

  case class CompetitionData(raceType: Long, horseData1: CompetitionHorseData, horseData2: CompetitionHorseData)

  case class CompetitionHorseData(no: Int, time: Double)

  private[this] val DEFAULT_RATE = 1500.0

  private[this] val raceTypeArray = Array[Long](
    101000,
    111000,
    101150,
    101200,
    111200,
    101300,
    101400,
    111400,
    101500,
    111500,
    101600,
    111600,
    101700,
    111700,
    101800,
    111800,
    101870,
    101900,
    102000,
    112000,
    102100,
    112200,
    102300,
    112300,
    102400,
    112400,
    102500,
    112500,
    112600,
    113000,
    113200,
    113400,
    113600
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
      case (horseId, arr) => horseId -> arr.map { d =>
        val raceId = d(data.cols - 1).toLong
        val x = d(1 until data.cols - 2)
        val raceType = makeRaceType(x)
        new Data(x, d(data.cols - 2), raceId, raceType)
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
              val head :: tail = subListBeforeRaceId(raceId.toLong, races)
              PredictData(horseId = vec(2), raceType = head.raceType, rank = vec(1), odds = vec(3), prevDataList = tail)
          }
        case _ => Array[PredictData]()
      })
    }.filter {
      case (raceId, arr) =>
        arr.nonEmpty
    }

    var oddsCount = 0.0
    var raceCount = 0
    var betCount = 0
    var betWinCount = 0
    var betLoseCount = 0

    raceMap.foreach {
      case (raceId, horses) =>
//        val extraHorses = (for {
//          horse <- horses
//          data <- horse.prevDataList.take(1)
//          extra <- raceMap.getOrElse(data.raceId, Array()).take(3) if !horses.map(_.horseId).contains(extra.horseId)
//        } yield extra).groupBy(_.horseId).values.map(_.head).toSeq
//
//        val allHorses = horses ++ extraHorses

        val allCompetitions = makeAllCompetitions(horses)
        val ratings = horses.map(_ => DEFAULT_RATE)
        val raceType = horses.head.raceType

        val sortedCompetitions = allCompetitions.sortBy(competitionData => -Math.abs(competitionData.raceType - raceType))

        sortedCompetitions.groupBy(_.raceType).foreach {
          case (thisRaceType, seq) =>
            val ratingUpdates = horses.map(_ => 0.0)
            seq.foreach {
              case CompetitionData(_, CompetitionHorseData(no1, time1), CompetitionHorseData(no2, time2)) =>
                val e1 = 1.0 / (1.0 + Math.pow(10.0, (ratings(no2) - ratings(no1)) / 400.0))
                val e2 = 1.0 / (1.0 + Math.pow(10.0, (ratings(no1) - ratings(no2)) / 400.0))
                val k = Math.max(4, 16 - Math.abs(thisRaceType - raceType) / 50)
                //val k = 16
                if (time1 < time2) {
                  ratingUpdates(no1) += k * (1.0 - e1)
                  ratingUpdates(no2) -= k * e2
                } else if (time1 > time2) {
                  ratingUpdates(no1) -= k * e1
                  ratingUpdates(no2) += k * (1.0 - e2)
                }
              case _ =>
            }
            ratings.indices.foreach {
              case index =>
                ratings(index) += ratingUpdates(index)
            }
        }

        raceCount += 1
        val sortedRatings = ratings.sortBy(-_)

        val m = mean(sortedRatings)
        val s = stddev(sortedRatings)

        val head = sortedRatings.head
        val second = sortedRatings(1)

        val headScore = (head - m) * 10 / s + 50
        val secondScore = (second - m) * 10 / s + 50

        val ratingTopIndex = ratings.zipWithIndex.maxBy(_._1)._2
        val ratingSecondIndex = ratings.zipWithIndex.filter(_._2 != ratingTopIndex).maxBy(_._1)._2
        val oddsTopIndex = horses.zipWithIndex.minBy(_._1.odds)._2
        val ratingTop = horses(ratingTopIndex)

        val directWin = allCompetitions.map {
          case CompetitionData(_, CompetitionHorseData(no1, time1), CompetitionHorseData(no2, time2)) =>
            if (no1 == ratingTopIndex && no2 == ratingSecondIndex)
              if (time1 < time2) 1 else -1
            else if (no1 == ratingSecondIndex && no2 == ratingTopIndex)
              if (time1 < time2) 1 else -1
            else
              0
        }.sum

        val directWinToOdds = allCompetitions.map {
          case CompetitionData(_, CompetitionHorseData(no1, time1), CompetitionHorseData(no2, time2)) =>
            if (no1 == ratingTopIndex && no2 == oddsTopIndex)
              if (time1 < time2) 1 else -1
            else if (no1 == oddsTopIndex && no2 == ratingTopIndex)
              if (time1 < time2) 1 else -1
            else
              0
        }.sum

        if (allCompetitions.length > 50 && ratingTop.odds > 1.2 && directWin > 0 && directWinToOdds > 0) {
          betCount += 1

          if (ratingTop.rank == 1.0) {
            oddsCount += ratingTop.odds
            betWinCount += 1
          } else {
            betLoseCount += 1
          }
        }
    }

    val rtn = oddsCount / betCount
    val p = betWinCount.toDouble / betCount.toDouble
    val r = oddsCount / betWinCount - 1.0
    val kf = ((r + 1) * p - 1) / r
    val g = Math.pow(Math.pow(1 + r * kf, p) * Math.pow(1 - kf, 1 - p), betCount)
    println(raceCount, oddsCount / betWinCount, betCount, betWinCount, betLoseCount, rtn, kf, g)
  }

  def makeAllCompetitions(horses: Array[PredictData]): Array[CompetitionData] =
    (for {
      raceType <- raceTypeArray
      i <- 0 until (horses.length - 1)
      time1 <- horses(i).prevDataList.filter(_.raceType == raceType).map(_.time).sorted.headOption.toSeq
      j <- (i + 1) until horses.length
      time2 <- horses(j).prevDataList.filter(_.raceType == raceType).map(_.time).sorted.headOption.toSeq
    } yield {
      val horseData1 = CompetitionHorseData(i, time1)
      val horseData2 = CompetitionHorseData(j, time2)
      CompetitionData(raceType, horseData1, horseData2)
    })


  def subListBeforeRaceId(raceId: Long, list: List[Data]): List[Data] = list match {
    case x :: xs if x.raceId == raceId =>
      x :: xs
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

  def makeRaceType(vector: DenseVector[Double]): Long =
    100000 + vector(1).toLong * 10000 + vector(3).toLong
}