import java.io._

import breeze.linalg._
import breeze.stats._

object Main {
  type Gene = DenseVector[Double]

  case class Data(x: DenseVector[Double], time: Double, raceId: Long, raceType: Long, rank: Option[Int] = None)

  case class PredictData(horseId: Double, raceType: Long, rank: Int, odds: Double, oddsFuku: Double,age: Double, prevDataList: Seq[Data])

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
      }.toList
    }

    val race: DenseMatrix[Double] = csvread(raceCSV)
    val raceSize = race.rows

    val raceArray = Array.ofDim[DenseVector[Double]](raceSize)
    for (i <- 0 until raceSize) {
      raceArray(i) = race(i, ::).t
    }

    val raceMap_ = raceArray.groupBy(_ (0)).map {
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
              PredictData(horseId = vec(2), raceType = head.raceType, rank = vec(1).toInt, odds = vec(3), oddsFuku = vec(5),
                age = head.x(0), prevDataList = tail)
          }
        case _ => Array[PredictData]()
      })
    }.filter {
      case (raceId, arr) =>
        arr.nonEmpty
    }

    val raceMap = raceMap_.map {
      case (raceId, arr) =>
        raceId -> arr.map {
          pred =>
            pred.copy(prevDataList = pred.prevDataList.map {
              data =>
                data.copy(rank = raceMap_.get(data.raceId).flatMap(_.find(_.horseId == pred.horseId)).map(_.rank))
            })
        }
    }



    var oddsCount = 0.0
    var raceCount = 0
    var betCount = 0
    var betWinCount = 0
    var betLoseCount = 0

    raceMap.foreach {
      case (raceId, horses) =>

        val (competitionLength1, ratings1) = calcRatings(horses, makeAllCompetitions)
        val (competitionLength2, ratings2) = calcRatings(horses, makeAllCompetitionsRentai)

        val seq = ratings1.map(_._2).take(3).filter(r1 => ratings2.take(5).exists(_._2 == r1))

        seq.foreach {
          ratingTop =>
          if (competitionLength1 > 100 && competitionLength2 > 30 && ratingTop.odds < 20.0) {
            betCount += 1

            if (ratingTop.rank <= 2.0 || horses.length >= 8.0 && ratingTop.rank <= 3.0) {
              oddsCount += ratingTop.oddsFuku
              betWinCount += 1
            } else {
              betLoseCount += 1
            }
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

  def calcRatings(horses: Array[PredictData], makeCompetitions: Array[PredictData] => Array[CompetitionData]): (Int, Array[(Double, PredictData)]) = {
    val allCompetitions = makeCompetitions(horses)
    val ratings = horses.map(_ => DEFAULT_RATE)
    val raceType = horses.head.raceType

    val sortedCompetitions = allCompetitions.sortBy(competitionData => -Math.abs(competitionData.raceType - raceType))

    for (i <- 0 until 10) {
      sortedCompetitions.groupBy(_.raceType).foreach {
        case (thisRaceType, seq) =>
          val ratingUpdates = horses.map(_ => 0.0)
          seq.foreach {
            case CompetitionData(_, CompetitionHorseData(no1, time1), CompetitionHorseData(no2, time2)) =>
              val e1 = 1.0 / (1.0 + Math.pow(10.0, (ratings(no2) - ratings(no1)) / 400.0))
              val e2 = 1.0 / (1.0 + Math.pow(10.0, (ratings(no1) - ratings(no2)) / 400.0))
              val k = Math.max(4, 32 - Math.abs(thisRaceType - raceType) / 25)
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
    }

    allCompetitions.length -> ratings.zipWithIndex.sortBy(_._1).map {
      case (rating, index) =>
        rating -> horses(index)
    }
  }

  def makeAllCompetitionsRentai(horses: Array[PredictData]): Array[CompetitionData] =
    for {
      raceType <- raceTypeArray
      i <- 0 until (horses.length - 1)
      time1 <- horses(i).prevDataList.
        filter(_.raceType == raceType).
        filter(horses(i).age - _.x(0) < 25).
        filter(_.rank.getOrElse(100) <= 3).
        map(_.time).sorted.headOption.toSeq
      j <- (i + 1) until horses.length
      time2 <- horses(j).prevDataList.
        filter(_.raceType == raceType).
        filter(horses(j).age - _.x(0) < 25).
        filter(_.rank.getOrElse(100) <= 3).
        map(_.time).sorted.headOption.toSeq
    } yield {
      val horseData1 = CompetitionHorseData(i, time1)
      val horseData2 = CompetitionHorseData(j, time2)
      CompetitionData(raceType, horseData1, horseData2)
    }

  def makeAllCompetitions(horses: Array[PredictData]): Array[CompetitionData] =
    for {
      raceType <- raceTypeArray
      i <- 0 until (horses.length - 1)
      time1 <- horses(i).prevDataList.
        filter(_.raceType == raceType).
        filter(horses(i).age - _.x(0) < 25).
        map(_.time).sorted.headOption.toSeq
      j <- (i + 1) until horses.length
      time2 <- horses(j).prevDataList.
        filter(_.raceType == raceType).
        filter(horses(j).age - _.x(0) < 25).
        map(_.time).sorted.headOption.toSeq
    } yield {
      val horseData1 = CompetitionHorseData(i, time1)
      val horseData2 = CompetitionHorseData(j, time2)
      CompetitionData(raceType, horseData1, horseData2)
    }


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