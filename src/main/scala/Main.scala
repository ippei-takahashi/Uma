import java.io._

import breeze.linalg._
import breeze.stats._

object Main {
  type Gene = DenseVector[Double]

  case class Data(x: DenseVector[Double], time: Double, raceId: Long, raceType: Long, isGoodBaba: Boolean, rank: Option[Int] = None)

  case class PredictData(horseId: Int, raceType: Long, rank: Int, odds: Double, oddsFuku: Double, age: Double,
                         isGoodBaba: Boolean, prevDataList: Seq[Data])

  private[this] val ratingMapDShort = scala.collection.mutable.Map[Int, (Double, Int)]()

  private[this] val ratingMapDMiddle = scala.collection.mutable.Map[Int, (Double, Int)]()

  private[this] val ratingMapDSemiLong = scala.collection.mutable.Map[Int, (Double, Int)]()

  private[this] val ratingMapDLong = scala.collection.mutable.Map[Int, (Double, Int)]()

  private[this] val ratingMapSShort = scala.collection.mutable.Map[Int, (Double, Int)]()

  private[this] val ratingMapSMiddle = scala.collection.mutable.Map[Int, (Double, Int)]()

  private[this] val ratingMapSSemiLong = scala.collection.mutable.Map[Int, (Double, Int)]()

  private[this] val ratingMapSLong = scala.collection.mutable.Map[Int, (Double, Int)]()

  private[this] val DEFAULT_RATE = 1500.0

  case class CompetitionData(raceType: Long, horseData1: CompetitionHorseData, horseData2: CompetitionHorseData)

  case class CompetitionHorseData(no: Int, time: Double)

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
        new Data(x, d(data.cols - 2), raceId, raceType, isGoodBaba = x(4) + x(5) == 1.0 && x(8) == 1.0)
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
              PredictData(horseId = vec(2).toInt, raceType = head.raceType, rank = vec(1).toInt, odds = vec(3), oddsFuku = vec(5),
                age = head.x(0), isGoodBaba = head.isGoodBaba, prevDataList = tail)
          }
        case _ => Array[PredictData]()
      })
    }.filter {
      case (_, arr) =>
        arr.nonEmpty
    }.map {
      case (raceId, arr) =>
        raceId -> arr.sortBy(_.rank)
    }
    val raceSeq_ = raceMap_.toSeq.sortBy(_._1)

    val raceSeq = raceSeq_.map {
      case (raceId, arr) =>
        raceId -> arr.map {
          pred =>
            pred.copy(prevDataList = pred.prevDataList.map {
              prevData =>
                prevData.copy(rank = raceMap_.get(prevData.raceId).flatMap {
                  _.find(_.horseId == pred.horseId)
                }.map(_.rank))
            })
        }
    }

    var oddsCount = 0.0
    var raceCount = 0
    var betCount = 0
    var betWinCount = 0
    val analyzeArray = Array.ofDim[Int](20)

    val num1Range = 100000000
    val num2Range = 10000
    val num3Range = 100

    val ranges = for {
      num1 <- 500000000 to 1400000000 by num1Range
      num2 <- 10000 to 50000 by num2Range
      num3 <- 100 to 800 by num3Range
    } yield {
        (raceId: Double) =>
          raceId >= num1 &&
            raceId < num1 + num1Range &&
            (raceId % (num2Range * 10)) >= num2 &&
            (raceId % (num2Range * 10)) < num2 + num2Range &&
            (raceId % (num3Range * 10)) >= num3 &&
            (raceId % (num3Range * 10)) < num3 + num3Range
      }

    for {
      ri <- 0 until (ranges.length - 1)
    } {
      for (loop <- 0 until 10) {
        raceSeq.filter {
          case (raceId, _) =>
            ranges(ri)(raceId)
        }.foreach {
          case (raceId, horses) =>
            val ratingUpdates = horses.map(_ => 0.0)
            val ratingCountUpdates = horses.map(_ => 0)

            val ratingMap = getRatingMap(horses.head.raceType)
            val (ratings, ratingCounts) = horses.map {
              horse =>
                ratingMap.getOrElse(horse.horseId, (DEFAULT_RATE, 0))
            }.unzip

            for {
              i <- 0 until 3
              j <- (i + 1) until horses.length
            } {
              val e1 = 1.0 / (1.0 + Math.pow(10.0, (ratings(j) - ratings(i)) / 400.0))
              val e2 = 1.0 / (1.0 + Math.pow(10.0, (ratings(i) - ratings(j)) / 400.0))
              val k = 16

              ratingUpdates(i) += k * (1.0 - e1)
              ratingUpdates(j) -= k * e2

              ratingCountUpdates(i) += 1
              ratingCountUpdates(j) += 1
            }

            horses.zipWithIndex.foreach {
              case (horse, index) =>
                ratingMap.put(horse.horseId, (ratings(index) + ratingUpdates(index), ratingCounts(index) + ratingCountUpdates(index)))
            }
        }
      }

      raceSeq.filter {
        case (raceId, arr) =>
          ranges(ri + 1)(raceId)
      }.foreach {
        case (raceId, horses) =>
          val raceType = horses.head.raceType

          val ratingMap = getRatingMap(raceType)
          val ratingInfo = horses.map {
            horse =>
              horse -> ratingMap.getOrElse(horse.horseId, (DEFAULT_RATE, 0))
          }
          val newRatingInfo = ratingInfo.sortBy(-_._2._1).zipWithIndex.map {
            case ((horse, (rating, ratingCount)), index) =>
              (horse, rating, ratingCount, index)
          }

          val newRatingInfoOdds = ratingInfo.sortBy(_._1.odds).zipWithIndex.map {
            case ((horse, (rating, ratingCount)), index) =>
              (horse, rating, ratingCount, index)
          }

          val newRatingInfoTime = ratingInfo.sortBy(
            _._1.prevDataList.filter(_.raceType == raceType).map(_.time).sorted.headOption.getOrElse(Double.MaxValue)
          ).zipWithIndex.map {
            case ((horse, (rating, ratingCount)), index) =>
              (horse, rating, ratingCount, index)
          }

          val newRatingInfoScore = newRatingInfo.map {
            case (horse, rating, ratingCount, _) =>
              val indexTime = newRatingInfoTime.find(_._1.horseId == horse.horseId).get._4
              val score = rating +
                (indexTime match {
                  case 0 => 20
                  case 1 => 15
                  case 2 => 10
                  case 3 => 5
                  case 4 => 5
                  case _ => 0
                })
              (horse, score, ratingCount)
          }.sortBy(-_._2).zipWithIndex.map {
            case ((horse, rating, ratingCount), index) =>
              (horse, rating, ratingCount, index)
          }

          val ratingTop = newRatingInfo.head
          val ratingSecond = newRatingInfo(1)

          raceCount += 1

          val sortedScores = newRatingInfoScore.sortBy(-_._2)
          val scoreDiff = sortedScores.head._2 - sortedScores(1)._2
          val scoreDiff2 = sortedScores.head._2 - sortedScores(2)._2
          val scoreDiff3 = sortedScores.head._2 - sortedScores(3)._2

          val predictOdds = (1 + Math.pow(10, -scoreDiff / 400)) *
            (1 + Math.pow(10, -scoreDiff2 / 400)) *
            (1 + Math.pow(10, -scoreDiff3 / 400)) *
            6.5

          if (sortedScores.head._3 > 0 && predictOdds < sortedScores.head._1.odds) {
            betCount += 1
            if (sortedScores.head._1.rank <= 2 || (sortedScores.head._1.rank <= 3 && horses.length >= 8)) {
              betWinCount += 1
              oddsCount += sortedScores.head._1.oddsFuku
            }
          }
      }
    }


    val rtn = oddsCount / betCount
    val p = betWinCount.toDouble / betCount.toDouble
    val r = oddsCount / betWinCount - 1.0
    val kf = ((r + 1) * p - 1) / r
    val g = Math.pow(Math.pow(1 + r * kf, p) * Math.pow(1 - kf, 1 - p), betCount)
    println(raceCount, oddsCount / betWinCount, betCount, betWinCount, betWinCount.toDouble / betCount.toDouble, rtn, kf, g)
  }

  def subListBeforeRaceId(raceId: Long, list: List[Data]): List[Data] = list match {
    case x :: xs if x.raceId == raceId =>
      x :: xs
    case _ :: xs =>
      subListBeforeRaceId(raceId, xs)
    case _ =>
      Nil
  }

  def makeAllCompetitions(horses: Array[PredictData]): Array[CompetitionData] =
    for {
      raceType <- raceTypeArray
      i <- 0 until (horses.length - 1)
      time1 <- horses(i).prevDataList.filter(_.raceType == raceType).filter(horses(i).age - _.x(0) < 25).map(_.time).sorted.headOption.toSeq
      j <- (i + 1) until horses.length
      time2 <- horses(j).prevDataList.filter(_.raceType == raceType).filter(horses(j).age - _.x(0) < 25).map(_.time).sorted.headOption.toSeq
    } yield {
      val horseData1 = CompetitionHorseData(i, time1)
      val horseData2 = CompetitionHorseData(j, time2)
      CompetitionData(raceType, horseData1, horseData2)
    }

  def getRatingMap(raceType: Long): scala.collection.mutable.Map[Int, (Double, Int)] =
    (raceType / 10000, raceType % 10000) match {
      case (10, dist) if dist <= 1200 =>
        ratingMapDShort
      case (10, dist) if dist <= 1500 =>
        ratingMapDMiddle
      case (10, dist) if dist <= 1800 =>
        ratingMapDSemiLong
      case (10, _) =>
        ratingMapDLong
      case (11, dist) if dist <= 1200 =>
        ratingMapSShort
      case (11, dist) if dist <= 1500 =>
        ratingMapSMiddle
      case (11, dist) if dist <= 1800 =>
        ratingMapSSemiLong
      case (11, _) =>
        ratingMapSLong
    }


  def makeRaceType(vector: DenseVector[Double]): Long =
    100000 + vector(1).toLong * 10000 + vector(3).toLong
}