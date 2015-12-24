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

    Seq(500000000,
      600000000,
      700000000,
      800000000,
      900000000,
      1000000000,
      1100000000,
      1200000000,
      1300000000).foreach {
      num =>
        for (loop <- 0 until 10) {
          raceSeq.filter {
            case (raceId, _) =>
              raceId >= num & raceId < num + 100000000
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
            raceId >= num + 100000000 && raceId < num + 200000000 && (raceId % 100000) < 20000 && arr.head.isGoodBaba
        }.foreach {
          case (raceId, horses) =>
            //            val allCompetitions = makeAllCompetitions(horses)
            val raceType = horses.head.raceType

            //            val sortedCompetitions = allCompetitions.sortBy(competitionData => -Math.abs(competitionData.raceType - raceType))

            val ratingMap = getRatingMap(raceType)
            val ratingInfo = horses.map {
              horse =>
                horse -> ratingMap.getOrElse(horse.horseId, (DEFAULT_RATE, 0))
            }
            //            val ratings = ratingInfo.map(_._2._1)
            //
            //            sortedCompetitions.groupBy(_.raceType).foreach {
            //              case (thisRaceType, seq) =>
            //                val ratingUpdates = horses.map(_ => 0.0)
            //                seq.foreach {
            //                  case CompetitionData(_, CompetitionHorseData(no1, time1), CompetitionHorseData(no2, time2)) =>
            //                    val e1 = 1.0 / (1.0 + Math.pow(10.0, (ratings(no2) - ratings(no1)) / 400.0))
            //                    val e2 = 1.0 / (1.0 + Math.pow(10.0, (ratings(no1) - ratings(no2)) / 400.0))
            //                    val k = Math.max(4, 16 - Math.abs(thisRaceType - raceType) / 50)
            //                    //val k = 16
            //                    if (time1 < time2) {
            //                      ratingUpdates(no1) += k * (1.0 - e1)
            //                      ratingUpdates(no2) -= k * e2
            //                    } else if (time1 > time2) {
            //                      ratingUpdates(no1) -= k * e1
            //                      ratingUpdates(no2) += k * (1.0 - e2)
            //                    }
            //                  case _ =>
            //                }
            //                ratings.indices.foreach {
            //                  case index =>
            //                    ratings(index) += ratingUpdates(index)
            //                }
            //            }
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
                val score = rating + (indexTime match {
                  case 0 => 400
                  case 1 => 350
                  case 2 => 250
                  case 3 => 200
                  case 4 => 200
                  case _ => 0
                }) + horse.prevDataList.take(10).map {
                  prevData =>
                    prevData.rank match {
                      case Some(1) if prevData.raceType == raceType =>
                        50
                      case Some(1) =>
                        20
                      case Some(n) if n <= 3 && prevData.raceType == raceType =>
                        30
                      case Some(n) if n <= 3 =>
                        10
                      case Some(_) if prevData.raceType == raceType  =>
                        -10
                      case Some(_) =>
                        -3
                      case _ =>
                        0
                    }
                }.sum
                (horse, score, ratingCount)
            }.sortBy(-_._2).zipWithIndex.map {
              case ((horse, rating, ratingCount), index) =>
                (horse, rating, ratingCount, index)
            }

            val ratingTop = newRatingInfo.head
            val ratingSecond = newRatingInfo(1)

            raceCount += 1

            val oddsTop = newRatingInfo.sortBy(_._1.odds).head
            val ratingOddsRatio = ratingTop._1.odds *
              (1 / (1.0 + Math.pow(10.0, (ratingSecond._2 - ratingTop._2) / 400.0)) - 0.5)
            if (oddsTop._3 > 100 && ratingTop._2 - oddsTop._2 < 50) {
              betCount += 1
              if (oddsTop._1.rank == 1) {
                betWinCount += 1
                oddsCount += oddsTop._1.odds
              }
            }

            val top = newRatingInfo.sortBy(_._1.rank).head
            val topOdds = newRatingInfoOdds.sortBy(_._1.rank).head
            val topTime = newRatingInfoTime.sortBy(_._1.rank).head
            val topScore = newRatingInfoScore.sortBy(_._1.rank).head

            if (topScore._3 > 0) {
              analyzeArray(topScore._4) += 1
            }
        }
    }


    val rtn = oddsCount / betCount
    val p = betWinCount.toDouble / betCount.toDouble
    val r = oddsCount / betWinCount - 1.0
    val kf = ((r + 1) * p - 1) / r
    val g = Math.pow(Math.pow(1 + r * kf, p) * Math.pow(1 - kf, 1 - p), betCount)
    println(analyzeArray.toSeq)
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