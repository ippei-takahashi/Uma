import java.io._

import breeze.linalg._
import breeze.stats._

object Main {
  type Gene = DenseVector[Double]

  case class Data(raceDate: Int, raceType: Long, age: Int, rank: Int, odds: Double, stdTime: Double, raceId: Long,
                  paceRank: Int, isGoodBaba: Boolean, horseNo: Int)

  case class PredictData(horseId: Int, raceDate: Int, raceType: Long, age: Double, rank: Int, odds: Double, oddsFuku: Double,
                         stdTime: Double, paceRank: Int, isGoodBaba: Boolean, horseNo: Int, prevDataList: Seq[Data])

  val CATEGORY_SHIBA_SHORT = 0

  val CATEGORY_SHIBA_MIDDLE = 1

  val CATEGORY_SHIBA_SEMI_LONG = 2

  val CATEGORY_SHIBA_LONG = 3

  val CATEGORY_SHIBA_VERY_LONG = 4

  val CATEGORY_SHIBA_VERY_VERY_LONG = 9

  val CATEGORY_DIRT_SHORT = 5

  val CATEGORY_DIRT_MIDDLE = 6

  val CATEGORY_DIRT_SEMI_LONG = 7

  val CATEGORY_DIRT_LONG = 8

  private[this] val raceTimeMap = scala.collection.mutable.Map[Long, List[Double]]()

  private[this] var maxTimeRaceList: List[Long] = Nil

  private[this] val timeRaceMap = Map[Int, List[(Int, Double, Double)]](
    CATEGORY_SHIBA_SHORT -> List(
      (1011200, 58.45, 1.7),
      (2011200, 58.75, 1.7),
      (3011200, 58.1, 1.7),
      (4011200, 58.2, 1.7),
      (6111200, 58.0, 1.7),
      (7011200, 58.25, 1.7),
      (8011200, 57.6, 1.7),
      (9011200, 58.1, 1.7),
      (10011200, 57.85, 1.7)
    ),
    CATEGORY_SHIBA_MIDDLE -> List(
      (4011400, 62.4, 1.8),
      (5011400, 61.7, 1.8),
      (7011400, 62.9, 1.8),
      (8011400, 62.1, 1.8),
      (8111400, 61.9, 1.8),
      (9011400, 62.8, 1.8)
    ),
    CATEGORY_SHIBA_SEMI_LONG -> List(
      (1011500, 65.0, 1.85),

      (4111600, 65.7, 1.9),
      (5011600, 66.3, 1.9),
      (6111600, 66.9, 1.9),
      (7011600, 67.4, 1.9),
      (8011600, 66.6, 1.9),
      (8111600, 66.45, 1.9),
      (9111600, 66.55, 1.9)
    ),
    CATEGORY_SHIBA_LONG -> List(
      (1011800, 72.2, 2.0),
      (2011800, 72.6, 2.0),
      (3011800, 71.8, 2.0),
      (4111800, 70.4, 2.0),
      (5011800, 71.0, 2.0),
      (6011800, 72.0, 2.0),
      (8111800, 71.15, 2.0),
      (9111800, 71.2, 2.0),
      (10011800, 71.5, 2.0)
    ),
    CATEGORY_SHIBA_VERY_LONG -> List(
      (1012000, 76.65, 2.1),
      (2012000, 77.2, 2.1),
      (3012000, 76.1, 2.1),
      (4012000, 76.15, 2.1),
      (4112000, 75.05, 2.1),
      (5012000, 75.65, 2.1),
      (6012000, 76.45, 2.1),
      (7012000, 76.85, 2.1),
      (8012000, 75.65, 2.1),
      (9012000, 76.65, 2.1),
      (10012000, 75.95, 2.1)
    ),
    CATEGORY_SHIBA_VERY_VERY_LONG -> List(
      (4012200, 80.65, 2.2),
      (6112200, 81.0, 2.2),
      (7012200, 81.75, 2.2),
      (8112200, 80.25, 2.2),
      (9012200, 81.25, 2.2),

      (4012400, 85.25, 2.3),
      (5012400, 85.0, 2.3),
      (8112400, 84.65, 2.3),
      (9112400, 85.5, 2.3),

      (6012500, 88.0, 2.35),

      (3012600, 90.1, 2.4),
      (10012600, 89.95, 2.4)
    ),
    CATEGORY_DIRT_SHORT -> List(
      (1001000, 56.15, 1.7),
      (2001000, 56.1, 1.7),
      (10001000, 55.65, 1.7),

      (3001150, 60.9, 1.85),

      (4001200, 61.9, 1.9),
      (6001200, 62.2, 1.9),
      (7001200, 62.1, 1.9),
      (8001200, 61.8, 1.9),
      (9001200, 61.7, 1.9)
    ),
    CATEGORY_DIRT_MIDDLE -> List(
      (5001300, 63.3, 1.95),

      (5001400, 65.8, 2.0),
      (7001400, 66.5, 2.0),
      (8001400, 66.4, 2.0),
      (9001400, 66.4, 2.0)
    ),
    CATEGORY_DIRT_SEMI_LONG -> List(
      (5001600, 70.5, 2.15),

      (1001700, 73.8, 2.2),
      (2001700, 73.9, 2.2),
      (3001700, 74.1, 2.2),
      (7001700, 74.1, 2.2),
      (10001700, 73.5, 2.2)
    ),
    CATEGORY_DIRT_LONG -> List(
      (4001800, 76.1, 2.25),
      (6001800, 76.8, 2.25),
      (7001800, 76.2, 2.25),
      (8001800, 75.8, 2.25),
      (9001800, 75.9, 2.25),

      (8001900, 78.3, 2.3),

      (9002000, 80.5, 2.35),

      (5002100, 82.4, 2.4)
    )
  )

  val timeRaceFlattenMap = timeRaceMap.values.toList.flatten.map {
    case (key, m, s) =>
      key.toLong ->(m, s)
  }.toMap

  def main(args: Array[String]) {

    val dataCSV = new File("past.csv")

    val data: DenseMatrix[Double] = csvread(dataCSV)
    val size = data.rows

    val array = Array.ofDim[DenseVector[Double]](size)
    for (i <- 0 until size) {
      array(i) = data(i, ::).t
    }

    val raceMap = array.groupBy(_ (0)).map {
      case (raceId, arr) =>
        val horses = arr.map {
          d =>
            val x = d(3 until data.cols - 1)
            val raceType = makeRaceType(x, raceId.toLong)
            val paceRank = d(d.length - 8).toInt match {
              case 0 => d(d.length - 9).toInt match {
                case 0 => d(d.length - 10).toInt
                case n => n
              }
              case n => n
            }
            val rank = d(d.length - 1).toInt
            val oddsFuku = rank match {
              case 1 =>
                d(d.length - 4)
              case 2 =>
                d(d.length - 3)
              case 3 =>
                d(d.length - 2)
              case _ =>
                0
            }
            val time = d(d.length - 6)
            val time3f = d(d.length - 7)
            val dist = raceType % 10000
            val positionBonus = if (paceRank <= 6)
              0.1 * (paceRank - 6) * Math.pow(dist / 1000, 0.3)
            else
              0.05 * (paceRank - 6) * Math.pow(dist / 1000, 0.3)
            val horseNo = x(1).toInt
            val stdTime = time / 3 + time3f + positionBonus
            new PredictData(horseId = d(1).toInt, raceDate = d(2).toInt, age = d(3).toInt, rank = rank,
              odds = d(d.length - 5), oddsFuku = oddsFuku, stdTime = stdTime, raceType = raceType,
              isGoodBaba = x(11) + x(12) == 1.0 && x(7) + x(8) == 1.0, horseNo = horseNo,
              paceRank = paceRank, prevDataList = Nil)
        }

        raceId -> horses
    }
    val raceSeq_ = raceMap.toSeq.sortBy(_._2.head.raceDate)

    val horseMap = array.groupBy(_ (1)).map {
      case (horseId, arr) =>
        horseId -> arr.map { d =>
          val raceId = d(0)
          val x = d(3 until data.cols - 1)
          val raceType = makeRaceType(x, raceId.toLong)
          val paceRank = d(d.length - 8).toInt match {
            case 0 => d(d.length - 9).toInt match {
              case 0 => d(d.length - 10).toInt
              case n => n
            }
            case n => n
          }
          val rank = d(d.length - 1).toInt
          val time = d(d.length - 6)
          val time3f = d(d.length - 7)
          val dist = raceType % 10000
          val positionBonus = if (paceRank <= 6)
            0.1 * (paceRank - 6) * Math.pow(dist / 1000, 0.3)
          else
            0.05 * (paceRank - 6) * Math.pow(dist / 1000, 0.3)
          val horseNo = x(1).toInt
          val insideBonus = 0 * (horseNo - 6)
          val stdTime = time / 3 + time3f + positionBonus + insideBonus
          new Data(raceDate = d(2).toInt, age = d(3).toInt, rank = rank,
            odds = d(d.length - 5).toInt, stdTime = stdTime, raceId = raceId.toLong,
            raceType = raceType, paceRank = paceRank, isGoodBaba = x(11) + x(12) == 1.0 && x(7) + x(8) == 1.0,
            horseNo = horseNo)
        }.toList
    }

    val raceSeq = raceSeq_.map {
      case (raceId, horses) =>
        raceId -> horses.map {
          pred =>
            pred.copy(prevDataList = horseMap(pred.horseId).filter(x => x.raceDate < pred.raceDate && x.isGoodBaba))
        }
    }

    val outFile = new File("result.csv")
    val pw = new PrintWriter(outFile)

    var betRaceCount = 0.0
    var winRaceCount = 0.0
    var betCount = 0.0
    var winCount = 0.0
    var oddsCount = 0.0

    horseMap.foreach {
      case (_, races) =>
        races.filter(x => x.isGoodBaba && timeRaceFlattenMap.get(x.raceType).isDefined).foreach {
          data =>
            val raceCategory = getRaceCategory(data.raceType)
            val timeRace = timeRaceMap(raceCategory)
            if (races.count(x => timeRace.exists(_._1 == x.raceType)) > 3) {
              val (m, s) = timeRaceFlattenMap(data.raceType)
              val stdTime = (m - data.stdTime) / s * 10 + 50
              raceTimeMap.put(data.raceType, stdTime :: raceTimeMap.getOrElse(data.raceType, Nil))
            }
        }
        (CATEGORY_SHIBA_VERY_LONG to CATEGORY_SHIBA_VERY_LONG).foreach {
          raceCategory =>
            val timeRace = timeRaceMap(raceCategory)
            val infos = for {
              data <- races
              time <- timeRace.find(_._1 == data.raceType).toList
            } yield {
              val m = time._2
              val s = time._3

              (data.raceType, (m - data.stdTime) / s * 10 + 50)
            }
            if (infos.length > 3) {
              maxTimeRaceList = infos.maxBy(_._2)._1 :: maxTimeRaceList
            }
        }

    }
    Seq(
      "maxTimeRace.csv" ->(maxTimeRaceList.groupBy(x => x).map {
        case (key, value) =>
          key.asInstanceOf[Long] -> value.map(_.toDouble)
      }, raceTimeMap)
    ).foreach {
      case (fileName, (map1, map2)) =>
        val mat = DenseMatrix(map1.toArray.map {
          case (key, list) =>
            val m = mean(map2(key))
            val m2 = mean(map2(key).sortBy(-_).take(map2(key).length / 100))
            (key.toDouble, list.length.toDouble / map2(key).length.toDouble, m, m2)
        }.sortBy(_._2): _*)
        csvwrite(new File(fileName), mat)
    }

    try {
      raceSeq.foreach {
        case (raceId, horses) if horses.head.isGoodBaba =>
          val raceCategory = getRaceCategory(horses.head.raceType)
          val secondaryRaceCategory = getSecondaryRaceCategory(raceCategory)
          val raceDate = horses.head.raceDate

          val timeRace = timeRaceMap(raceCategory)
          val timeRaceSecondary = secondaryRaceCategory.map(timeRaceMap)

          val timeList = horses.map {
            horse =>
              for {
                data <- horse.prevDataList if raceDate - data.raceDate < 10000
                time <- timeRace.find(_._1 == data.raceType).toList match {
                  case Nil =>
                    timeRaceSecondary.toList.flatMap {
                      secondary =>
                        secondary.find(_._1 == data.raceType).toList
                    }
                  case list =>
                    list
                }
              } yield {
                val m = time._2
                val s = time._3

                (m - data.stdTime) / s * 10 + 50
              }
          }

          val res = horses.zip(timeList).map {
            case (horse, prevStdList) =>
              val time = prevStdList.sortBy(-_) match {
                case Nil => Double.NaN
                case list =>
                  mean(list.take(3))
              }
              (horse.copy(prevDataList = Nil), time)
          }.sortBy(_._1.odds).toSeq

          val (timeMean, timeStd) = res.toList.map(_._2).filterNot(_.isNaN) match {
            case Nil => (Double.NaN, Double.NaN)
            case list => (mean(list), stddev(list))
          }
          val stdRes = res.map {
            case (horse, time) =>
              (horse, (time - timeMean) / timeStd * 10 + 50)
          }

          val removeSeq = if (timeMean.isNaN)
            Nil
          else
            stdRes.filter {
              x => x._2 < 45
            }

          val shareSum = removeSeq.map {
            x =>
              78.8 / (x._1.odds - 1)
          }.sum

          if (shareSum > 55 && res.count(_._2.isNaN) < 3) {
            pw.println("%010d".format(raceId.toLong))
            betRaceCount += 1
            if (stdRes.exists(x => x._2 >= 45 && x._1.rank == 1)) {
              winRaceCount += 1
            }
            stdRes.filter {
              x =>
                x._2 >= 45
            }.foreach {
              x =>
                pw.println(x, true)
                val betRate = x._2 / 50
                betCount += betRate
                if (x._1.rank == 1) {
                  winCount += betRate
                  oddsCount += x._1.odds * betRate
                }
            }
            stdRes.filterNot {
              x =>
                x._2 >= 45
            }.foreach {
              x =>
                pw.println(x, false)
            }
            pw.println
          }
        case _ =>
      }
    } catch {
      case e: Exception =>
    } finally {
      pw.close
    }

    println(betCount, betRaceCount, winRaceCount / betRaceCount, winCount / betCount, oddsCount / winCount, oddsCount / betCount)
  }

  def subListBeforeRaceId(raceId: Long, list: List[Data]): List[Data] = list match {
    case x :: xs if x.raceId == raceId =>
      x :: xs
    case _ :: xs =>
      subListBeforeRaceId(raceId, xs)
    case _ =>
      Nil
  }

  def getRaceCategory(raceType: Long) = ((raceType / 10000) % 10, raceType % 10000) match {
    case (1, dist) if dist <= 1200 =>
      CATEGORY_SHIBA_SHORT
    case (1, dist) if dist <= 1400 =>
      CATEGORY_SHIBA_MIDDLE
    case (1, dist) if dist <= 1600 =>
      CATEGORY_SHIBA_SEMI_LONG
    case (1, dist) if dist <= 1800 =>
      CATEGORY_SHIBA_LONG
    case (1, dist) if dist <= 2000 =>
      CATEGORY_SHIBA_VERY_LONG
    case (1, dist) =>
      CATEGORY_SHIBA_VERY_VERY_LONG
    case (0, dist) if dist <= 1200 =>
      CATEGORY_DIRT_SHORT
    case (0, dist) if dist <= 1400 =>
      CATEGORY_DIRT_MIDDLE
    case (0, dist) if dist <= 1700 =>
      CATEGORY_DIRT_SEMI_LONG
    case (0, dist) =>
      CATEGORY_DIRT_LONG
  }

  def getSecondaryRaceCategory(raceCategory: Int) = raceCategory match {
    case CATEGORY_SHIBA_SHORT =>
      None
    case CATEGORY_SHIBA_MIDDLE =>
      None
    case CATEGORY_SHIBA_SEMI_LONG =>
      Some(CATEGORY_SHIBA_MIDDLE)
    case CATEGORY_SHIBA_LONG =>
      Some(CATEGORY_SHIBA_VERY_LONG)
    case CATEGORY_SHIBA_VERY_LONG =>
      Some(CATEGORY_SHIBA_LONG)
    case CATEGORY_SHIBA_VERY_VERY_LONG =>
      Some(CATEGORY_SHIBA_VERY_LONG)
    case CATEGORY_DIRT_SHORT =>
      Some(CATEGORY_DIRT_MIDDLE)
    case CATEGORY_DIRT_MIDDLE =>
      None
    case CATEGORY_DIRT_SEMI_LONG =>
      None
    case CATEGORY_DIRT_LONG =>
      None
  }

  def makeRaceType(vector: DenseVector[Double], raceId: Long): Long = {
    val babaCode = (raceId / 1000000) % 100
    babaCode * 1000000 + vector(5).toLong * 100000 + vector(2).toLong * 10000 + vector(6).toLong
  }
}