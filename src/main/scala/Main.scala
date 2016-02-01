import java.io._

import breeze.linalg._
import breeze.stats._
import breeze.plot._

object Main {
  type Gene = DenseVector[Double]

  case class Data(raceDate: Int, raceType: Long, age: Int, rank: Int, odds: Double, stdTime: Double, raceId: Long,
                  paceRank: Int, isGoodBaba: Boolean, horseNo: Int)

  case class PredictData(horseId: Int, raceDate: Int, raceType: Long, age: Double, rank: Int, odds: Double, oddsFuku: Double,
                         stdTime: Double, paceRank: Int, isGoodBaba: Boolean, horseNo: Int, prevDataList: Seq[Data])

  private[this] val LEARNING_RATE = 0.003

  private[this] val NUM_OF_LOOPS = 1000000

  private[this] val CATEGORY_SHIBA_SHORT = 0

  private[this] val CATEGORY_SHIBA_MIDDLE = 1

  private[this] val CATEGORY_SHIBA_SEMI_LONG = 2

  private[this] val CATEGORY_SHIBA_LONG = 3

  private[this] val CATEGORY_SHIBA_VERY_LONG = 4

  private[this] val CATEGORY_SHIBA_VERY_VERY_LONG = 5

  private[this] val CATEGORY_DIRT_SHORT = 6

  private[this] val CATEGORY_DIRT_MIDDLE = 7

  private[this] val CATEGORY_DIRT_SEMI_LONG = 8

  private[this] val CATEGORY_DIRT_LONG = 9

  private[this] val STD_THREASHOLD = -2

  private[this] val SHARE_THREASHOLDS = Map[Int, Double](
    CATEGORY_SHIBA_SHORT -> 68,
    CATEGORY_SHIBA_MIDDLE -> 68,
    CATEGORY_SHIBA_SEMI_LONG -> 68,
    CATEGORY_SHIBA_LONG -> 65,
    CATEGORY_SHIBA_VERY_LONG -> 65,
    CATEGORY_SHIBA_VERY_VERY_LONG -> 65,
    CATEGORY_DIRT_SHORT -> 58,
    CATEGORY_DIRT_MIDDLE -> 58,
    CATEGORY_DIRT_SEMI_LONG -> 55,
    CATEGORY_DIRT_LONG -> 55
  )

  private[this] val timeRaceMap = scala.collection.mutable.Map[Int, List[(Int, Double, Double)]](
    CATEGORY_SHIBA_SHORT -> List(
      (1011200, 58.45, 1.5),
      (2011200, 58.75, 1.5),
      (3011200, 58.1, 1.5),
      (4011200, 58.2, 1.5),
      (6111200, 58.0, 1.5),
      (7011200, 58.25, 1.5),
      (8011200, 57.6, 1.5),
      (9011200, 58.1, 1.5),
      (10011200, 57.85, 1.5)
    ),
    CATEGORY_SHIBA_MIDDLE -> List(
      (4011400, 62.4, 1.6),
      (5011400, 61.7, 1.6),
      (7011400, 62.9, 1.6),
      (8011400, 62.1, 1.6),
      (8111400, 61.9, 1.6),
      (9011400, 62.8, 1.6)
    ),
    CATEGORY_SHIBA_SEMI_LONG -> List(
      (1011500, 65.0, 1.65),

      (4111600, 65.7, 1.7),
      (5011600, 66.3, 1.7),
      (6111600, 66.9, 1.7),
      (7011600, 67.4, 1.7),
      (8011600, 66.6, 1.7),
      (8111600, 66.45, 1.7),
      (9111600, 66.55, 1.7)
    ),
    CATEGORY_SHIBA_LONG -> List(
      (1011800, 72.2, 1.8),
      (2011800, 72.6, 1.8),
      (3011800, 71.8, 1.8),
      (4111800, 70.4, 1.8),
      (5011800, 71.0, 1.8),
      (6011800, 71.8, 1.8),
      (8111800, 71.15, 1.8),
      (9111800, 71.2, 1.8),
      (10011800, 71.5, 1.8)
    ),
    CATEGORY_SHIBA_VERY_LONG -> List(
      (1012000, 76.65, 1.9),
      (2012000, 77.2, 1.9),
      (3012000, 76.1, 1.9),
      (4012000, 76.15, 1.9),
      (4112000, 75.05, 1.9),
      (5012000, 75.65, 1.9),
      (6012000, 76.45, 1.9),
      (7012000, 76.85, 1.9),
      (8012000, 75.65, 1.9),
      (9012000, 76.65, 1.9),
      (10012000, 75.95, 1.9)
    ),
    CATEGORY_SHIBA_VERY_VERY_LONG -> List(
      (4012200, 80.65, 2.0),
      (6112200, 81.0, 2.0),
      (7012200, 81.75, 2.0),
      (8112200, 80.25, 2.0),
      (9012200, 81.25, 2.0),

      (4012400, 85.25, 2.1),
      (5012400, 85.0, 2.1),
      (8112400, 84.65, 2.1),
      (9112400, 85.5, 2.1),

      (6012500, 88.0, 2.15),

      (3012600, 90.1, 2.2),
      (10012600, 89.95, 2.2)
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

  private[this] val timeErrorRaceMap = scala.collection.mutable.Map[Int, List[(Int, Double, Int)]](
    CATEGORY_SHIBA_SHORT -> List(
      (1011200, 0.0, 0),
      (2011200, 0.0, 0),
      (3011200, 0.0, 0),
      (4011200, 0.0, 0),
      (6111200, 0.0, 0),
      (7011200, 0.0, 0),
      (8011200, 0.0, 0),
      (9011200, 0.0, 0),
      (10011200, 0.0, 0)
    ),
    CATEGORY_SHIBA_MIDDLE -> List(
      (4011400, 0.0, 0),
      (5011400, 0.0, 0),
      (7011400, 0.0, 0),
      (8011400, 0.0, 0),
      (8111400, 0.0, 0),
      (9011400, 0.0, 0)
    ),
    CATEGORY_SHIBA_SEMI_LONG -> List(
      (1011500, 0.0, 0),

      (4111600, 0.0, 0),
      (5011600, 0.0, 0),
      (6111600, 0.0, 0),
      (7011600, 0.0, 0),
      (8011600, 0.0, 0),
      (8111600, 0.0, 0),
      (9111600, 0.0, 0)
    ),
    CATEGORY_SHIBA_LONG -> List(
      (1011800, 0.0, 0),
      (2011800, 0.0, 0),
      (3011800, 0.0, 0),
      (4111800, 0.0, 0),
      (5011800, 0.0, 0),
      (6011800, 0.0, 0),
      (8111800, 0.0, 0),
      (9111800, 0.0, 0),
      (10011800, 0.0, 0)
    ),
    CATEGORY_SHIBA_VERY_LONG -> List(
      (1012000, 0.0, 0),
      (2012000, 0.0, 0),
      (3012000, 0.0, 0),
      (4012000, 0.0, 0),
      (4112000, 0.0, 0),
      (5012000, 0.0, 0),
      (6012000, 0.0, 0),
      (7012000, 0.0, 0),
      (8012000, 0.0, 0),
      (9012000, 0.0, 0),
      (10012000, 0.0, 0)
    ),
    CATEGORY_SHIBA_VERY_VERY_LONG -> List(
      (4012200, 0.0, 0),
      (6112200, 0.0, 0),
      (7012200, 0.0, 0),
      (8112200, 0.0, 0),
      (9012200, 0.0, 0),

      (4012400, 0.0, 0),
      (5012400, 0.0, 0),
      (8112400, 0.0, 0),
      (9112400, 0.0, 0),

      (6012500, 0.0, 0),

      (3012600, 0.0, 0),
      (10012600, 0.0, 0)
    ),
    CATEGORY_DIRT_SHORT -> List(
      (1001000, 0.0, 0),
      (2001000, 0.0, 0),
      (10001000, 0.0, 0),

      (3001150, 0.0, 0),

      (4001200, 0.0, 0),
      (6001200, 0.0, 0),
      (7001200, 0.0, 0),
      (8001200, 0.0, 0),
      (9001200, 0.0, 0)
    ),
    CATEGORY_DIRT_MIDDLE -> List(
      (5001300, 0.0, 0),

      (5001400, 0.0, 0),
      (7001400, 0.0, 0),
      (8001400, 0.0, 0),
      (9001400, 0.0, 0)
    ),
    CATEGORY_DIRT_SEMI_LONG -> List(
      (5001600, 0.0, 0),

      (1001700, 0.0, 0),
      (2001700, 0.0, 0),
      (3001700, 0.0, 0),
      (7001700, 0.0, 0),
      (10001700, 0.0, 0)
    ),
    CATEGORY_DIRT_LONG -> List(
      (4001800, 0.0, 0),
      (6001800, 0.0, 0),
      (7001800, 0.0, 0),
      (8001800, 0.0, 0),
      (9001800, 0.0, 0),

      (8001900, 0.0, 0),

      (9002000, 0.0, 0),

      (5002100, 0.0, 0)
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
    }.filter(_._2.head.isGoodBaba).groupBy(_._2.head.raceType).toSeq.flatMap {
      case (_, seq) =>
        seq.sortBy(arr => mean(arr._2.toList.map(_.stdTime))).take((seq.length * 0.8).toInt)
    }

    for (i <- 0 until NUM_OF_LOOPS) {
      raceSeq.foreach {
        case (raceId, horses) =>
          val raceCategory = getRaceCategory(horses.head.raceType)
          val secondaryRaceCategory = getSecondaryRaceCategory(raceCategory)
          val raceDate = horses.head.raceDate

          val timeRace = timeRaceMap(raceCategory)
          val timeRaceSecondary = secondaryRaceCategory.map(timeRaceMap)

          timeRace.find(_._1 == horses.head.raceType) match {
            case Some((_, m_, s_)) =>
              horses.foreach {
                horse =>
                  val stdScore_ = (m_ - horse.stdTime) / s_ * 10 + 50

                  for {
                    data <- horse.prevDataList if raceDate - data.raceDate < 10000
                    time <- timeRace.find(_._1 == data.raceType).toList match {
                      case Nil =>
                        timeRaceSecondary.flatMap {
                          secondary =>
                            secondary.find(_._1 == data.raceType).toList
                        }
                      case list =>
                        list
                    }
                  } yield {
                    val m = time._2
                    val s = time._3

                    val stdScore = (m - data.stdTime) / s * 10 + 50

                    timeErrorRaceMap.foreach {
                      case (category, seq) =>
                        timeErrorRaceMap.put(category, seq.map {
                          case (raceType, error, errorCount) if raceType == data.raceType =>
                            (raceType, error + (stdScore - stdScore_) * LEARNING_RATE, errorCount + 1)
                          case (raceType, error, errorCount) =>
                            (raceType, error, errorCount)
                        })
                    }
                  }
              }
            case _ =>
          }
        case _ =>
      }

      val (_, errorSum, errorCountSum) = timeErrorRaceMap.toSeq.flatMap(_._2).reduce[(Int, Double, Int)] {
        case ((_, xs, ys), (_, x, y)) =>
          (0, xs + x, ys + y)
      }
      val errorMeanSum = errorSum / errorCountSum
      timeErrorRaceMap.foreach {
        case (category, seq) =>
          val timeRace = timeRaceMap(category)
          timeRaceMap.put(
            category, seq.map {
              case (raceType, error, errorCount) =>
                val (_, m, s) = timeRace.find(_._1 == raceType).get
                val errorMean = if (errorCount == 0) 0 else error / errorCount - errorMeanSum
                (raceType, m + errorMean, s)
            })
          timeErrorRaceMap.put(category, seq.map {
            case (raceType, error, errorCount) =>
              (raceType, error * 0.9, errorCount)
          })
      }

      if (i % 20 == 0) {
        var betRaceCount = 0.0
        var winRaceCount = 0.0
        var betCount = 0.0
        var winCount = 0.0
        var oddsCount = 0.0
        val stdWinMap = scala.collection.mutable.Map[Int, List[Boolean]]()

        raceSeq.foreach {
          case (raceId, horses) =>
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
                      timeRaceSecondary.flatMap {
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

            val (timeMean, timeMid) = res.toList.map(_._2).filterNot(_.isNaN) match {
              case Nil => (Double.NaN, Double.NaN)
              case list => (mean(list), list.sorted.apply(list.length / 2))
            }
            val stdRes = res.map {
              case (horse, time) =>
                (horse, time - timeMean)
            }

            val removeSeq = if (timeMean.isNaN)
              Nil
            else
              stdRes.filter {
                x => x._2 < STD_THREASHOLD
              }

            val shareSum = removeSeq.map {
              x =>
                78.8 / (x._1.odds - 1)
            }.sum

            stdRes.foreach {
              case (pred, std) if !std.isNaN =>
                val xs = stdWinMap.getOrElse(std.toInt, Nil)
                stdWinMap.put(Math.max(-20, std.toInt), (pred.rank == 1) :: xs)
              case _=>
            }

            if (shareSum > SHARE_THREASHOLDS(raceCategory) && res.count(_._2.isNaN) < 3) {
              betRaceCount += 1
              if (stdRes.exists(x => x._2 >= STD_THREASHOLD && x._1.rank == 1)) {
                winRaceCount += 1
              }
              stdRes.filter {
                x =>
                  x._2 >= STD_THREASHOLD
              }.foreach {
                x =>
                  val betRate = x._2 / 50
                  betCount += betRate
                  if (x._1.rank == 1) {
                    winCount += betRate
                    oddsCount += x._1.odds * betRate
                  }
              }
            }
          case _ =>
        }
        val (xarr, yarr) = stdWinMap.toArray.sortBy(_._1).map {
          case (std, list) =>
            val win = list.count(x => x)
            val lose = list.count(x => !x)
            println(std.toDouble, win, lose, win.toDouble / (win + lose))
            (std.toDouble, win.toDouble / (win + lose))
        }.unzip
//        val f = Figure()
//        val p = f.subplot(0)
//        p += plot(DenseVector(xarr),DenseVector(yarr), '.')
//        p.xlabel = "x axis"
//        p.ylabel = "y axis"
//        f.saveas("lines.png")
        println(betCount, betRaceCount, winRaceCount / betRaceCount, winCount / betCount, oddsCount / winCount, oddsCount / betCount)

        val outFile = new File("studyResult.csv")
        val pw = new PrintWriter(outFile)
        try {
          timeRaceMap.toSeq.sortBy(_._1).foreach {
            case (_, seq) =>
              pw.println(seq.sortBy(_._1 % 10000).mkString("List(", ",\n", ")"))
          }
        } finally {
          pw.close()
        }
      }
    }
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
      Nil
    case CATEGORY_SHIBA_MIDDLE =>
      List(CATEGORY_SHIBA_SEMI_LONG)
    case CATEGORY_SHIBA_SEMI_LONG =>
      List(CATEGORY_SHIBA_MIDDLE)
    case CATEGORY_SHIBA_LONG =>
      List(CATEGORY_SHIBA_VERY_LONG, CATEGORY_SHIBA_VERY_VERY_LONG)
    case CATEGORY_SHIBA_VERY_LONG =>
      List(CATEGORY_SHIBA_LONG, CATEGORY_SHIBA_LONG)
    case CATEGORY_SHIBA_VERY_VERY_LONG =>
      List(CATEGORY_SHIBA_LONG, CATEGORY_SHIBA_VERY_LONG)
    case CATEGORY_DIRT_SHORT =>
      List(CATEGORY_DIRT_MIDDLE)
    case CATEGORY_DIRT_MIDDLE =>
      List(CATEGORY_DIRT_SHORT)
    case CATEGORY_DIRT_SEMI_LONG =>
      Nil
    case CATEGORY_DIRT_LONG =>
      Nil
  }

  def makeRaceType(vector: DenseVector[Double], raceId: Long): Long = {
    val babaCode = (raceId / 1000000) % 100
    babaCode * 1000000 + vector(5).toLong * 100000 + vector(2).toLong * 10000 + vector(6).toLong
  }
}