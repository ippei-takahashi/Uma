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

  private[this] val LEARNING_RATE = 0.0005

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
    CATEGORY_SHIBA_SHORT -> 38,
    CATEGORY_SHIBA_MIDDLE -> 38,
    CATEGORY_SHIBA_SEMI_LONG -> 38,
    CATEGORY_SHIBA_LONG -> 35,
    CATEGORY_SHIBA_VERY_LONG -> 35,
    CATEGORY_SHIBA_VERY_VERY_LONG -> 35,
    CATEGORY_DIRT_SHORT -> 28,
    CATEGORY_DIRT_MIDDLE -> 28,
    CATEGORY_DIRT_SEMI_LONG -> 25,
    CATEGORY_DIRT_LONG -> 25
  )

  private[this] val timeRaceMap = scala.collection.mutable.Map[Int, List[(Int, Double, Double)]](
    CATEGORY_SHIBA_SHORT -> List(
      (1011200, 58.43765077048586, 1.5),
      (2011200, 58.82585934913802, 1.5),
      (3011200, 58.013314207806914, 1.5),
      (4011200, 58.28461752843878, 1.5),
      (6111200, 58.19007029920478, 1.5),
      (7011200, 58.13398817656369, 1.5),
      (8011200, 57.91141409100686, 1.5),
      (9011200, 58.169821497275976, 1.5),
      (10011200, 57.91397306510776, 1.5)
    ),
    CATEGORY_SHIBA_MIDDLE -> List(
      (4011400, 62.27304611820301, 1.6),
      (5011400, 61.91216577047677, 1.6),
      (7011400, 62.54024163669171, 1.6),
      (8011400, 62.12013626646127, 1.6),
      (8111400, 62.218035238319324, 1.6),
      (9011400, 62.93307325752795, 1.6)
    ),
    CATEGORY_SHIBA_SEMI_LONG -> List(
      (1011500, 64.72426478930322, 1.65),
      (4111600, 65.8511273025396, 1.7),
      (5011600, 66.35200371986919, 1.7),
      (6111600, 66.89353041125254, 1.7),
      (7011600, 66.94788326677327, 1.7),
      (8011600, 66.24970984309415, 1.7),
      (8111600, 66.60205619821527, 1.7),
      (9111600, 66.4410511746216, 1.7)
    ),
    CATEGORY_SHIBA_LONG -> List(
      (1011800, 72.21748524409779, 1.8),
      (2011800, 72.57899017943711, 1.8),
      (3011800, 71.7202451945094, 1.8),
      (4111800, 70.31460630228092, 1.8),
      (5011800, 71.12862606082747, 1.8),
      (6011800, 71.56725878962095, 1.8),
      (8111800, 71.02580349039357, 1.8),
      (9111800, 71.15613243441345, 1.8),
      (10011800, 71.61421237987544, 1.8)
    ),
    CATEGORY_SHIBA_VERY_LONG -> List(
      (1012000, 76.40210208476523, 1.9),
      (2012000, 77.28910435583585, 1.9),
      (3012000, 75.91151289882677, 1.9),
      (4012000, 75.68782869363847, 1.9),
      (4112000, 75.41429035054125, 1.9),
      (5012000, 75.79272699386921, 1.9),
      (6012000, 76.22718989864491, 1.9),
      (7012000, 76.53378320867436, 1.9),
      (8012000, 75.61828048286283, 1.9),
      (9012000, 76.45856294780879, 1.9),
      (10012000, 75.97913045479397, 1.9)
    ),
    CATEGORY_SHIBA_VERY_VERY_LONG -> List(
      (4012200, 80.864377281439, 2.0),
      (6112200, 80.83069335325474, 2.0),
      (7012200, 81.50733068218753, 2.0),
      (8112200, 80.23021252021998, 2.0),
      (9012200, 81.16361044615533, 2.0),
      (4012400, 85.20028225999893, 2.1),
      (5012400, 85.29692869021689, 2.1),
      (8112400, 84.70430263003067, 2.1),
      (9112400, 85.35710818924866, 2.1),
      (6012500, 88.0745388952903, 2.15),
      (3012600, 90.19696597347051, 2.2),
      (10012600, 89.8466247274885, 2.2)
    ),
    CATEGORY_DIRT_SHORT -> List(
      (1001000, 56.75810966097955, 1.7),
      (2001000, 56.365883363571214, 1.7),
      (10001000, 55.643637709971316, 1.7),
      (3001150, 61.223327838498854, 1.85),
      (4001200, 62.21525734092809, 1.9),
      (6001200, 62.38014142230501, 1.9),
      (7001200, 62.33030660730873, 1.9),
      (8001200, 62.13960304484699, 1.9),
      (9001200, 61.651247615382914, 1.9)
    ),
    CATEGORY_DIRT_MIDDLE -> List(
      (5001300, 62.672878193131375, 1.95),
      (5001400, 65.17129375245729, 2.0),
      (7001400, 66.2558753675327, 2.0),
      (8001400, 66.23770091957972, 2.0),
      (9001400, 66.15667314758339, 2.0)
    ),
    CATEGORY_DIRT_SEMI_LONG -> List(
      (5001600, 70.52883479730231, 2.15),
      (1001700, 74.0027851944908, 2.2),
      (2001700, 73.94438941669941, 2.2),
      (3001700, 74.05633331336809, 2.2),
      (7001700, 74.39370796032252, 2.2),
      (10001700, 73.43816738697241, 2.2)
    ),
    CATEGORY_DIRT_LONG -> List(
      (4001800, 76.20973049007549, 2.25),
      (6001800, 76.55368792836467, 2.25),
      (7001800, 76.28125182922382, 2.25),
      (8001800, 75.91813389060748, 2.25),
      (9001800, 75.72246437810817, 2.25),
      (8001900, 78.70925896660967, 2.3),
      (9002000, 80.84778959432089, 2.35),
      (5002100, 82.59575041637993, 2.4)
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
        val outFile = new File("result.csv")
        val pw = new PrintWriter(outFile)

        try {
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
                      mean(list.take(list.length / 2 + 1))
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
                case _ =>
              }

              val cond = (x: (PredictData, Double)) =>
                x._2 >= STD_THREASHOLD &&
                  Math.pow((x._2 + 10) / 100, 1.3) * Math.pow(Math.min(x._1.odds, 100), 0.2) > Math.min(1.0 / Math.pow(shareSum, 0.5), 0.15)

              if (shareSum > SHARE_THREASHOLDS(raceCategory) && res.count(_._2.isNaN) < 3) {
                betRaceCount += 1
                if (stdRes.exists(x => cond(x) && x._1.rank == 1)) {
                  winRaceCount += 1
                }
                pw.println("%010d".format(raceId.toLong))
                stdRes.filter(cond).foreach {
                  x =>
                    var bonus = 0
                    if (x._1.age >= 72) {
                      bonus += 50
                    }
                    pw.println(true, x)
                    val betRate = 1.0 / (res.count(_._2.isNaN) + 2) * (100 + bonus)
                    betCount += betRate
                    if (x._1.rank == 1) {
                      winCount += betRate
                      oddsCount += x._1.odds * betRate
                    }
                }
                stdRes.filterNot(cond).foreach {
                  x =>
                    pw.println(false, x)
                }
                pw.println
              }
            case _ =>
          }
        } finally {
          pw.close()
        }

        val (xarr, yarr) = stdWinMap.toArray.sortBy(_._1).map {
          case (std, list) =>
            val win = list.count(x => x)
            val lose = list.count(x => !x)
            (std.toDouble, win.toDouble / (win + lose))
        }.unzip
        println(betCount, betRaceCount, winRaceCount / betRaceCount, winCount / betCount, oddsCount / winCount, oddsCount / betCount)

        val outFileStudy = new File("studyResult.csv")
        val pwStudy = new PrintWriter(outFileStudy)

        try {
          timeRaceMap.toSeq.sortBy(_._1).foreach {
            case (_, seq) =>
              pwStudy.println(seq.sortBy(_._1 % 10000).mkString("List(", ",\n", ")"))
          }
        } finally {
          pwStudy.close()
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