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

  private[this] val STD_THRESHOLD = -2

  private[this] val SHARE_THRESHOLDS = Map[Int, Double](
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
      (1011200, 58.457085287726166, 1.5),
      (2011200, 58.81954248755394, 1.5),
      (3011200, 58.05216471775275, 1.5),
      (4011200, 58.2828651053038, 1.5),
      (6111200, 58.165752044467666, 1.5),
      (7011200, 58.18229988343318, 1.5),
      (8011200, 57.86067908256523, 1.5),
      (9011200, 58.17432261847639, 1.5),
      (10011200, 57.91864555255517, 1.5)
    ),
    CATEGORY_SHIBA_MIDDLE -> List(
      (4011400, 62.32069762769104, 1.6),
      (5011400, 61.87859182931989, 1.6),
      (7011400, 62.635283802694026, 1.6),
      (8011400, 62.10761612221206, 1.6),
      (8111400, 62.156220286517915, 1.6),
      (9011400, 62.91013370645605, 1.6)
    ),
    CATEGORY_SHIBA_SEMI_LONG -> List(
      (1011500, 64.79488775205323, 1.65),
      (4111600, 65.8310792327133, 1.7),
      (5011600, 66.35430603535279, 1.7),
      (6111600, 66.91082508462358, 1.7),
      (7011600, 67.06212452850785, 1.7),
      (8011600, 66.31872976364536, 1.7),
      (8111600, 66.5769707471643, 1.7),
      (9111600, 66.46972603448327, 1.7)
    ),
    CATEGORY_SHIBA_LONG -> List(
      (1011800, 72.21124617608055, 1.8),
      (2011800, 72.58417383781706, 1.8),
      (3011800, 71.74124451290552, 1.8),
      (4111800, 70.33915883633739, 1.8),
      (5011800, 71.10074863462451, 1.8),
      (6011800, 71.62469368864227, 1.8),
      (8111800, 71.05337612385843, 1.8),
      (9111800, 71.16610721519271, 1.8),
      (10011800, 71.59232846225483, 1.8)
    ),
    CATEGORY_SHIBA_VERY_LONG -> List(
      (1012000, 76.45972732348218, 1.9),
      (2012000, 77.26726565142266, 1.9),
      (3012000, 75.95775271577878, 1.9),
      (4012000, 75.77892194461978, 1.9),
      (4112000, 75.34453551553757, 1.9),
      (5012000, 75.76434212060003, 1.9),
      (6012000, 76.28120829678231, 1.9),
      (7012000, 76.60509836134192, 1.9),
      (8012000, 75.62260477911236, 1.9),
      (9012000, 76.50575269886089, 1.9),
      (10012000, 75.9739354057137, 1.9)
    ),
    CATEGORY_SHIBA_VERY_VERY_LONG -> List(
      (4012200, 80.8271009433692, 2.0),
      (6112200, 80.8775413373504, 2.0),
      (7012200, 81.56391314533124, 2.0),
      (8112200, 80.24050075710095, 2.0),
      (9012200, 81.18994119275396, 2.0),
      (4012400, 85.21327853683113, 2.1),
      (5012400, 85.23880234100902, 2.1),
      (8112400, 84.69486792604708, 2.1),
      (9112400, 85.39142835499726, 2.1),
      (6012500, 88.07025106322132, 2.15),
      (3012600, 90.17732577863964, 2.2),
      (10012600, 89.8678618313929, 2.2)
    ),
    CATEGORY_DIRT_SHORT -> List(
      (1001000, 56.65366843818948, 1.7),
      (2001000, 56.34456431088742, 1.7),
      (10001000, 55.677998817568216, 1.7),
      (3001150, 61.181698168273975, 1.85),
      (4001200, 62.17331619558795, 1.9),
      (6001200, 62.36161696497647, 1.9),
      (7001200, 62.30585916455702, 1.9),
      (8001200, 62.089306705145134, 1.9),
      (9001200, 61.69054650928704, 1.9)
    ),
    CATEGORY_DIRT_MIDDLE -> List(
      (5001300, 62.83805536322283, 1.95),
      (5001400, 65.32938514216086, 2.0),
      (7001400, 66.3315247433738, 2.0),
      (8001400, 66.29390770224329, 2.0),
      (9001400, 66.23312028211869, 2.0)
    ),
    CATEGORY_DIRT_SEMI_LONG -> List(
      (5001600, 70.54264586955897, 2.15),
      (1001700, 73.98526612947502, 2.2),
      (2001700, 73.95632333512083, 2.2),
      (3001700, 74.0850990886107, 2.2),
      (7001700, 74.35151929425683, 2.2),
      (10001700, 73.46372436118041, 2.2)
    ),
    CATEGORY_DIRT_LONG -> List(
      (4001800, 76.20457838598593, 2.25),
      (6001800, 76.62020099874381, 2.25),
      (7001800, 76.28264183789895, 2.25),
      (8001800, 75.91290675725936, 2.25),
      (9001800, 75.78226901527742, 2.25),
      (8001900, 78.6487250132651, 2.3),
      (9002000, 80.80532496903841, 2.35),
      (5002100, 82.57934599815749, 2.4)
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
    }.filter(x => x._2.head.isGoodBaba && x._2.head.raceDate >= 20100000)

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
                    time <- timeRace.filter(_._1 == data.raceType) ++ timeRaceSecondary.flatMap(_.filter(_._1 == data.raceType))
                  } yield {
                    val m = time._2
                    val s = time._3

                    val stdScore = (m - data.stdTime) / s * 10 + 50

                    timeErrorRaceMap.foreach {
                      case (category, seq) =>
                        timeErrorRaceMap.put(category, seq.map {
                          case (raceType, error, errorCount) if raceType == data.raceType =>
                            (raceType, error + (stdScore_ - stdScore) * LEARNING_RATE, errorCount + 1)
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

      timeErrorRaceMap.foreach {
        case (category, seq) =>
          val timeRace = timeRaceMap(category)
          timeRaceMap.put(
            category, seq.map {
              case (raceType, error, errorCount) =>
                val (_, m, s) = timeRace.find(_._1 == raceType).get
                val errorMean = if (errorCount == 0) 0 else error / errorCount
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
        val moneyArray = scala.collection.mutable.ArrayBuffer[Double]()
        var money = 100000.0

        try {
          raceSeq.foreach {
            case (raceId, horses)  if Seq(
              CATEGORY_SHIBA_SHORT,
              CATEGORY_SHIBA_MIDDLE,
              CATEGORY_SHIBA_SEMI_LONG,
              CATEGORY_DIRT_SHORT,
              CATEGORY_DIRT_MIDDLE,
              CATEGORY_DIRT_SEMI_LONG,
              CATEGORY_DIRT_LONG
            ).contains(getRaceCategory(horses.head.raceType)) &&
              horses.map(_.prevDataList.length).sorted.apply(horses.length / 2) >= 3 =>

              val raceCategory = getRaceCategory(horses.head.raceType)
              val secondaryRaceCategory = getSecondaryRaceCategory(raceCategory)
              val raceDate = horses.head.raceDate

              val timeRace = timeRaceMap(raceCategory)
              val timeRaceSecondary = secondaryRaceCategory.map(timeRaceMap)

              val timeListStrict = horses.map {
                horse =>
                  for {
                    data <- horse.prevDataList if raceDate - data.raceDate < 10000
                    time <- timeRace.filter(_._1 == data.raceType)
                  } yield {
                    val m = time._2
                    val s = time._3

                    (m - data.stdTime) / s * 10 + 50
                  }
              }

              val timeList = horses.map {
                horse =>
                  for {
                    data <- horse.prevDataList if raceDate - data.raceDate < 10000
                    time <- timeRace.filter(_._1 == data.raceType) ++ timeRaceSecondary.flatMap(_.filter(_._1 == data.raceType))
                  } yield {
                    val m = time._2
                    val s = time._3

                    (m - data.stdTime) / s * 10 + 50
                  }
              }

              val res = horses.zip(timeListStrict.zip(timeList)).map {
                case (horse, (prevStdListStrict, prevStdList)) =>
                  val timeStrict = prevStdListStrict.sortBy(-_) match {
                    case Nil => Double.NaN
                    case list =>
                      mean(list.take(list.length / 2 + 1))
                  }
                  val time = prevStdList.sortBy(-_) match {
                    case Nil => Double.NaN
                    case list =>
                      mean(list.take(list.length / 2 + 1))
                  }
                  (horse.copy(prevDataList = Nil), timeStrict, time)
              }.sortBy(_._1.odds).toSeq

              val (timeStrictMean, timeStrictMid) = res.toList.map(_._2).filterNot(_.isNaN) match {
                case Nil => (Double.NaN, Double.NaN)
                case list => (mean(list), list.sorted.apply(list.length / 2))
              }

              val (timeMean, timeMid) = res.toList.map(_._3).filterNot(_.isNaN) match {
                case Nil => (Double.NaN, Double.NaN)
                case list => (mean(list), list.sorted.apply(list.length / 2))
              }

              val stdRes = res.map {
                case (horse, timeStrict, time) =>
                  val std = if (timeStrict.isNaN)
                    time - (timeStrictMean * 2 + timeMean) / 3
                  else
                    (timeStrict * 2 + time - (timeStrictMean * 2 + timeMean)) / 3
                  (horse, std)
              }

              val removeSeq = if (timeMean.isNaN)
                Nil
              else
                stdRes.filter {
                  x => x._2 < STD_THRESHOLD
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

              val score = (x: (PredictData, Double)) =>
                Math.pow((x._2 + 10) / 100, 1.3) * Math.pow(Math.min(x._1.odds, 100), 0.2)
              val cond1 = (x: (PredictData, Double)) =>
                x._2 >= STD_THRESHOLD &&
                  score(x) > Math.min(1.2 / Math.pow(shareSum, 0.5), 0.15)
              val cond2 = (x: (PredictData, Double)) =>
                Math.pow((x._2 + 10) / 100, 1.3) > 0.135

              val targetNum = Math.sqrt((if (shareSum > SHARE_THRESHOLDS(raceCategory) && res.take(5).count(_._2.isNaN) < 2)
                stdRes.count(x => cond1(x) || cond2(x))
              else if (res.take(5).count(_._2.isNaN) < 2)
                stdRes.count(cond2)
              else
                0) + 1)

              val oddsTop = stdRes.sortBy(_._1.odds).head
              val oddsSecond = stdRes.sortBy(_._1.odds).apply(1)

              if (shareSum > SHARE_THRESHOLDS(raceCategory) && res.take(5).count(_._2.isNaN) < 2 &&
                stdRes.exists(x => cond1(x) || cond2(x))) {
                betRaceCount += 1
                if (stdRes.exists(x => (cond1(x) || cond2(x)) && x._1.rank == 1)) {
                  winRaceCount += 1
                }
                pw.println("%010d".format(raceId.toLong))
                stdRes.filter(x => cond1(x) || cond2(x)).foreach {
                  x =>
                    var bonus = 0
                    if (x._1.age >= 72) {
                      bonus += 50
                    }
                    if (oddsTop._2 < 0) {
                      bonus += 10
                    }
                    if (oddsSecond._2 < 0) {
                      bonus += 10
                    }
                    pw.println(true, x)
                    val betRate = Math.max(money, 1000000) * 0.001 / (res.count(_._2.isNaN) + res.take(5).count(_._2.isNaN) + 1) * (100 + bonus) *
                      Math.pow((x._2 + 10) / 100, 0.4) / targetNum
                    betCount += betRate
                    money -= betRate
                    if (x._1.rank == 1) {
                      winCount += betRate
                      oddsCount += x._1.odds * betRate
                      money += x._1.odds * betRate
                    }
                    moneyArray += money
                }
                stdRes.filterNot(x => cond1(x) || cond2(x)).foreach {
                  x =>
                    pw.println(false, x)
                }
                pw.println
              } else if (res.take(5).count(_._2.isNaN) < 2 && stdRes.exists(x => cond2(x))) {
                betRaceCount += 1
                if (stdRes.exists(x => cond2(x) && x._1.rank == 1)) {
                  winRaceCount += 1
                }
                pw.println("%010d".format(raceId.toLong))
                stdRes.filter(cond2).foreach {
                  x =>
                    var bonus = 0
                    if (x._1.age >= 72) {
                      bonus += 50
                    }
                    if (oddsTop._2 < 0) {
                      bonus += 10
                    }
                    if (oddsSecond._2 < 0) {
                      bonus += 10
                    }
                    pw.println(true, x)
                    val betRate = Math.max(money, 1000000) * 0.001 / (res.count(_._2.isNaN) + res.take(5).count(_._2.isNaN) + 1) * (100 + bonus) *
                      Math.pow((x._2 + 10) / 100, 0.4) / targetNum
                    betCount += betRate
                    money -= betRate
                    if (x._1.rank == 1) {
                      winCount += betRate
                      oddsCount += x._1.odds * betRate
                      money += x._1.odds * betRate
                    }
                    moneyArray += money
                }
                stdRes.filterNot(cond2).foreach {
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

        println(moneyArray.min)
        println(betCount, betRaceCount, winRaceCount / betRaceCount, winCount / betCount, oddsCount / winCount, oddsCount / betCount)

        val xarr = moneyArray.indices.toArray.map(_.toDouble)
        val yarr = moneyArray.toArray

        val f = Figure()
        val p = f.subplot(0)
        p += plot(DenseVector(xarr), DenseVector(yarr), '.')
        p.xlabel = "x axis"
        p.ylabel = "y axis"
        f.saveas("lines.png")

        val outFileStudy = new File("studyResult.csv")
        val pwStudy = new PrintWriter(outFileStudy)

        try {
          timeRaceMap.toSeq.sortBy(_._1).foreach {
            case (_, seq) =>
              pwStudy.println(seq.sortBy(_._1 % 10000).mkString("List(\n", ",\n", "\n)"))
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
      List(CATEGORY_SHIBA_MIDDLE)
    case CATEGORY_SHIBA_MIDDLE =>
      List(CATEGORY_SHIBA_SEMI_LONG)
    case CATEGORY_SHIBA_SEMI_LONG =>
      List(CATEGORY_SHIBA_MIDDLE)
    case CATEGORY_SHIBA_LONG =>
      List(CATEGORY_SHIBA_VERY_LONG, CATEGORY_SHIBA_VERY_VERY_LONG)
    case CATEGORY_SHIBA_VERY_LONG =>
      List(CATEGORY_SHIBA_LONG, CATEGORY_SHIBA_VERY_VERY_LONG)
    case CATEGORY_SHIBA_VERY_VERY_LONG =>
      List(CATEGORY_SHIBA_LONG, CATEGORY_SHIBA_VERY_LONG)
    case CATEGORY_DIRT_SHORT =>
      List(CATEGORY_DIRT_MIDDLE)
    case CATEGORY_DIRT_MIDDLE =>
      List(CATEGORY_DIRT_SHORT)
    case CATEGORY_DIRT_SEMI_LONG =>
      List(CATEGORY_DIRT_LONG)
    case CATEGORY_DIRT_LONG =>
      List(CATEGORY_DIRT_SEMI_LONG)
  }

  def makeRaceType(vector: DenseVector[Double], raceId: Long): Long = {
    val babaCode = (raceId / 1000000) % 100
    babaCode * 1000000 + vector(5).toLong * 100000 + vector(2).toLong * 10000 + vector(6).toLong
  }

  def dateDiff(date1: Long, date2: Long): Double = {
    val year1 = date1 / 10000
    val month1 = (date1 % 10000) / 100
    val day1 = date1 % 100

    val year2 = date2 / 10000
    val month2 = (date2 % 10000) / 100
    val day2 = date2 % 100

    Math.abs(
      year1 * 10.0 + month1 * 10.0 / 12.0 + day1 / 30.0 -
        (year2 * 10.0 + month2 * 10.0 / 12.0 + day2 / 30.0)
    )
  }
}