import java.io._

import breeze.linalg._
import breeze.stats._

object Main {
  type Gene = DenseVector[Double]

  case class Data(raceDate: Int, raceType: Long, age: Int, rank: Int, odds: Double, stdTime: Double, raceId: Long,
                  paceRank: Int, isGoodBaba: Boolean)

  case class PredictData(horseId: Int, raceDate: Int, raceType: Long, age: Double, rank: Int, odds: Double, oddsFuku: Double,
                         stdTime: Double, paceRank: Int, isGoodBaba: Boolean, prevDataList: Seq[Data])

  val CATEGORY_SHIBA_SHORT = 0

  val CATEGORY_SHIBA_LONG = 1

  val CATEGORY_DIRT_SHORT = 2

  val CATEGORY_DIRT_LONG = 3

  private[this] val raceTimeMap = scala.collection.mutable.Map[Long, List[Double]]()

  private[this] var maxTimeRaceList: List[Long] = Nil

  private[this] val timeRaceMap = Map[Int, List[(Int, Double, Double)]](
    CATEGORY_SHIBA_SHORT -> List(
      (1011200, 59.86627725856714, 1.4),
      (1011500, 67.21542168674708, 1.6),
      (2011200, 60.46375810103866, 1.4),
      (3011200, 59.60491114478575, 1.4),
      (4011200, 59.554986997093444, 1.4),
      (4011400, 64.68954507071987, 1.6),
      (4111600, 67.75546127610005, 1.7),
      (5011400, 63.93251648245111, 1.6),
      (5011600, 68.67828177800882, 1.7),
      (6111200, 59.44579518480834, 1.4),
      (6111600, 69.23755333614658, 1.7),
      (7011200, 59.805556868304966, 1.4),
      (7011400, 65.22735302635388, 1.6),
      (7011600, 69.58187631027267, 1.7),
      (8011200, 58.92250568575101, 1.4),
      (8011400, 63.97032625956406, 1.6),
      (8011600, 68.72838719823839, 1.7),
      (8111400, 64.05835037491483, 1.5),
      (8111600, 68.6025213838886, 1.6),
      (9011200, 59.58569297531873, 1.4),
      (9011400, 64.77843883546991, 1.6),
      (9111600, 68.87864086408651, 1.7),
      (10011200, 59.43778253883242, 1.4)
    ),
    CATEGORY_SHIBA_LONG -> List(
      (1011800, 73.39517259978401, 1.8),
      (1012000, 77.99399999999998, 2.1),
      (2011800, 73.96731259561437, 1.8),
      (2012000, 78.91077396657871, 2.1),
      (3011800, 72.71763939309334, 1.8),
      (3012000, 77.5750989199017, 2.1),
      (3012600, 91.47091366303451, 2.8),
      (4012000, 77.3539815489198, 2.1),
      (4012200, 81.78975494537943, 2.3),
      (4111800, 71.41831140350881, 1.6),
      (4112000, 76.11914849428877, 1.9),
      (5011800, 72.26319554382896, 1.8),
      (5012000, 77.04664836122746, 2.1),
      (5012400, 86.35626535626553, 2.4),
      (6011800, 73.08086860912587, 1.8),
      (6012000, 77.80719553308034, 2.1),
      (6112200, 82.4301582532051, 2.3),
      (7011800, 72.96638461538467, 1.8),
      (7012000, 78.08107395887327, 2.1),
      (8012000, 77.02602339181286, 2.1),
      (8111800, 72.1567966360857, 1.6),
      (8112200, 81.61349955076369, 2.1),
      (8112400, 85.95039855072453, 2.2),
      (9012000, 77.95255534864467, 2.1),
      (9012200, 82.48536106750397, 2.3),
      (9111800, 72.17544853635492, 1.6),
      (9112400, 86.48649344569291, 2.2),
      (10011800, 72.7662011173183, 1.8),
      (10012000, 77.39079514824797, 2.1)
    ),
    CATEGORY_DIRT_SHORT -> List(
      (1001000, 55.32655411255407, 1.4),
      (1001700, 71.78053802511637, 2.3),
      (2001000, 55.443401302747074, 1.4),
      (2001700, 71.86261545775599, 2.3),
      (3001150, 60.10023399558508, 1.6),
      (3001700, 72.45523225030099, 2.3),
      (4001200, 61.082226264259616, 1.7),
      (5001300, 63.03983043284242, 1.8),
      (5001400, 65.60017746678841, 1.9),
      (5001600, 69.53333765597515, 2.1),
      (6001200, 61.40826331481219, 1.7),
      (7001000, 55.602571428571396, 1.4),
      (7001200, 61.36778285134449, 1.7),
      (7001400, 66.1955133555928, 1.9),
      (7001700, 72.27376776289088, 2.3),
      (8001200, 61.13099026404533, 1.7),
      (8001400, 66.06586804410725, 1.9),
      (9001200, 60.88088697017252, 1.7),
      (9001400, 66.00195548616955, 1.9),
      (10001000, 54.91316997063153, 1.4),
      (10001700, 71.79594137312472, 2.3)
    ),
    CATEGORY_DIRT_LONG -> List(
      (4001800, 76.30205067404378, 2.5),
      (5002100, 82.6509392611145, 2.8),
      (6001800, 77.03110812425354, 2.5),
      (7001800, 76.4577370030581, 2.5),
      (8001800, 76.02865541922284, 2.5),
      (8001900, 78.55120313374375, 2.6),
      (9001800, 76.13945840554578, 2.5),
      (9002000, 80.76188235294114, 2.7)
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
            val stdTime = time / 3 + time3f + (if (paceRank <= 6)
              0.1 * (paceRank - 6)
            else
              0.05 * (paceRank - 6)
              )
            new PredictData(horseId = d(1).toInt, raceDate = d(2).toInt, age = d(3).toInt, rank = rank,
              odds = d(d.length - 5), oddsFuku = oddsFuku, stdTime = stdTime, raceType = raceType, isGoodBaba = x(11) + x(12) == 1.0 && x(7) + x(8) == 1.0,
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
          val stdTime = time / 3 + time3f + (if (paceRank <= 6)
            0.1 * (paceRank - 6)
          else
            0.05 * (paceRank - 6)
            )
          new Data(raceDate = d(2).toInt, age = d(3).toInt, rank = rank,
            odds = d(d.length - 5).toInt, stdTime = stdTime, raceId = raceId.toLong,
            raceType = raceType, paceRank = paceRank, isGoodBaba = x(11) + x(12) == 1.0 && x(7) + x(8) == 1.0)
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
        (0 to 3).foreach {
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
            (key.toDouble, list.length.toDouble / map2(key).length.toDouble)
        }.sortBy(_._2): _*)
        csvwrite(new File(fileName), mat)
    }

    try {
      raceSeq.foreach {
        case (raceId, horses) if horses.head.isGoodBaba =>
          val raceCategory = getRaceCategory(horses.head.raceType)
          val raceDate = horses.head.raceDate

          val timeRace = timeRaceMap(raceCategory)

          val timeList = horses.map {
            horse =>
              for {
                data <- horse.prevDataList if raceDate - data.raceDate < 10000
                time <- timeRace.find(_._1 == data.raceType).toList
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
                case list => mean(list.take(3))
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
              x => x._2 < 47
            }

          val shareSum = removeSeq.map {
            x =>
              78.8 / (x._1.odds - 1)
          }.sum

          if (shareSum > 55 && res.count(_._2.isNaN) < 3) {
            pw.println("%010d".format(raceId.toLong))
            betRaceCount += 1
            if (stdRes.exists(x => x._2 >= 47 && x._1.rank == 1)) {
              winRaceCount += 1
            }
            stdRes.filter {
              x =>
                x._2 >= 47
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
                x._2 >= 47
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
    case (1, dist) if dist <= 1600 =>
      CATEGORY_SHIBA_SHORT
    case (1, dist) =>
      CATEGORY_SHIBA_LONG
    case (0, dist) if dist <= 1600 =>
      CATEGORY_DIRT_SHORT
    case (0, dist) =>
      CATEGORY_DIRT_LONG
  }

  def makeRaceType(vector: DenseVector[Double], raceId: Long): Long = {
    val babaCode = (raceId / 1000000) % 100
    babaCode * 1000000 + vector(5).toLong * 100000 + vector(2).toLong * 10000 + vector(6).toLong
  }
}