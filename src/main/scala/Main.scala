import java.io._

import breeze.linalg._
import breeze.stats._

object Main {
  type Gene = DenseVector[Double]

  case class Data(raceDate: Int, raceType: Long, age: Int, rank: Int, odds: Double, time: Double, time3f: Double,
                  raceId: Long, isGoodBaba: Boolean)

  case class PredictData(horseId: Int, raceDate: Int, raceType: Long, age: Double, rank: Int, odds: Double, oddsFuku: Double,
                         time: Double, time3f: Double, isGoodBaba: Boolean, prevDataList: Seq[Data])

  val CATEGORY_SHIBA_SHORT = 0

  val CATEGORY_SHIBA_LONG = 1

  val CATEGORY_DIRT_SHORT = 2

  val CATEGORY_DIRT_LONG = 3

  private[this] val raceTimeMap = scala.collection.mutable.Map[Long, List[Double]]()

  private[this] var maxTimeRaceList: List[Long] = Nil

  val test = List(
    (1001000, 56.52655411255407, 1.3074738356609008),
    (1001700, 74.98053802511637, 2.4704445666189856),
    (1011200, 59.26627725856714, 1.3857313796120014),
    (1011500, 66.41542168674708, 1.5817721085459027),
    (1011800, 73.09517259978401, 1.8248703036564402),
    (1012000, 77.59399999999998, 1.8582487537191434),
    (2001000, 56.743401302747074, 1.5291899014949724),
    (2001700, 74.96261545775599, 2.203518734996997),
    (2011200, 59.76375810103866, 1.6492602252309314),
    (2011800, 73.96731259561437, 2.8156728778213083),
    (2012000, 78.71077396657871, 3.0678664036102883),
    (3001150, 61.40023399558508, 1.5193223064416905),
    (3001700, 75.55523225030099, 2.2392292492810997),
    (3011200, 59.30491114478575, 1.5972456455899147),
    (3011800, 73.21763939309334, 2.076366242344346),
    (3012000, 77.5750989199017, 2.36118111406246),
    (3012600, 91.17091366303451, 2.81983619535265),
    (4001200, 62.482226264259616, 1.7595011112384171),
    (4001800, 77.90205067404378, 2.4919766661408205),
    (4011200, 59.054986997093444, 1.3022911589234374),
    (4011400, 63.88954507071987, 1.756377139443304),
    (4012000, 77.9539815489198, 1.8382688029989351),
    (4012200, 81.08975494537943, 1.7823763613447339),
    (4111600, 66.75546127610005, 1.5131782010720647),
    (4111800, 71.41831140350881, 1.6400639539826853),
    (4112000, 75.01914849428877, 1.5144297290273345),
    (5001300, 64.63983043284242, 2.0283890281303556),
    (5001400, 67.00017746678841, 2.1844553339408423),
    (5001600, 71.43333765597515, 2.285836685857262),
    (5002100, 83.3509392611145, 2.809139593406518),
    (5011400, 62.83251648245111, 1.605581784902636),
    (5011600, 67.27828177800882, 1.7053973436544096),
    (5011800, 71.76319554382896, 2.1777629505107345),
    (5012000, 76.14664836122746, 1.7878610446780874),
    (5012400, 85.25626535626553, 2.3890621821301052),
    (6001200, 62.90826331481219, 1.9382591867952472),
    (6001800, 79.13110812425354, 2.8797663785334997),
    (6011800, 72.88086860912587, 1.778223201186664),
    (6012000, 77.60719553308034, 2.1183703221466414),
    (6111200, 58.54579518480834, 1.4252366402046035),
    (6111600, 68.13755333614658, 1.766964873091891),
    (6112200, 81.8301582532051, 2.375533354324408),
    (7001000, 57.202571428571396, 1.2760578814247938),
    (7001200, 62.56778285134449, 1.5813469220764669),
    (7001400, 67.1955133555928, 1.9445052921577715),
    (7001700, 75.07376776289088, 2.0683360096996117),
    (7001800, 77.7577370030581, 2.365637271942422),
    (7011200, 59.105556868304966, 1.4985457063199923),
    (7011400, 64.22735302635388, 1.7376817417818178),
    (7011600, 68.38187631027267, 1.6091696962923376),
    (7011800, 72.96638461538467, 1.7660755003483146),
    (7012000, 77.98107395887327, 2.100680780796424),
    (8001200, 62.03099026404533, 1.8204991325333997),
    (8001400, 66.96586804410725, 2.03427032507443),
    (8001800, 76.92865541922284, 2.548913996350728),
    (8001900, 78.35120313374375, 2.219791531744543),
    (8011200, 57.82250568575101, 1.3336138196294707),
    (8011400, 63.57032625956406, 1.4611307936085587),
    (8011600, 68.32838719823839, 1.6351780927808806),
    (8012000, 76.42602339181286, 1.878506212672539),
    (8111400, 62.15835037491483, 1.256459467992518),
    (8111600, 66.6025213838886, 2.0819938795784734),
    (8111800, 71.9567966360857, 1.8219661738432549),
    (8112200, 80.61349955076369, 2.1253292691274828),
    (8112400, 84.95039855072453, 2.257000060477995),
    (9001200, 62.28088697017252, 2.051396026400355),
    (9001400, 67.00195548616955, 2.091395070358406),
    (9001800, 77.43945840554578, 2.5299343554190536),
    (9002000, 80.36188235294114, 2.1884816795784228),
    (9011200, 58.68569297531873, 1.4027288756522203),
    (9011400, 63.57843883546991, 1.4938551436417742),
    (9012000, 77.35255534864467, 2.093894852948091),
    (9012200, 81.78536106750397, 2.0543650926606465),
    (9111600, 67.47864086408651, 1.6789626199735577),
    (9111800, 71.87544853635492, 1.8880180010856515),
    (9112400, 85.78649344569291, 2.1331676524049743),
    (10001000, 56.51316997063153, 1.4538235045729428),
    (10001700, 74.79594137312472, 2.4327316513911583),
    (10011200, 58.93778253883242, 1.4976986245568726),
    (10011800, 72.5662011173183, 1.8736050814465037),
    (10012000, 77.19079514824797, 2.1966104825569666)
  )

  private[this] val timeRaceMap = Map[Int, List[(Int, Double, Double)]](
    CATEGORY_SHIBA_SHORT -> List(
      (1011200, 59.26627725856714, 1.3857313796120014),
      (1011500, 66.41542168674708, 1.5817721085459027),
      (2011200, 59.76375810103866, 1.6492602252309314),
      (3011200, 59.30491114478575, 1.5972456455899147),
      (4011200, 59.054986997093444, 1.3022911589234374),
      (4011400, 63.88954507071987, 1.756377139443304),
      (4111600, 66.75546127610005, 1.5131782010720647),
      (5011400, 62.83251648245111, 1.605581784902636),
      (5011600, 67.27828177800882, 1.7053973436544096),
      (6111200, 58.54579518480834, 1.4252366402046035),
      (6111600, 68.13755333614658, 1.766964873091891),
      (7011200, 59.105556868304966, 1.4985457063199923),
      (7011400, 64.22735302635388, 1.7376817417818178),
      (7011600, 68.38187631027267, 1.6091696962923376),
      (8011200, 57.82250568575101, 1.3336138196294707),
      (8011400, 63.57032625956406, 1.4611307936085587),
      (8011600, 68.32838719823839, 1.6351780927808806),
      (8111400, 62.15835037491483, 1.256459467992518),
      (8111600, 66.6025213838886, 2.0819938795784734),
      (9011200, 58.68569297531873, 1.4027288756522203),
      (9011400, 63.57843883546991, 1.4938551436417742),
      (9111600, 67.47864086408651, 1.6789626199735577),
      (10011200, 58.93778253883242, 1.4976986245568726)
    ),
    CATEGORY_SHIBA_LONG -> List(
      (1011800, 73.09517259978401, 1.8248703036564402),
      (1012000, 77.59399999999998, 1.8582487537191434),
      (2011800, 73.96731259561437, 2.8156728778213083),
      (2012000, 78.71077396657871, 3.0678664036102883),
      (3011800, 73.21763939309334, 2.076366242344346),
      (3012000, 77.5750989199017, 2.36118111406246),
      (3012600, 91.17091366303451, 2.81983619535265),
      (4012000, 77.9539815489198, 1.8382688029989351),
      (4012200, 81.08975494537943, 1.7823763613447339),
      (4111800, 71.41831140350881, 1.6400639539826853),
      (4112000, 75.01914849428877, 1.5144297290273345),
      (5011800, 71.76319554382896, 2.1777629505107345),
      (5012000, 76.14664836122746, 1.7878610446780874),
      (5012400, 85.25626535626553, 2.3890621821301052),
      (6011800, 72.88086860912587, 1.778223201186664),
      (6012000, 77.60719553308034, 2.1183703221466414),
      (6112200, 81.8301582532051, 2.375533354324408),
      (7011800, 72.96638461538467, 1.7660755003483146),
      (7012000, 77.98107395887327, 2.100680780796424),
      (8012000, 76.42602339181286, 1.878506212672539),
      (8111800, 71.9567966360857, 1.8219661738432549),
      (8112200, 80.61349955076369, 2.1253292691274828),
      (8112400, 84.95039855072453, 2.257000060477995),
      (9012000, 77.35255534864467, 2.093894852948091),
      (9012200, 81.78536106750397, 2.0543650926606465),
      (9111800, 71.87544853635492, 1.8880180010856515),
      (9112400, 85.78649344569291, 2.1331676524049743),
      (10011800, 72.5662011173183, 1.8736050814465037),
      (10012000, 77.19079514824797, 2.1966104825569666)
    ),
    CATEGORY_DIRT_SHORT -> List(
      (1001000, 56.52655411255407, 1.3074738356609008),
      (1001700, 74.98053802511637, 2.4704445666189856),
      (2001000, 56.743401302747074, 1.5291899014949724),
      (2001700, 74.96261545775599, 2.203518734996997),
      (3001150, 61.40023399558508, 1.5193223064416905),
      (3001700, 75.55523225030099, 2.2392292492810997),
      (4001200, 62.482226264259616, 1.7595011112384171),
      (5001300, 64.63983043284242, 2.0283890281303556),
      (5001400, 67.00017746678841, 2.1844553339408423),
      (5001600, 71.43333765597515, 2.285836685857262),
      (6001200, 62.90826331481219, 1.9382591867952472),
      (7001000, 57.202571428571396, 1.2760578814247938),
      (7001200, 62.56778285134449, 1.5813469220764669),
      (7001400, 67.1955133555928, 1.9445052921577715),
      (7001700, 75.07376776289088, 2.0683360096996117),
      (8001200, 62.03099026404533, 1.8204991325333997),
      (8001400, 66.96586804410725, 2.03427032507443),
      (9001200, 62.28088697017252, 2.051396026400355),
      (9001400, 67.00195548616955, 2.091395070358406),
      (10001000, 56.51316997063153, 1.4538235045729428),
      (10001700, 74.79594137312472, 2.4327316513911583)
    ),
    CATEGORY_DIRT_LONG -> List(
      (4001800, 77.90205067404378, 2.4919766661408205),
      (5002100, 83.3509392611145, 2.809139593406518),
      (6001800, 79.13110812425354, 2.8797663785334997),
      (7001800, 77.7577370030581, 2.365637271942422),
      (8001800, 76.92865541922284, 2.548913996350728),
      (8001900, 78.35120313374375, 2.219791531744543),
      (9001800, 77.43945840554578, 2.5299343554190536),
      (9002000, 80.36188235294114, 2.1884816795784228)
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
            new PredictData(horseId = d(1).toInt, raceDate = d(2).toInt, age = d(3).toInt, rank = rank,
              odds = d(d.length - 5), oddsFuku = oddsFuku, time = d(d.length - 6),
              time3f = d(d.length - 7), raceType = raceType, isGoodBaba = x(11) + x(12) == 1.0 && x(7) + x(8) == 1.0,
              prevDataList = Nil)
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
          val rank = d(d.length - 1).toInt
          val horse_ = new Data(raceDate = d(2).toInt, age = d(3).toInt, rank = rank,
            odds = d(d.length - 5).toInt, time = d(d.length - 6), time3f = d(d.length - 7), raceId = raceId.toLong,
            raceType = raceType, isGoodBaba = x(11) + x(12) == 1.0 && x(7) + x(8) == 1.0)
          val horse = horse_.copy(time3f = raceMap(raceId).filter(x => x.rank >= horse_.rank).map(_.time3f).sorted.head)
          horse
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
          race =>
            val (m, s) = timeRaceFlattenMap(race.raceType)
            val stdTime = (m - race.time) / s * 10 + 50
            raceTimeMap.put(race.raceType, stdTime :: raceTimeMap.getOrElse(race.raceType, Nil))
        }
        (0 to 3).foreach {
          raceCategory =>
            val timeRace = timeRaceMap(raceCategory)
            val infos = for {
              race <- races
              time <- timeRace.find(_._1 == race.raceType).toList
            } yield {
              val m = time._2
              val s = time._3

              (race.raceType, (m - race.time) / s * 10 + 50)
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
                prev <- horse.prevDataList if raceDate - prev.raceDate < 10000
                time <- timeRace.find(_._1 == prev.raceType).toList
              } yield {
                val m = time._2
                val s = time._3

                (m - prev.time) / s * 10 + 50
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
            case (horse, time, time3f) =>
              (horse, (time - timeMean) / timeStd * 10 + 50)
          }
          val oddsTop = stdRes.sortBy(_._1.odds).head
          val prevLengthMean = mean(horses.map(_.prevDataList.length.toDouble).toSeq)

          val removeSeq = if (timeMean.isNaN)
            Nil
          else
            stdRes.filter {
              x => x._2 < 50
            }

          val shareSum = removeSeq.map {
            x =>
              78.8 / (x._1.odds - 1)
          }.sum

          if (shareSum > 60 && res.count(_._2.isNaN) < 3) {
            pw.println("%010d".format(raceId.toLong))
            betRaceCount += 1
            if (stdRes.exists(x => x._2 >= 50 && x._1.rank == 1)) {
              winRaceCount += 1
            }
            stdRes.filter {
              x =>
                x._2 >= 50
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
                x._2 >= 50
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