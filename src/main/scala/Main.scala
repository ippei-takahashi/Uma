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
  private[this] val raceTime3fMap = scala.collection.mutable.Map[Long, List[Double]]()

  private[this] val time3fRaceMap = Map[Int, List[(Int, Double, Double)]](
    CATEGORY_SHIBA_SHORT -> List(
      (1011200, 35.67979511143062, 1.0279668620866114),
      (1011500, 36.36524096385539, 1.1555021828293874),
      (2011200, 35.97376434762235, 1.1931978414264874),
      (3011200, 35.496440281030535, 1.1426624081551768),
      (4011200, 35.443937586048604, 0.9763215515617614),
      (4011400, 36.170645978467354, 1.2752696190485617),
      (4111600, 35.018672323434, 1.1476251430613995),
      (5011400, 35.09745200698084, 1.1600726158564252),
      (5011600, 35.680427605926635, 1.2407232129466839),
      (6111200, 35.29796541200404, 1.0388820629902389),
      (6111600, 36.272679701786465, 1.2422251992493607),
      (7011200, 35.46420132325149, 1.1119733078241145),
      (7011400, 36.474978279756694, 1.3006632127864723),
      (7011600, 36.504150943396196, 1.2160952801921898),
      (8011200, 34.732827054286556, 0.993174175599597),
      (8011400, 35.76355348635776, 1.0684022410275884),
      (8111400, 35.1726993865031, 0.9874253242480524),
      (8011600, 36.233097428638726, 1.191909412995646),
      (8111600, 36.00762160091917, 1.589835810475051),
      (9011200, 35.22698942229457, 1.0350978911048494),
      (9011400, 36.13898237179488, 1.108296794772634),
      (9111600, 35.84234473447351, 1.2739163480666071),
      (10011200, 35.36797268344933, 1.0594717023040627)
    ),
    CATEGORY_SHIBA_LONG -> List(
      (1011800, 36.09459007551244, 1.261932594858037),
      (1012000, 36.687333333333334, 1.3596929313622668),
      (2011800, 36.65029066802659, 1.9061260979230343),
      (2012000, 37.296873350923494, 2.087420879263155),
      (3011800, 36.155089360513794, 1.4436482312101377),
      (3012000, 36.5288241899262, 1.6809849502510228),
      (3012600, 36.98224643755245, 1.953247764232521),
      (4111800, 34.95015350877193, 1.2745449371555715),
      (4012000, 36.220415149308054, 1.3664346424875793),
      (4112000, 35.19923676012457, 1.2531136258819677),
      (4012200, 36.19499557130203, 1.3345830440913438),
      (5011800, 35.21559366754612, 1.562836458996014),
      (5012000, 35.438786610878566, 1.3114542738072619),
      (5012400, 36.08666666666663, 1.688997538795615),
      (6011800, 35.91716602528856, 1.2522876215043353),
      (6012000, 36.27758322798138, 1.4279052376388626),
      (6112200, 36.38281250000002, 1.6213192622152042),
      (7011800, 36.137461538461584, 1.2502619042258845),
      (7012000, 36.5901607050286, 1.486585222459748),
      (8111800, 35.468738532110194, 1.3072116364340853),
      (8012000, 35.55102999434079, 1.310426589418678),
      (8112200, 35.820471698113236, 1.4628392362203848),
      (8112400, 35.93108695652174, 1.6545274579723228),
      (9111800, 35.5188385269122, 1.407357436864946),
      (9012000, 36.33738050900063, 1.4936003075666167),
      (9012200, 36.55857142857147, 1.4672826255431282),
      (9112400, 36.389887640449425, 1.6317506694898287),
      (10011800, 35.855505222249105, 1.313088976475071)
    ),
    CATEGORY_DIRT_SHORT -> List(
      (1001000, 36.06810389610391, 0.9252199769957462),
      (1001200, 36.43811320754714, 1.5410856376899456),
      (1001700, 39.20057487091229, 1.7405663073776723),
      (2001000, 36.23992183517411, 1.0837135807995517),
      (2001700, 39.3499682151588, 1.519257148704136),
      (3001150, 37.61706622516566, 1.069383812819924),
      (3001700, 39.49290974729244, 1.5695965818944961),
      (4001200, 37.75101320398814, 1.2526253567700802),
      (5001300, 37.67940696117798, 1.455237849663945),
      (5001400, 38.14184159821524, 1.559447298868707),
      (5001600, 38.59385493213462, 1.6551650567497127),
      (6001200, 38.15755698604123, 1.3859308709475842),
      (7001000, 36.406269841269734, 0.8503877060020076),
      (7001200, 37.689165398275, 1.1528430762321125),
      (7001400, 38.65037562604341, 1.447230406844544),
      (7001700, 39.228305724726105, 1.398193324068824),
      (8001200, 37.29932492328676, 1.2837493212658244),
      (8001400, 38.37785031322124, 1.459461153229115),
      (9001200, 37.51426216640505, 1.4527273046117841),
      (9001400, 38.54121424282452, 1.5116450111245732),
      (10001000, 36.253823788546306, 0.9932854562426685),
      (10001700, 39.4610430528376, 1.6540118473868566)
    ),
    CATEGORY_DIRT_LONG -> List(
      (4001800, 39.37387695955754, 1.76024798382762),
      (5002100, 39.104552285535366, 1.9742509558711114),
      (6001800, 39.880947063688934, 1.9730517873179063),
      (7001800, 39.14513761467885, 1.6598338902757281),
      (8001800, 38.76708769397332, 1.7363323685015226),
      (8001900, 38.89047006155571, 1.5476942364986326),
      (9001800, 38.94173310225296, 1.739957787269001),
      (9002000, 39.03643137254903, 1.5865047190301034)
    )
  )

  private[this] val timeRaceMap = Map[Int, List[(Int, Double, Double)]](
    CATEGORY_SHIBA_SHORT -> List(
      (1011200, 70.71000000000009, 1.308530774100239),
      (1011500, 91.05042168674685, 1.641910724949325),
      (2011200, 70.91930662918683, 1.527635488112701),
      (3011200, 70.18856178536985, 1.4832552743935143),
      (4011200, 70.51421294171643, 1.2408335946981068),
      (4011400, 83.25755541481931, 1.7079776043403587),
      (4111600, 96.11899176441229, 1.799560655785535),
      (5011400, 83.47683827806871, 1.7632745927203473),
      (5011600, 96.09092799584069, 2.0305398302801283),
      (6111200, 69.92206002034575, 1.3617828888766939),
      (6111600, 96.42681108454041, 2.1148937842186086),
      (7011200, 70.5282939508507, 1.4221289298292177),
      (7011400, 83.96398783666373, 1.6706381961598622),
      (7011600, 97.55515723270445, 1.9837617561797114),
      (8011200, 69.94075348561255, 1.2779664027090003),
      (8011400, 83.02236466002607, 1.3856530736884596),
      (8111400, 82.94229038854794, 1.2314045250761843),
      (8011600, 96.41297475819773, 1.5981810676330557),
      (8111600, 96.12052470317876, 1.8609618471671308),
      (9011200, 70.58244914564681, 1.3717250644644232),
      (9011400, 83.3606770833332, 1.4734245641344637),
      (9111600, 96.71004500450051, 1.8077114199275726),
      (10011200, 69.65270487412954, 1.4042418368174117)
    ),
    CATEGORY_SHIBA_LONG -> List(
      (1011800, 111.13313915857604, 2.2269048953416055),
      (1012000, 123.22654545454557, 1.746931001918637),
      (2011800, 111.53834778174414, 3.0702320133372285),
      (2012000, 124.04670184696589, 3.054533341361377),
      (3011800, 109.94930745601775, 2.2351769131542993),
      (3012000, 122.24654796278444, 2.2985173861879735),
      (3012600, 162.3473009220453, 2.9571052683201673),
      (4111800, 108.89203947368408, 1.9424703661063827),
      (4012000, 123.08402767662027, 1.97447838388934),
      (4112000, 121.25146417445479, 1.575442988644012),
      (4012200, 135.18350752878645, 1.8669808444569547),
      (5011800, 109.71523306948133, 2.5330566387842697),
      (5012000, 122.90564330543957, 2.42480057171173),
      (5012400, 148.88654791154817, 2.813832685679168),
      (6011800, 110.88166575041225, 2.312105303314178),
      (6012000, 123.37833122629552, 2.6855830285091202),
      (6112200, 136.63671874999986, 2.8721113736864856),
      (7011800, 110.11938461538443, 1.9525482269767611),
      (7012000, 123.88071021254564, 2.571210861193425),
      (8111800, 109.50077981651368, 2.1860281978121265),
      (8012000, 122.76177136389349, 2.411578431750238),
      (8112200, 136.1006064690025, 2.5407387988703585),
      (8112400, 148.18956521739133, 2.505581838106229),
      (9111800, 109.1279927154994, 2.083246229520343),
      (9012000, 123.93742396027295, 2.5600029117293635),
      (9012200, 136.4931083202511, 2.3278980352268355),
      (9112400, 149.72386235955067, 2.393337959757307),
      (10011800, 109.3377969395189, 1.988295422621954)
    ),
    CATEGORY_DIRT_SHORT -> List(
      (1001000, 60.81498701298702, 1.0884386108084438),
      (1001200, 122.21142857142861, 2.1842925271851628),
      (1001700, 108.14218779881466, 2.2091342584981106),
      (2001000, 61.09152081563298, 1.315948972428854),
      (2001700, 108.123879380603, 2.0660083076813702),
      (3001150, 69.90847019867542, 1.54337537960583),
      (3001700, 108.43178339350178, 2.0533199531550372),
      (4001200, 73.11299649690104, 1.62390216283417),
      (5001300, 80.49577643908961, 1.9182940603189622),
      (5001400, 87.16138119866132, 2.1093234927738673),
      (5001600, 100.00790092504523, 2.211732510526607),
      (6001200, 73.1838257216424, 1.7918671009625888),
      (7001000, 61.11323809523819, 1.1992737864274874),
      (7001200, 73.59482496194825, 1.4406053462910529),
      (7001400, 86.61341402337217, 1.7228558697489882),
      (7001700, 108.40922046285023, 2.0745436512603725),
      (8001200, 73.51467553131009, 1.7216969869101255),
      (8001400, 86.53577096384202, 1.8714697657323283),
      (9001200, 73.60087912087921, 1.890763327337309),
      (9001400, 86.55569712981869, 1.8970243120400319),
      (10001000, 60.94290198237887, 1.298420031463923),
      (10001700, 108.17174902152653, 2.3795920033965743)
    ),
    CATEGORY_DIRT_LONG -> List(
      (4001800, 115.53665129758927, 2.302310098285448),
      (5002100, 135.84802755165958, 3.036325517902439),
      (6001800, 116.99180452164358, 2.848814458209519),
      (7001800, 115.92137614678924, 2.4294068583695996),
      (8001800, 115.08236376759333, 2.616957038338896),
      (8001900, 121.89903749300501, 2.2135110851092303),
      (9001800, 115.48791161178588, 2.553272323001846),
      (9002000, 127.49694117647062, 2.0245398086682442)
    )
  )

  val timeRaceFlattenMap = timeRaceMap.values.toList.flatten.map {
    case (key, m, s) =>
      key.toLong -> (m, s)
  }.toMap

  val time3fRaceFlattenMap = time3fRaceMap.values.toList.flatten.map {
    case (key, m, s) =>
      key.toLong -> (m, s)
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
            new PredictData(horseId = d(1).toInt, raceDate = d(2).toInt, age = d(3).toInt, rank = d(d.length - 1).toInt,
              odds = d(d.length - 2), oddsFuku = (d(d.length - 2) - 1) / 5 + 1, time = d(d.length - 3),
              time3f = d(d.length - 4), raceType = raceType, isGoodBaba = x(11) + x(12) == 1.0 && x(7) + x(8) == 1.0,
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
          val horse_ = new Data(raceDate = d(2).toInt, age = d(3).toInt, rank = d(d.length - 1).toInt,
            odds = d(d.length - 2).toInt, time = d(d.length - 3), time3f = d(d.length - 4), raceId = raceId.toLong,
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

    var raceCount = 0.0
    var oddTopWinCount = 0.0

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
            val (m3f, s3f) = time3fRaceFlattenMap(race.raceType)
            val stdTime = (m - race.time) / s * 10 + 50
            val stdTime3f = (m3f - race.time3f) / s3f * 10 + 50
            raceTimeMap.put(race.raceType, stdTime :: raceTimeMap.getOrElse(race.raceType, Nil))
            raceTime3fMap.put(race.raceType, stdTime3f :: raceTime3fMap.getOrElse(race.raceType, Nil))
        }
    }
    Seq(
      "raceTimeMean.csv" -> raceTimeMap,
      "raceTime3fMean.csv" -> raceTime3fMap
    ).foreach {
      case (fileName, map) =>
        val mat = DenseMatrix(map.toArray.map {
          case (key, list) =>
            val m = mean(list)
            val s = stddev(list)
            (key.toDouble, list.length.toDouble, m, s)
        }.sortBy(_._3): _*)
        csvwrite(new File(fileName), mat)
    }

    try {
      raceSeq.foreach {
        case (raceId, horses)  if horses.head.isGoodBaba =>
          val raceCategory = getRaceCategory(horses.head.raceType)
          val raceDate = horses.head.raceDate

          val timeRace = timeRaceMap(raceCategory)
          val time3fRace = time3fRaceMap(raceCategory)

          val timeList = horses.map {
            horse =>
              for {
                prev <- horse.prevDataList if raceDate - prev.raceDate < 10000
                time <- timeRace.find(_._1 == prev.raceType).toList
                time3f <- time3fRace.find(_._1 == prev.raceType).toList
              } yield {
                val m = time._2
                val s = time._3

                val m3f = time3f._2
                val s3f = time3f._3

                ((m - prev.time) / s * 10 + 50, (m3f - prev.time3f) / s3f * 10 + 50)
              }
          }


          val res = horses.zip(timeList).map {
            case (horse, prevStdList) =>
              val time = prevStdList.map(_._1).sortBy(-_).headOption.getOrElse(Double.NaN)
              val time3f = prevStdList.map(_._2).sortBy(-_).headOption.getOrElse(Double.NaN)
              (horse.copy(prevDataList = Nil), time, time3f)
          }.sortBy(_._1.odds).toSeq

          val (timeMean, timeStd) = res.toList.map(_._2).filterNot(_.isNaN) match {
            case Nil => (Double.NaN, Double.NaN)
            case list => (mean(list), stddev(list))
          }
          val (time3fMean, time3fStd) = res.toList.map(_._3).filterNot(_.isNaN) match {
            case Nil => (Double.NaN, Double.NaN)
            case list => (mean(list), stddev(list))
          }
          val stdRes = res.map {
            case (horse, time, time3f) =>
              (horse, (time - timeMean) / timeStd * 10 + 50, (time3f - time3fMean) / time3fStd * 10 + 50)
          }
          val oddsTop = stdRes.sortBy(_._1.odds).head
          val prevLengthMean = mean(horses.map(_.prevDataList.length.toDouble).toSeq)

          val removeSeq = if (timeMean.isNaN || time3fMean.isNaN)
            Nil
          else
            stdRes.filter {
              x => x._2 < 50 && x._3 < 45
            }

          val shareSum = removeSeq.map {
            x =>
              78.8 / (x._1.odds - 1)
          }.sum

          if (shareSum > 50 && res.count(_._2.isNaN) < 3) {
            betRaceCount += 1
            if (stdRes.exists(x => (x._2 >= 50 || x._3 >= 45) && x._1.rank == 1)) {
              winRaceCount += 1
            }
            stdRes.filter {
              x =>
                (x._2 >= 50 || x._3 >= 45)
            }.foreach {
              x =>
                val betRate = (x._2 + x._3) / 100
                betCount += betRate
                if (x._1.rank == 1) {
                  winCount += betRate
                  oddsCount += x._1.odds * betRate
                }
            }
          }

          if (removeSeq.nonEmpty && oddsTop._2 < 50 && oddsTop._3 < 50) {
            raceCount += 1
            if (oddsTop._1.rank == 1) {
              oddTopWinCount += 1
            }
            pw.println("%010d".format(raceId.toLong))
            stdRes.foreach(pw.println)
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
    println(raceCount, oddTopWinCount / raceCount)
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