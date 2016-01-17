import java.io._

import breeze.linalg._
import breeze.stats._

object Main {
  type Gene = DenseVector[Double]

  case class Data(raceDate: Int, raceType: Long, age: Int, rank: Int, odds: Double, time: Double, time3f: Double,
                  raceId: Long, isGoodBaba: Boolean)

  case class PredictData(horseId: Int, raceDate: Int, raceType: Long, age: Double,  rank: Int, odds: Double, oddsFuku: Double,
                         time: Double, time3f: Double,isGoodBaba: Boolean, prevDataList: Seq[Data])

  val CATEGORY_SHIBA_SHORT = 0

  val CATEGORY_SHIBA_LONG = 1

  val CATEGORY_DIRT_SHORT = 2

  val CATEGORY_DIRT_LONG = 3

  private[this] val raceTimeMap = scala.collection.mutable.Map[Int, List[Double]]()

  private[this] val stdRaceMap = Map[Int, List[(Int, Double, Double)]](
    CATEGORY_SHIBA_SHORT -> List(
      (7111600, 36.184150943396196, 1.2160952801921898),
      (1111500, 36.12524096385539, 1.1555021828293874),
      (4111200, 35.243937586048604, 0.9763215515617614),
      (7111400, 36.174978279756694, 1.3006632127864723),
      (9111200, 34.99698942229457, 1.0350978911048494),
      (1111200, 35.48479511143062, 1.0279668620866114),
      (4111400, 35.940645978467354, 1.2752696190485617),
      (8111200, 34.532827054286556, 0.993174175599597),
      (6111200, 35.11796541200404, 1.0388820629902389),
      (4111600, 34.748672323434, 1.1476251430613995),
      (7111200, 35.23920132325149, 1.1119733078241145),
      (8111400, 35.260562719812455, 1.1652072512783451),
      (2111200, 35.74376434762235, 1.1931978414264874),
      (9111400, 35.83898237179488, 1.108296794772634),
      (9111600, 35.50734473447351, 1.2739163480666071),
      (8111600, 35.71410218978104, 1.4632488512988644),
      (5111400, 34.90245200698084, 1.1600726158564252),
      (6111600, 35.992679701786465, 1.2422251992493607),
      (3111200, 35.256440281030535, 1.1426624081551768),
      (5111600, 35.415427605926635, 1.2407232129466839),
      (10111200, 35.09297268344933, 1.0594717023040627)
    ),
    CATEGORY_SHIBA_LONG -> List(
      (4112200, 36.22499557130203, 1.3345830440913438),
      (3112600, 37.02224643755245, 1.953247764232521),
      (9112200, 36.57857142857147, 1.4672826255431282),
      (7111800, 36.132461538461584, 1.2502619042258845),
      (8112400, 35.97608695652174, 1.6545274579723228),
      (9112400, 36.389887640449425, 1.6317506694898287),
      (8112200, 35.880471698113236, 1.4628392362203848),
      (2112000, 37.326873350923494, 2.087420879263155),
      (1112000, 36.727333333333334, 1.3596929313622668),
      (6112200, 36.38281250000002, 1.6213192622152042),
      (1111800, 36.13959007551244, 1.261932594858037),
      (2111800, 36.65029066802659, 1.9061260979230343),
      (5112400, 36.11666666666663, 1.688997538795615),
      (4112000, 35.626192075218356, 1.5923330981222286),
      (3112000, 36.5438241899262, 1.6809849502510228),
      (9112000, 36.36738050900063, 1.4936003075666167),
      (8112000, 35.58102999434079, 1.310426589418678),
      (3111800, 36.180089360513794, 1.4436482312101377),
      (6111800, 35.94216602528856, 1.2522876215043353),
      (4111800, 34.95015350877193, 1.2745449371555715),
      (10112000, 36.35811320754714, 1.5410856376899456),
      (5112000, 35.458786610878566, 1.3114542738072619),
      (7112000, 36.6101607050286, 1.486585222459748),
      (10111800, 35.860505222249105, 1.313088976475071),
      (8111800, 35.468738532110194, 1.3072116364340853),
      (6112000, 36.26258322798138, 1.4279052376388626),
      (9111800, 35.5188385269122, 1.407357436864946),
      (5111800, 35.18559366754612, 1.562836458996014)
    ),
    CATEGORY_DIRT_SHORT -> List(
      (7001000, 36.369269841269734, 0.8503877060020076),
      (1001000, 36.00410389610391, 0.9252199769957462),
      (7001200, 37.594165398275, 1.1528430762321125),
      (2001000, 36.19592183517411, 1.0837135807995517),
      (7001400, 38.50037562604341, 1.447230406844544),
      (10001000, 36.194823788546306, 0.9932854562426685),
      (5001300, 37.60840696117798, 1.455237849663945),
      (3001150, 37.56006622516566, 1.069383812819924),
      (7001700, 39.069305724726105, 1.398193324068824),
      (2001700, 39.1819682151588, 1.519257148704136),
      (1001700, 39.03657487091229, 1.7405663073776723),
      (3001700, 39.37290974729244, 1.5695965818944961),
      (4001200, 37.69101320398814, 1.2526253567700802),
      (9001200, 37.40926216640505, 1.4527273046117841),
      (10001700, 39.2730430528376, 1.6540118473868566),
      (9001400, 38.40821424282452, 1.5116450111245732),
      (8001200, 37.20932492328676, 1.2837493212658244),
      (8001400, 38.23785031322124, 1.459461153229115),
      (5001400, 38.04184159821524, 1.559447298868707),
      (5001600, 38.43385493213462, 1.6551650567497127),
      (6001200, 38.06055698604123, 1.3859308709475842)
    ),
    CATEGORY_DIRT_LONG -> List(
      (9002000, 38.94443137254903, 1.5865047190301034),
      (8001900, 38.80047006155571, 1.5476942364986326),
      (5002100, 38.964552285535366, 1.9742509558711114),
      (7001800, 39.10513761467885, 1.6598338902757281),
      (4001800, 39.34387695955754, 1.76024798382762),
      (8001800, 38.66708769397332, 1.7363323685015226),
      (9001800, 38.89173310225296, 1.739957787269001),
      (6001800, 39.770947063688934, 1.9730517873179063)
    )
  )

  def main(args: Array[String]) {

    val dataCSV = new File("past.csv")

    val data: DenseMatrix[Double] = csvread(dataCSV)
    val size = data.rows

    val array = Array.ofDim[DenseVector[Double]](size)
    for (i <- 0 until size) {
      array(i) = data(i, ::).t
    }

    val raceMap = array.groupBy(_(0)).map {
      case (raceId, arr) =>
        val horses = arr.map {
          d =>
            val x = d(3 until data.cols - 1)
            val raceType = makeRaceType(x, raceId.toLong)
            new PredictData(horseId = d(1).toInt, raceDate = d(2).toInt, age = d(3).toInt, rank = d(d.length - 1).toInt,
              odds = d(d.length - 2), oddsFuku = (d(d.length - 2) - 1) / 4 + 1, time = d(d.length - 3),
              time3f = d(d.length - 4), raceType = raceType, isGoodBaba = x(11) + x(12) == 1.0 && x(7) == 1.0,
              prevDataList = Nil)
        }

        raceId -> horses
    }
    val raceSeq_ = raceMap.toSeq.sortBy(_._2.head.raceDate)

    val horseMap = array.groupBy(_(1)).map {
      case (horseId, arr) =>
        horseId -> arr.map { d =>
          val raceId = d(0)
          val x = d(3 until data.cols - 1)
          val raceType = makeRaceType(x, raceId.toLong)
          val horse_ = new Data(raceDate = d(2).toInt, age = d(3).toInt, rank = d(d.length - 1).toInt,
            odds = d(d.length - 2).toInt, time = d(d.length - 3), time3f = d(d.length - 4), raceId = raceId.toLong,
            raceType = raceType, isGoodBaba = x(11) + x(12) == 1.0 && x(7) == 1.0)
          val horse = horse_.copy(time3f = raceMap(raceId).filter(x => x.rank >= horse_.rank && x.rank < horse_.rank + 3).map(_.time3f).sorted.head)
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
    var winCount = 0.0
    var oddsCount = 0.0

    try {
      raceSeq.foreach {
        case (raceId, horses) =>
          val raceCategory = getRaceCategory(horses.head.raceType)
          val raceDate = horses.head.raceDate
          val stdRace = stdRaceMap(raceCategory)
          val stdList = horses.map {
            horse =>
              for {
                prev <- horse.prevDataList if raceDate - prev.raceDate < 10000
                race <- stdRace.find(_._1 == prev.raceType).toList
              } yield {
                val m = race._2
                val s = race._3
                (m - prev.time3f) / s * 10 + 50
              }
          }
          val res = horses.zip(stdList).map {
            case (horse, prevStdList) =>
              (horse.copy(prevDataList = Nil), prevStdList.sortBy(-_).headOption.getOrElse(Double.NaN))
          }.sortBy(-_._2)

          val scoreMean = res.toList.map(_._2).filterNot(_.isNaN) match {
            case Nil => Double.NaN
            case list => mean(list)
          }
          val oddsTop = res.sortBy(_._1.odds).head
          val prevLengthMean = mean(horses.map(_.prevDataList.length.toDouble).toSeq)

          if (!scoreMean.isNaN && !oddsTop._2.isNaN && scoreMean > oddsTop._2 && prevLengthMean > 5) {
            raceCount += 1
            if (oddsTop._1.rank == 1) {
              winCount += 1
            }
            pw.println("%010d".format(raceId.toLong))
            res.foreach(pw.println)
            pw.println
          }
      }
    } catch {
      case e: Exception =>
    } finally {
      pw.close
    }

    println(raceCount, winCount / raceCount)
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