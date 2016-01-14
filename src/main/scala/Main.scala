import java.io._

import breeze.linalg._
import breeze.stats._

object Main {
  type Gene = DenseVector[Double]

  case class Data(raceDate: Int, age: Int, rank: Int, odds: Double, time: Double, raceId: Long, raceType: Long, isGoodBaba: Boolean)

  case class PredictData(horseId: Int, raceDate: Int, raceType: Long, rank: Int, odds: Double, oddsFuku: Double,
                         time: Double, age: Double, isGoodBaba: Boolean, prevDataList: Seq[Data])

  val CATEGORY_SHIBA_SHORT = 0

  val CATEGORY_SHIBA_LONG = 1

  val CATEGORY_DIRT_SHORT = 2

  val CATEGORY_DIRT_LONG = 3

  private[this] val raceTimeMap = scala.collection.mutable.Map[Int, List[Double]]()

  private[this] val stdRaceMap = Map[Int, List[(Int, Double, Double)]](
    CATEGORY_SHIBA_SHORT -> List(
      (111200, 70.24547511312217, 1.3971966032000496),
      (611200, 69.36627282491943, 1.4908284778804677),
      (211200, 70.33542168674698, 1.2819770069306973),
      (411400, 82.77350427350427, 1.7879195487516562),
      (711200, 69.93351648351648, 1.3274829531954615),
      (811400, 82.46907131011609, 1.4066224714402877),
      (411600, 95.62542213883678, 1.8411218824595075),
      (911400, 82.86107117181884, 1.5659055058288736),
      (911600, 96.18441860465116, 1.8418874785189014),
      (511400, 82.95085344320188, 1.7293116687042962),
      (611600, 95.92786615469007, 2.1861284090289956),
      (311200, 69.63586216289839, 1.487017042816897),
      (811600, 95.64538216560509, 1.7949165490406667),
      (511600, 95.50099800399201, 1.990869851887232)
    ),
    CATEGORY_SHIBA_LONG -> List(
      (312000, 121.49013698630137, 2.08316961625407),
      (412000, 121.48623922959959, 2.365516605499596),
      (912000, 123.26569620253164, 2.5205250069980436),
      (611800, 110.06941120153184, 2.255786023156695),
      (812000, 122.01967213114754, 2.4741563717498885),
      (1012000, 121.25686456400742, 2.0615796439905787),
      (512000, 122.19173174092789, 2.421929071629471),
      (311800, 109.16940966010734, 2.059611355440442),
      (712000, 123.09738675958189, 2.3328649689182326),
      (411800, 108.1749569707401, 2.013894107759768),
      (1011800, 108.64453441295547, 2.0121442969528274),
      (811800, 108.79785875281743, 2.2397944081856873),
      (612000, 122.5134110787172, 2.5562905527012774),
      (911800, 108.41492842535788, 2.016384145904715),
      (511800, 108.91318124207858, 2.559480165892545)
    ),
    CATEGORY_DIRT_SHORT -> List(
      (301150, 69.96671018276762, 1.7592070495340544),
      (501300, 80.60885416666667, 2.0170649611275193),
      (901200, 73.71911298838437, 1.8006114146185128),
      (901400, 86.69107896323086, 1.9826543523189482),
      (401200, 73.15599471054952, 1.6411268647534247),
      (801200, 73.55891860123421, 1.848696000416368),
      (801400, 86.73286852589642, 2.023490175911063),
      (501400, 87.25122261844116, 2.169095859747181),
      (501600, 100.05096256684492, 2.2720015279892576),
      (601200, 73.21419403548508, 1.8870009612147394)
    ),
    CATEGORY_DIRT_LONG -> List(
      (701700, 108.15888483585201, 2.0775308715357026),
      (201700, 107.91068883610451, 2.158105410873806),
      (101700, 107.92916041979011, 2.3173115895639134),
      (1001700, 107.90361093038387, 2.2782467753163655),
      (301700, 108.30663758180118, 2.1285282151499425),
      (401800, 115.39772727272727, 2.3277014532411786),
      (801800, 115.03623455294403, 2.670184758029812),
      (901800, 115.48652741514361, 2.6392406646362865),
      (601800, 117.02284644194756, 2.960570401032924)
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

    val raceSeq_ = array.groupBy(_(0)).map {
      case (raceId, arr) => raceId -> arr.groupBy(_ (1)).map {
        case (horseId, arr2) =>
          val races = arr2.map { d =>
            val x = d(3 until data.cols - 1)
            val raceType = makeRaceType(x, raceId.toLong)
            new Data(raceDate = d(2).toInt, age = d(3).toInt, rank = d(d.length - 1).toInt,
              odds = d(d.length - 2).toInt, time = d(d.length - 3).toInt, raceId = raceId.toLong,
              raceType = raceType, isGoodBaba = x(9) + x(10) == 1.0 && x(5) == 1.0)
          }.toList
          races match {
            case head :: _ =>
              Some(PredictData(
                horseId = horseId.toInt, raceDate = head.raceDate, raceType = head.raceType, rank = head.rank, odds = head.odds,
                oddsFuku = (head.odds - 1) / 4 + 1, time = head.time, age = head.age, isGoodBaba = head.isGoodBaba,
                prevDataList = Nil)
              )
            case _ =>
              None
          }
      }.toArray.collect {
        case Some(x) => x
      }
    }.toSeq.sortBy(_._2.head.raceDate)

    val horseMap = array.groupBy(_(1)).map {
      case (horseId, arr) =>
        horseId -> arr.map { d =>
          val raceId = d(0)
          val x = d(3 until data.cols - 1)
          val raceType = makeRaceType(x, raceId.toLong)
          new Data(raceDate = d(2).toInt, age = d(3).toInt, rank = d(d.length - 1).toInt,
            odds = d(d.length - 2).toInt, time = d(d.length - 3).toInt, raceId = raceId.toLong,
            raceType = raceType, isGoodBaba = x(9) + x(10) == 1.0 && x(5) == 1.0)
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
          val stdRace = stdRaceMap(raceCategory)
          val stdList = horses.map {
            horse =>
              for {
                prev <- horse.prevDataList
                race <- stdRace.find(_._1 == prev.raceType).toList
              } yield {
                val m = race._2
                val s = race._3
                (m - prev.time) / s * 10 + 50
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

          if (!scoreMean.isNaN && !oddsTop._2.isNaN && scoreMean > oddsTop._2) {
            raceCount += 1
            if (oddsTop._1.rank == 1) {
              winCount += 1
            }
          }

          pw.println("%010d".format(raceId.toLong))
          res.foreach(pw.println)
          pw.println
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
    case (0, dist) if dist <= 1600 =>
      CATEGORY_SHIBA_SHORT
    case (0, dist) =>
      CATEGORY_SHIBA_LONG
    case (1, dist) if dist <= 1600 =>
      CATEGORY_DIRT_SHORT
    case (1, dist) =>
      CATEGORY_DIRT_LONG
  }

  def makeRaceType(vector: DenseVector[Double], raceId: Long): Long = {
    val babaCode = (raceId / 1000000) % 100
    babaCode * 100000 + vector(2).toLong * 10000 + vector(4).toLong
  }
}