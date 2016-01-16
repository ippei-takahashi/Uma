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
      (7111600, 40.43283018867924, 2.87800651500473),
      (1111500, 40.26807228915662, 2.8834145495108916),
      (4111200, 39.537425424506655, 2.791342993993305),
      (7111400, 40.13466550825369, 2.906798942535526),
      (9111200, 39.17298616761595, 2.7444268670273266),
      (1111200, 39.678971962616825, 2.7591594942372804),
      (4111400, 40.063964534515515, 2.8867736112845708),
      (8111200, 38.54820528033225, 2.7575048198394843),
      (1011200, 40.54123989218329, 3.017839932520702),
      (6111200, 39.28102746693794, 2.7498278038192185),
      (4111600, 38.77981033191914, 2.831787533169386),
      (7111200, 39.409404536862, 2.7622803607107365),
      (8111400, 39.34103165298945, 2.826860387141197),
      (2111200, 39.84453033497306, 2.824191948998778),
      (9111400, 39.98681891025641, 2.809344565420209),
      (9111600, 39.60175517551755, 2.883547958217206),
      (8111600, 39.898773722627735, 2.9595415058495007),
      (5111400, 39.052792321116925, 2.842939231593943),
      (6111600, 40.092622028414684, 2.8660315702952093),
      (3111200, 39.41567709050834, 2.811212834501862),
      (5111600, 39.57501949571094, 2.8605997910080108),
      (10111200, 39.202678093197646, 2.7890118703877347)
    ),
    CATEGORY_SHIBA_LONG -> List(
      (4112200, 40.520549158547385, 2.873836992569284),
      (3112600, 41.20222129086337, 3.207378842331787),
      (9112200, 40.74215070643642, 2.960916068499911),
      (7111800, 40.443846153846154, 2.7937162484221703),
      (8112400, 40.0195652173913, 3.028594896640252),
      (9112400, 40.50702247191011, 3.1005553166256052),
      (8112200, 40.038409703504044, 2.9735641998292084),
      (2112000, 40.96992084432718, 3.391295243129602),
      (1112000, 40.83878787878788, 2.9044535736357404),
      (6112200, 40.39519230769231, 3.0411055940549243),
      (1111800, 40.10318230852211, 2.924900653695433),
      (2111800, 40.619479857215704, 3.1598643626804495),
      (5112400, 40.211629811629814, 3.0451571826159065),
      (4112000, 39.65715245130961, 3.044786766547214),
      (3112000, 40.551138915624, 3.132653834038861),
      (9112000, 40.39391682184978, 2.9472165352608672),
      (8112000, 39.577532541029996, 2.851264167801283),
      (3111800, 40.176682490924325, 2.9592815349928636),
      (6111800, 40.01566794942276, 2.8652299619601567),
      (4111800, 38.92998903508772, 2.843391464509341),
      (5112000, 39.56244769874477, 2.8972123256025397),
      (7112000, 40.6135977190254, 2.995240666224083),
      (10111800, 39.9246052951178, 2.883596813697459),
      (8111800, 39.497477064220185, 2.913054220710276),
      (6112000, 40.354681837336704, 2.995966778477294),
      (9111800, 39.54297855119385, 2.9406368258345754),
      (5111800, 39.218891820580474, 3.0311707818379503)
    ),
    CATEGORY_DIRT_SHORT -> List(
      (7001000, 40.3384126984127, 2.749155586164876),
      (1001000, 40.01922077922078, 2.74641032308897),
      (7001200, 41.4142567224759, 2.8309842404353125),
      (2001000, 40.370509770603226, 2.797748122178467),
      (7001400, 42.565358931552586, 2.9413277781763405),
      (10001000, 40.22671806167401, 2.738804661159647),
      (5001300, 41.557804551539486, 2.9556111695322884),
      (3001150, 41.596158940397355, 2.7869906684199828),
      (7001700, 43.1836784409257, 2.9241510678698948),
      (2001700, 43.31185819070904, 3.0053816905731576),
      (1001700, 43.01764199655766, 3.090527925581996),
      (3001700, 43.40880866425993, 3.0152142295864164),
      (4001200, 41.68664780382646, 2.8263308448678757),
      (9001200, 41.50784929356358, 2.933682120036034),
      (10001700, 43.27095156555773, 3.0416451165284752),
      (9001400, 42.31486129458388, 2.9875250758989607),
      (8001200, 41.238015683600406, 2.913058482804085),
      (8001400, 42.203379492251894, 2.9539013994639043),
      (5001400, 41.99077679748504, 3.026311487947384),
      (5001600, 42.41015907322556, 3.0745756510123123),
      (6001200, 42.035984550752135, 2.9312240137254046)
    ),
    CATEGORY_DIRT_LONG -> List(
      (9002000, 42.94196078431373, 3.030004986504336),
      (8001900, 43.08690542809178, 3.0363211505618137),
      (5002100, 43.101721978710076, 3.223712037592804),
      (7001800, 43.23174311926606, 3.088219411331349),
      (4001800, 43.40006059807667, 3.1222189748679736),
      (8001800, 42.76884879105016, 3.093810443691929),
      (9001800, 42.989861351819755, 3.136850543987038),
      (6001800, 43.921592224979325, 3.2467853800811173)
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
      case (raceId, arr) =>
        val horses_ = arr.map {
          d =>
            val x = d(3 until data.cols - 1)
            val raceType = makeRaceType(x, raceId.toLong)
            new PredictData(horseId = d(1).toInt, raceDate = d(2).toInt, age = d(3).toInt, rank = d(d.length - 1).toInt,
              odds = d(d.length - 2), oddsFuku = (d(d.length - 2) - 1) / 4 + 1, time = d(d.length - 3),
              time3f = d(d.length - 4), raceType = raceType, isGoodBaba = x(11) + x(12) == 1.0 && x(7) == 1.0,
              prevDataList = Nil)
        }

        val horses = horses_.map {
          horse =>
            horse.copy(time3f = horses_.filter(_.rank <= horse.rank).map(_.time3f).sorted.head)
        }

        println(horses.sortBy(_.rank).head.time3f == horses_.sortBy(_.rank).head.time3f)

        raceId -> horses_
    }.toSeq.sortBy(_._2.head.raceDate)

    val horseMap = array.groupBy(_(1)).map {
      case (horseId, arr) =>
        horseId -> arr.map { d =>
          val raceId = d(0)
          val x = d(3 until data.cols - 1)
          val raceType = makeRaceType(x, raceId.toLong)
          new Data(raceDate = d(2).toInt, age = d(3).toInt, rank = d(d.length - 1).toInt,
            odds = d(d.length - 2).toInt, time = d(d.length - 3), time3f = d(d.length - 4), raceId = raceId.toLong,
            raceType = raceType, isGoodBaba = x(11) + x(12) == 1.0 && x(7) == 1.0)
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

          if (!scoreMean.isNaN && !oddsTop._2.isNaN && scoreMean > oddsTop._2 && prevLengthMean > 8) {
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