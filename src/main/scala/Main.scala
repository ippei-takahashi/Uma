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

  val test = List(
    (111200, 70.24547511312217, 1.3971966032000496),
    (411000, 60.18412348401323, 2.8856208896101783),
    (312000, 121.59013698630137, 2.08316961625407),
    (611200, 69.36627282491943, 1.4908284778804677),
    (301150, 70.01671018276762, 1.7592070495340544),
    (701700, 108.05888483585201, 2.0775308715357026),
    (501300, 80.65885416666667, 2.0170649611275193),
    (412000, 121.53623922959959, 2.365516605499596),
    (912000, 122.81569620253164, 2.5205250069980436),
    (211200, 70.33542168674698, 1.2819770069306973),
    (611800, 110.06941120153184, 2.255786023156695),
    (201700, 107.91068883610451, 2.158105410873806),
    (411400, 82.77350427350427, 1.7879195487516562),
    (812000, 122.01967213114754, 2.4741563717498885),
    (1012000, 121.45686456400742, 2.0615796439905787),
    (512000, 121.99173174092789, 2.421929071629471),
    (711200, 69.73351648351648, 1.3274829531954615),
    (311800, 109.36940966010734, 2.059611355440442),
    (712000, 122.99738675958189, 2.3328649689182326),
    (411800, 108.5249569707401, 2.013894107759768),
    (811400, 82.26907131011609, 1.4066224714402877),
    (1011800, 108.84453441295547, 2.0121442969528274),
    (811800, 108.84785875281743, 2.2397944081856873),
    (411600, 95.67542213883678, 1.8411218824595075),
    (101700, 107.92916041979011, 2.3173115895639134),
    (612000, 122.8134110787172, 2.5562905527012774),
    (911400, 82.61107117181884, 1.5659055058288736),
    (901200, 73.71911298838437, 1.8006114146185128),
    (911800, 108.31492842535788, 2.016384145904715),
    (1001700, 107.85361093038387, 2.2782467753163655),
    (511800, 108.91318124207858, 2.559480165892545),
    (301700, 108.30663758180118, 2.1285282151499425),
    (911600, 95.83441860465116, 1.8418874785189014),
    (901400, 86.19107896323086, 1.9826543523189482),
    (401800, 115.39772727272727, 2.3277014532411786),
    (511400, 82.95085344320188, 1.7293116687042962),
    (401200, 73.20599471054952, 1.6411268647534247),
    (801200, 73.55891860123421, 1.848696000416368),
    (611600, 95.92786615469007, 2.1861284090289956),
    (311200, 69.88586216289839, 1.487017042816897),
    (801400, 86.28286852589642, 2.023490175911063),
    (811600, 95.54538216560509, 1.7949165490406667),
    (501400, 87.20122261844116, 2.169095859747181),
    (511600, 95.50099800399201, 1.990869851887232),
    (801800, 115.01623455294403, 2.670184758029812),
    (901800, 115.50652741514361, 2.6392406646362865),
    (1011200, 69.43107067879636, 1.4658486992668793),
    (501600, 99.90096256684492, 2.2720015279892576),
    (601200, 73.61419403548508, 1.8870009612147394),
    (601800.0, 117.52284644194756, 2.960570401032924)
  )

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
      (501600, 99.90096256684492, 2.2720015279892576),
      (501400, 87.20122261844116, 2.169095859747181),
      (801200, 73.55891860123421, 1.848696000416368),
      (401200, 73.20599471054952, 1.6411268647534247)
    ),
    CATEGORY_DIRT_LONG -> List(
      (901800, 115.50652741514361, 2.6392406646362865),
      (801800, 115.01623455294403, 2.670184758029812),
      (401800, 115.39772727272727, 2.3277014532411786)
    )
  )

  private[this] val stdAndTimeMap = Map[Long, List[(Double, Double)]]() // TODO: 最小二乗法使って傾きと切片求める

  def main(args: Array[String]) {

    val dataCSV = new File("past.csv")

    val data: DenseMatrix[Double] = csvread(dataCSV)
    val size = data.rows

    val array = Array.ofDim[DenseVector[Double]](size)
    for (i <- 0 until size) {
      array(i) = data(i, ::).t
    }

    val raceSeq = array.groupBy(_ (0)).map {
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
            case head :: tail =>
              Some(PredictData(
                horseId = horseId.toInt, raceDate = head.raceDate, raceType = head.raceType, rank = head.rank, odds = head.odds,
                oddsFuku = (head.odds - 1) / 4 + 1, time = head.time, age = head.age, isGoodBaba = head.isGoodBaba, prevDataList = tail)
              )
            case _ =>
              None
          }
      }.toArray.collect {
        case Some(x) => x
      }
    }.toSeq.sortBy(_._2.head.raceDate)

    val horseSeq = raceSeq.flatMap(_._2.toSeq).groupBy(_.horseId).map {
      case (horseId, seq) =>
        seq.filter(_.isGoodBaba).groupBy(_.raceType).map {
          case (_, times) =>
            times.sortBy(_.time).head
        }.toSeq
    }

    val outFile = new File("result.csv")
    val pw = new PrintWriter(outFile)
    try {
      horseSeq.foreach {
        horses =>
          val meanList = stdRaceMap(CATEGORY_SHIBA_LONG).map {
            case (raceType, m, s) =>
              horses.find(_.raceType == raceType).map {
                x => (m - x.time) / s * 10 + 50
              }
          }.collect {
            case Some(x) => x
          }
          if (meanList.length > 1) {
            val meanStd = meanList match {
              case Nil => 50
              case list => mean(list)
            }
            stdRaceMap(CATEGORY_SHIBA_LONG).foreach {
              case (raceType, m, s) =>
                horses.find(_.raceType == raceType).foreach {
                  x =>
                    val std = (m - x.time) / s * 10 + 50
                    val list = raceTimeMap.getOrElse(raceType, Nil)
                    raceTimeMap.put(raceType, (std - meanStd) :: list)
                }
            }
          }
      }
    } catch {
      case e: Exception =>
    } finally {
      pw.close
    }

    Seq(
      "hensachi.csv" -> raceTimeMap
    ).foreach {
      case (fileName, map) =>
        val mat = DenseMatrix(map.toArray.map {
          case (key, list) =>
            val m = mean(list)
            (key.toDouble, m, list.length.toDouble)
        }.sortBy(_._2): _*)
        csvwrite(new File(fileName), mat)
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
    case (0, dist) if dist <= 1600 =>
      CATEGORY_SHIBA_SHORT
    case (0, dist) =>
      CATEGORY_SHIBA_LONG
    case (1, dist) if dist <= 1600 =>
      CATEGORY_SHIBA_SHORT
    case (1, dist) =>
      CATEGORY_SHIBA_LONG
  }

  def makeRaceType(vector: DenseVector[Double], raceId: Long): Long = {
    val babaCode = (raceId / 1000000) % 100
    babaCode * 100000 + vector(2).toLong * 10000 + vector(4).toLong
  }
}