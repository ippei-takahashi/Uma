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
      (511600, 95.50099800399201,1.990869851887232),
      (811600, 95.54538216560509,1.7949165490406667),
      (611600, 95.92786615469007,2.1861284090289956),
      (511400, 82.95085344320188,1.7293116687042962)
    ),
    CATEGORY_SHIBA_LONG -> List(
      (511800, 108.91318124207858,2.559480165892545),
      (911800, 108.31492842535788,2.016384145904715),
      (612000, 122.8134110787172,2.5562905527012774),
      (811800, 108.84785875281743,2.2397944081856873),
      (411800, 108.5249569707401,2.013894107759768)
    ),
    CATEGORY_DIRT_SHORT -> List(
      (501600, 99.90096256684492,2.2720015279892576),
      (501400, 87.20122261844116,2.169095859747181),
      (801200, 73.55891860123421,1.848696000416368),
      (401200, 73.20599471054952,1.6411268647534247)
    ),
    CATEGORY_DIRT_LONG -> List(
      (901800, 115.50652741514361,2.6392406646362865),
      (801800, 115.01623455294403,2.670184758029812),
      (401800, 115.39772727272727,2.3277014532411786)
    )
  )

  private[this] val stdAndTimeMap = Map[Long, List[(Double, Double)]] // TODO: 最小二乗法使って傾きと切片求める

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
          val meanList = stdRaceMap(CATEGORY_DIRT_LONG).map {
            case (raceType, m, s) =>
              horses.find(_.raceType == raceType).map {
                x => (m - x.time) / s * 10 + 50
              }
          }.collect {
            case Some(x) => x
          }
          if (meanList.length > 1) {
            val meanStd =  meanList match {
              case Nil => 50
              case list => mean(list)
            }
            stdRaceMap(CATEGORY_DIRT_LONG).foreach {
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
            (key.toDouble, m)
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
    case (0, dist)=>
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