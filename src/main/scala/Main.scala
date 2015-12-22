import java.io._

import breeze.linalg._

object Main {

  case class Data(x: DenseVector[Double], time: Double, raceId: Long, raceType: Long, isGoodBaba: Boolean)

  case class PredictData(horseId: Int, raceType: Long, rank: Double, odds: Double, age: Double, isGoodBaba: Boolean, prevDataList: Seq[Data])

  private[this] val ratingMapDShort = scala.collection.mutable.Map[Int, (Double, Int)]()

  private[this] val ratingMapDMiddle = scala.collection.mutable.Map[Int, (Double, Int)]()

  private[this] val ratingMapDSemiLong = scala.collection.mutable.Map[Int, (Double, Int)]()

  private[this] val ratingMapDLong = scala.collection.mutable.Map[Int, (Double, Int)]()

  private[this] val ratingMapSShort = scala.collection.mutable.Map[Int, (Double, Int)]()

  private[this] val ratingMapSMiddle = scala.collection.mutable.Map[Int, (Double, Int)]()

  private[this] val ratingMapSSemiLong = scala.collection.mutable.Map[Int, (Double, Int)]()

  private[this] val ratingMapSLong = scala.collection.mutable.Map[Int, (Double, Int)]()

  private[this] val DEFAULT_RATE = 1500.0

  private[this] val raceTypeArray = Array[Long](
    101000,
    111000,
    101150,
    101200,
    111200,
    101300,
    101400,
    111400,
    101500,
    111500,
    101600,
    111600,
    101700,
    111700,
    101800,
    111800,
    101870,
    101900,
    102000,
    112000,
    102100,
    112200,
    102300,
    112300,
    102400,
    112400,
    102500,
    112500,
    112600,
    113000,
    113200,
    113400,
    113600
  )

  def main(args: Array[String]) {
    val dataCSV = new File("data.csv")
    val coefficientCSV = new File("coefficient.csv")
    val raceCSV = new File("race.csv")

    val data: DenseMatrix[Double] = csvread(dataCSV)
    val size = data.rows

    val array = Array.ofDim[DenseVector[Double]](size)
    for (i <- 0 until size) {
      array(i) = data(i, ::).t
    }

    val race: DenseMatrix[Double] = csvread(raceCSV)
    val raceSize = race.rows

    val raceArray = Array.ofDim[DenseVector[Double]](raceSize)
    for (i <- 0 until raceSize) {
      raceArray(i) = race(i, ::).t
    }

    val raceMap = array.groupBy(_ (0)).map {
      case (raceId, arr) => raceId -> arr.groupBy(_ (1)).map {
        case (horseId, arr2) =>
          val races = arr2.map { d =>
            val x = d(2 until data.cols - 2)
            val raceType = makeRaceType(x)
            new Data(x, d(data.cols - 2), raceId.toLong, raceType, isGoodBaba = x(4) + x(5) == 1.0 && x(8) == 1.0)
          }.toList
          races match {
            case head :: tail =>
              Some(PredictData(
                horseId = horseId.toInt, raceType = head.raceType, rank = -1, odds = arr2.head(arr2.head.length - 1),
                age = head.x(0), isGoodBaba = head.isGoodBaba, prevDataList = tail)
              )
            case _ =>
              None
          }
      }.toArray.collect {
        case Some(x) => x
      }
    }

    Seq(
      "ratingDShort.csv" -> ratingMapDShort,
      "ratingDMiddle.csv" -> ratingMapDMiddle,
      "ratingDSemiLong.csv" -> ratingMapDSemiLong,
      "ratingDLong.csv" -> ratingMapDLong,
      "ratingSShort.csv" -> ratingMapSShort,
      "ratingSMiddle.csv" -> ratingMapSMiddle,
      "ratingSSemiLong.csv" -> ratingMapSSemiLong,
      "ratingSLong.csv" -> ratingMapSLong
    ).foreach {
      case (fileName, ratingMap) =>
        val mat = csvread(new File(fileName))
        for {
          i <- 0 until mat.rows
        } {
          val vec = mat(i, ::).t
          ratingMap.put(vec(0).toInt, (vec(1), vec(2).toInt))
        }
    }

    val outFile = new File("result.csv")
    val pw = new PrintWriter(outFile)

    try {
      raceMap.filter {
        case (raceId, seq) => seq.length > 0
      }.foreach {
        case (raceId, horses) =>
          val ratingMap = getRatingMap(horses.head.raceType)
          val ratingInfo = horses.map {
            horse =>
              horse -> ratingMap.getOrElse(horse.horseId, (DEFAULT_RATE, 0))
          }.sortBy(-_._2._1)

          val ratingTop = ratingInfo.head
          val ratingSecond = ratingInfo(1)

          if (ratingTop._2._2 > 300 && ratingTop._2._1 - ratingSecond._2._1 > 0) {
            pw.println("%10d, %f, %10d".format(raceId.toLong, ratingTop._1.odds, ratingTop._1.horseId))
            println("%10d, %f, %10d".format(raceId.toLong, ratingTop._1.odds, ratingTop._1.horseId))
            for {
              res <- ratingInfo
            } {
              pw.println("%f, %d, %10d".format(res._2._1, res._2._2, res._1.horseId))
              println("%f, %d, %10d".format(res._2._1, res._2._2, res._1.horseId))
            }
          }
      }
    } catch {
      case ex: Exception =>
        ex.printStackTrace()
    } finally {
      pw.close()
    }
  }

  def getRatingMap(raceType: Long): scala.collection.mutable.Map[Int, (Double, Int)] =
    (raceType / 10000, raceType % 10000) match {
      case (10, dist) if dist <= 1200 =>
        ratingMapDShort
      case (10, dist) if dist <= 1500 =>
        ratingMapDMiddle
      case (10, dist) if dist <= 1800 =>
        ratingMapDSemiLong
      case (10, _) =>
        ratingMapDLong
      case (11, dist) if dist <= 1200 =>
        ratingMapSShort
      case (11, dist) if dist <= 1500 =>
        ratingMapSMiddle
      case (11, dist) if dist <= 1800 =>
        ratingMapSSemiLong
      case (11, _) =>
        ratingMapSLong
    }

  def makeRaceType(vector: DenseVector[Double]): Long =
    100000 + vector(1).toLong * 10000 + vector(3).toLong
}