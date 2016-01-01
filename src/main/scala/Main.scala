import java.io._

import breeze.linalg._

object Main {

  case class Data(x: DenseVector[Double], horseNo: Int, time: Double, raceId: Long, raceType: Long, isGoodBaba: Boolean)

  case class PredictData(horseId: Int, horseNo: Int, raceType: Long, rank: Double, odds: Double, age: Double, isGoodBaba: Boolean, prevDataList: Seq[Data])

  private[this] val ratingMapDShort = scala.collection.mutable.Map[Int, (Double, Int)]()

  private[this] val ratingMapDMiddle = scala.collection.mutable.Map[Int, (Double, Int)]()

  private[this] val ratingMapDLong = scala.collection.mutable.Map[Int, (Double, Int)]()

  private[this] val ratingMapSShort = scala.collection.mutable.Map[Int, (Double, Int)]()

  private[this] val ratingMapSMiddle = scala.collection.mutable.Map[Int, (Double, Int)]()

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
            new Data(x, horseNo = d(3).toInt, time = d(data.cols - 2), raceId.toLong, raceType, isGoodBaba = x(9) + x(10) == 1.0 && x(5) == 1.0)
          }.toList
          races match {
            case head :: tail =>
              Some(PredictData(
                horseId = horseId.toInt, horseNo = head.horseNo, raceType = head.raceType, rank = -1,
                odds = arr2.head(arr2.head.length - 2), age = head.x(0), isGoodBaba = head.isGoodBaba, prevDataList = tail)
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
      "ratingDLong.csv" -> ratingMapDLong,
      "ratingSShort.csv" -> ratingMapSShort,
      "ratingSMiddle.csv" -> ratingMapSMiddle,
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
          val raceType = horses.head.raceType

          val ratingMap = getRatingMap(raceType)
          val ratingInfo = horses.map {
            horse =>
              horse -> ratingMap.getOrElse(horse.horseId, (DEFAULT_RATE, 0))
          }
          val newRatingInfo = ratingInfo.sortBy(-_._2._1).zipWithIndex.map {
            case ((horse, (rating, ratingCount)), index) =>
              (horse, rating, ratingCount, index)
          }

          val newRatingInfoTime = ratingInfo.sortBy(
            _._1.prevDataList.filter(_.raceType == raceType).map(_.time).sorted.headOption.getOrElse(Double.MaxValue)
          ).zipWithIndex.map {
            case ((horse, (rating, ratingCount)), index) =>
              (horse, rating, ratingCount, index)
          }

          val newRatingInfoScore = newRatingInfo.map {
            case (horse, rating, ratingCount, _) =>
              val indexTime = newRatingInfoTime.find(_._1.horseId == horse.horseId).get._4
              val score = rating +
                (indexTime match {
                  case 0 => 20
                  case 1 => 15
                  case 2 => 10
                  case 3 => 5
                  case 4 => 5
                  case _ => 0
                })
              (horse, score, ratingCount)
          }.sortBy(-_._2).zipWithIndex.map {
            case ((horse, rating, ratingCount), index) =>
              (horse, rating, ratingCount, index)
          }


          val sortedScores = newRatingInfoScore.sortBy(-_._2)
          val scoreDiffs = for {
            i <- 1 until sortedScores.length
          } yield sortedScores.head._2 - sortedScores(i)._2
          val predictOdds = scoreDiffs.foldLeft(1.0) {
            (x, y) =>
              x * (1 + Math.pow(10, -y / 400))
          } * 3 - 1

          val ratingTop = sortedScores.head

          if (ratingTop._3 > 0 && predictOdds < ratingTop._1.odds) {
            pw.println("%10d, %f, %10d, %d".format(raceId.toLong, ratingTop._1.odds, ratingTop._1.horseId, ratingTop._1.horseNo))
            println("%10d, %f, %10d, %d".format(raceId.toLong, ratingTop._1.odds, ratingTop._1.horseId, ratingTop._1.horseNo))
            for {
              res <- sortedScores
            } {
              pw.println("%f, %d, %10d, %d".format(res._2, res._3, res._1.horseId, res._1.horseNo))
              println("%f, %d, %10d, %d".format(res._2, res._3, res._1.horseId, res._1.horseNo))
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
      case (10, dist) if dist <= 1600 =>
        ratingMapDMiddle
      case (10, _) =>
        ratingMapDLong
      case (11, dist) if dist <= 1200 =>
        ratingMapSShort
      case (11, dist) if dist <= 1600 =>
        ratingMapSMiddle
      case (11, _) =>
        ratingMapSLong
    }

  def makeRaceType(vector: DenseVector[Double]): Long =
    100000 + vector(2).toLong * 10000 + vector(4).toLong
}