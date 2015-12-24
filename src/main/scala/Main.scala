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

    val num1Range = 100000000
    val num2Range = 10000
    val num3Range = 100

    val ranges = for {
      num1 <- 500000000 to 1400000000 by num1Range
      num2 <- 10000 to 50000 by num2Range
      num3 <- 100 to 800 by num3Range
    } yield {
        (raceId: Double) =>
          raceId >= num1 &&
            raceId < num1 + num1Range &&
            (raceId % (num2Range * 10)) >= num2 &&
            (raceId % (num2Range * 10)) < num2 + num2Range &&
            (raceId % (num3Range * 10)) >= num3 &&
            (raceId % (num3Range * 10)) < num3 + num3Range
      }

    for {
      ri <- 0 until (ranges.length - 1)
    } {
      for (loop <- 0 until 10) {
        raceMap.filter {
          case (raceId, _) =>
            ranges(ri)(raceId)
        }.foreach {
          case (raceId, horses) =>
            val ratingUpdates = horses.map(_ => 0.0)
            val ratingCountUpdates = horses.map(_ => 0)

            val ratingMap = getRatingMap(horses.head.raceType)
            val (ratings, ratingCounts) = horses.map {
              horse =>
                ratingMap.getOrElse(horse.horseId, (DEFAULT_RATE, 0))
            }.unzip

            for {
              i <- 0 until 3
              j <- (i + 1) until horses.length
            } {
              val e1 = 1.0 / (1.0 + Math.pow(10.0, (ratings(j) - ratings(i)) / 400.0))
              val e2 = 1.0 / (1.0 + Math.pow(10.0, (ratings(i) - ratings(j)) / 400.0))
              val k = 16

              ratingUpdates(i) += k * (1.0 - e1)
              ratingUpdates(j) -= k * e2

              ratingCountUpdates(i) += 1
              ratingCountUpdates(j) += 1
            }

            horses.zipWithIndex.foreach {
              case (horse, index) =>
                ratingMap.put(horse.horseId, (ratings(index) + ratingUpdates(index), ratingCounts(index) + ratingCountUpdates(index)))
            }
        }
      }

      raceMap.filter {
        case (raceId, arr) =>
          ranges(ri + 1)(raceId)
      }.foreach {
        case (raceId, horses) =>

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
              val score = rating + (indexTime match {
                case 0 => 400
                case 1 => 350
                case 2 => 250
                case 3 => 200
                case 4 => 200
                case _ => 0
              }) + horse.prevDataList.take(10).map {
                prevData =>
                  prevData.rank match {
                    case Some(1) if prevData.raceType == raceType =>
                      50
                    case Some(1) =>
                      20
                    case Some(n) if n <= 3 && prevData.raceType == raceType =>
                      30
                    case Some(n) if n <= 3 =>
                      10
                    case Some(_) if prevData.raceType == raceType =>
                      -10
                    case Some(_) =>
                      -3
                    case _ =>
                      0
                  }
              }.sum
              (horse, score, ratingCount)
          }.sortBy(-_._2).zipWithIndex.map {
            case ((horse, rating, ratingCount), index) =>
              (horse, rating, ratingCount, index)
          }

          val sortedScores = newRatingInfoScore.sortBy(-_._2)
          val scoreDiff = sortedScores.head._2 - sortedScores(1)._2
          val scoreDiff2 = sortedScores.head._2 - sortedScores(2)._2
          val scoreDiff3 = sortedScores.head._2 - sortedScores(3)._2

          val predictOdds = (1 + Math.pow(10, -scoreDiff / 400)) *
            (1 + Math.pow(10, -scoreDiff2 / 400)) *
            (1 + Math.pow(10, -scoreDiff3 / 400)) *
            6.5

          val ratingTop = sortedScores.head

          if (ratingTop._3 > 0 && predictOdds < ratingTop._1.odds) {
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