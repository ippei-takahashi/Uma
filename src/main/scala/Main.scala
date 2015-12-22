import java.io._

import breeze.linalg._
import breeze.stats._

object Main {
  type Gene = DenseVector[Double]

  case class Data(x: DenseVector[Double], time: Double, raceId: Long, raceType: Long)

  case class PredictData(horseId: Int, raceType: Long, rank: Int, odds: Double, oddsFuku: Double, age: Double, prevDataList: Seq[Data])

  private[this] val ratingMapDShort = scala.collection.mutable.Map[Int, (Double, Int)]()

  private[this] val ratingMapDMiddle = scala.collection.mutable.Map[Int, (Double, Int)]()

  private[this] val ratingMapDSemiLong = scala.collection.mutable.Map[Int, (Double, Int)]()

  private[this] val ratingMapDLong = scala.collection.mutable.Map[Int, (Double, Int)]()

  private[this] val ratingMapSShort = scala.collection.mutable.Map[Int, (Double, Int)]()

  private[this] val ratingMapSMiddle = scala.collection.mutable.Map[Int, (Double, Int)]()

  private[this] val ratingMapSSemiLong = scala.collection.mutable.Map[Int, (Double, Int)]()

  private[this] val ratingMapSLong = scala.collection.mutable.Map[Int, (Double, Int)]()

  private[this] val DEFAULT_RATE = 1500.0

  def main(args: Array[String]) {

    val dataCSV = new File("data.csv")
    val raceCSV = new File("race.csv")

    val data: DenseMatrix[Double] = csvread(dataCSV)
    val size = data.rows

    val array = Array.ofDim[DenseVector[Double]](size)
    for (i <- 0 until size) {
      array(i) = data(i, ::).t
    }

    val dataMap = array.groupBy(_ (0)).map {
      case (horseId, arr) => horseId -> arr.map { d =>
        val raceId = d(data.cols - 1).toLong
        val x = d(1 until data.cols - 2)
        val raceType = makeRaceType(x)
        new Data(x, d(data.cols - 2), raceId, raceType)
      }.toList
    }

    val race: DenseMatrix[Double] = csvread(raceCSV)
    val raceSize = race.rows

    val raceArray = Array.ofDim[DenseVector[Double]](raceSize)
    for (i <- 0 until raceSize) {
      raceArray(i) = race(i, ::).t
    }

    val raceMap = raceArray.groupBy(_ (0)).map {
      case (raceId, arr) => raceId -> (arr match {
        case validArray if validArray.forall(vec => dataMap.get(vec(2)) match {
          case Some(races) =>
            subListBeforeRaceId(raceId.toLong, races).nonEmpty
          case _ =>
            false
        }) =>
          validArray.map {
            vec =>
              val races = dataMap(vec(2))
              val head :: tail = subListBeforeRaceId(raceId.toLong, races)
              PredictData(horseId = vec(2).toInt, raceType = head.raceType, rank = vec(1).toInt, odds = vec(3), oddsFuku = vec(5),
                age = head.x(0), prevDataList = tail)
          }
        case _ => Array[PredictData]()
      })
    }.filter {
      case (_, arr) =>
        arr.nonEmpty
    }.map {
      case (raceId, arr) =>
        raceId -> arr.sortBy(_.rank)
    }.toSeq.sortBy(_._1)


    for (loop <- 0 until 10) {
      raceMap.foreach {
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
        val mat = DenseMatrix(ratingMap.toArray.map {
          case (key, (rating, count)) =>
            (key.toDouble, rating, count.toDouble)
        }.sortBy(_._2): _*)
        csvwrite(new File(fileName), mat)
    }

    var oddsCount = 0.0
    var raceCount = 0
    var betCount = 0
    var betWinCount = 0

    raceMap.foreach {
      case (raceId, horses) =>
        val ratingMap = getRatingMap(horses.head.raceType)
        val ratingInfo = horses.map {
          horse =>
            horse -> ratingMap.getOrElse(horse.horseId, (DEFAULT_RATE, 0))
        }.sortBy(-_._2._1)

        val ratingTop = ratingInfo.head
        val ratingSecond = ratingInfo(1)

        raceCount += 1
        if (ratingTop._2._2 > 300 && ratingTop._2._1 - ratingSecond._2._1 > 100) {
          betCount += 1
          if (ratingTop._1.rank == 1) {
            betWinCount += 1
            oddsCount += ratingTop._1.odds
          }
        }
    }


    val rtn = oddsCount / betCount
    val p = betWinCount.toDouble / betCount.toDouble
    val r = oddsCount / betWinCount - 1.0
    val kf = ((r + 1) * p - 1) / r
    val g = Math.pow(Math.pow(1 + r * kf, p) * Math.pow(1 - kf, 1 - p), betCount)
    println(raceCount, oddsCount / betWinCount, betCount, betWinCount, betWinCount.toDouble / betCount.toDouble, rtn, kf, g)
  }

  def subListBeforeRaceId(raceId: Long, list: List[Data]): List[Data] = list match {
    case x :: xs if x.raceId == raceId =>
      x :: xs
    case _ :: xs =>
      subListBeforeRaceId(raceId, xs)
    case _ =>
      Nil
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