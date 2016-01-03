import java.io._

import breeze.linalg._
import breeze.stats._

object Main {
  type Gene = DenseVector[Double]

  case class Data(raceDate: Int, age: Int, rank: Int, odds: Double, time: Double, raceId: Long, raceType: Long, isGoodBaba: Boolean)

  case class PredictData(horseId: Int, raceDate: Int, raceType: Long, rank: Int, odds: Double, oddsFuku: Double, age: Double,
                         isGoodBaba: Boolean, prevDataList: Seq[Data])

  private[this] val ratingMapDShort = scala.collection.mutable.Map[Int, (Double, Int)]()

  private[this] val ratingMapDLong = scala.collection.mutable.Map[Int, (Double, Int)]()

  private[this] val ratingMapSShort = scala.collection.mutable.Map[Int, (Double, Int)]()

  private[this] val ratingMapSLong = scala.collection.mutable.Map[Int, (Double, Int)]()

  private[this] val DEFAULT_RATE = 1500.0

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
            val raceType = makeRaceType(x)
            new Data(raceDate = d(2).toInt, age = d(3).toInt, rank = d(d.length - 1).toInt,
              odds = d(d.length - 2).toInt, time = d(d.length - 3).toInt, raceId = raceId.toLong,
              raceType = raceType, isGoodBaba = x(9) + x(10) == 1.0 && x(5) == 1.0)
          }.toList
          races match {
            case head :: tail =>
              Some(PredictData(
                horseId = horseId.toInt, raceDate = head.raceDate, raceType = head.raceType, rank = head.rank, odds = head.odds,
                oddsFuku = (head.odds - 1) / 5 + 1, age = head.age, isGoodBaba = head.isGoodBaba, prevDataList = tail)
              )
            case _ =>
              None
          }
      }.toArray.collect {
        case Some(x) => x
      }
    }.toSeq.sortBy(_._2.head.raceDate)

    var oddsCount = 0.0
    var raceCount = 0
    var betCount = 0
    var betWinCount = 0

    val dates = for {
      num1 <- 2008 to 2015
      num2 <- 1 to 12
      num3 <- 0 to 1
    } yield {
        num1 * 10000 + num2 * 100 + num3 * 15
      }

    val ranges = for (i <- 0 until (dates.length - 1)) yield {
      (raceDate: Int) =>
        dates(i) < raceDate && raceDate <= dates(i + 1)
    }

    val outFile = new File("result.csv")
    val pw = new PrintWriter(outFile)
    try {
      for {
        ri <- 0 until (ranges.length - 1)
      } {
        raceSeq.filter {
          case (_, arr) =>
            ranges(ri)(arr.head.raceDate)
        }.foreach {
          case (raceId, horses_) =>
            val horses = horses_.sortBy(_.rank)
            val ratingUpdates = horses.map(_ => 0.0)
            val ratingCountUpdates = horses.map(_ => 0)

            val ratingMap = getRatingMap(horses.head.raceType)
            val (ratings, ratingCounts) = horses.map {
              horse =>
                ratingMap.getOrElse(horse.horseId, (DEFAULT_RATE, 0))
            }.unzip

            val k = 32 + Math.min(160.0, ratingCounts.sum) / 10
            for {
              i <- 0 until 3
              j <- (i + 1) until horses.length
            } {
              val e1 = 1.0 / (1.0 + Math.pow(10.0, (ratings(j) - ratings(i)) / 400.0))
              val e2 = 1.0 / (1.0 + Math.pow(10.0, (ratings(i) - ratings(j)) / 400.0))

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

        raceSeq.filter {
          case (_, arr) =>
            ranges(ri + 1)(arr.head.raceDate)
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

            val newRatingInfoTime = ratingInfo.map(
              x => x -> x._1.prevDataList.filter(_.raceType == raceType).map(_.time).sorted.headOption
            ).collect {
              case (x, Some(time)) =>
                x -> time
            }.sortBy(
              _._2
            ).zipWithIndex.map {
              case (((horse, (rating, ratingCount)), _), index) =>
                (horse, rating, ratingCount, index)
            }

            val newRatingInfoScore = newRatingInfo.map {
              case (horse, rating, ratingCount, _) =>
                val indexTime = newRatingInfoTime.find(_._1.horseId == horse.horseId).map(_._4).getOrElse(-1)
                val score = rating +
                  (indexTime match {
                    case 0 => 0
                    case 1 => 0
                    case 2 => 0
                    case 3 => 0
                    case 4 => 0
                    case _ => 0
                  })
                (horse, score, ratingCount)
            }.sortBy(-_._2).zipWithIndex.map {
              case ((horse, rating, ratingCount), index) =>
                (horse, rating, ratingCount, index)
            }

            raceCount += 1

            val sortedScores = newRatingInfoScore.sortBy(-_._2)

            val scoreDiffs = for {
              i <- 1 until sortedScores.length
            } yield sortedScores.head._2 - sortedScores(i)._2
            val predictOdds = scoreDiffs.foldLeft(1.0) {
              (x, y) =>
                x * (1 + Math.pow(10, -y / 400))
            } * 1.5 - 1

            val ratingTop = sortedScores.head

            if (ratingTop._3 > 0 && predictOdds < ratingTop._1.odds) {
              sortedScores.foreach(pw.println)
              pw.println

              betCount += 1
              if (sortedScores.head._1.rank <= 2 || (sortedScores.head._1.rank <= 3 && horses.length >= 8)) {
                betWinCount += 1
                oddsCount += sortedScores.head._1.oddsFuku
              }
            }
        }
      }
    } catch {
      case e:Exception =>
    } finally {
      pw.close
    }

    Seq(
      "ratingDShort.csv" -> ratingMapDShort,
      "ratingDLong.csv" -> ratingMapDLong,
      "ratingSShort.csv" -> ratingMapSShort,
      "ratingSLong.csv" -> ratingMapSLong
    ).foreach {
      case (fileName, ratingMap) =>
        val mat = DenseMatrix(ratingMap.toArray.map {
          case (key, (rating, count)) =>
            (key.toDouble, rating, count.toDouble)
        }.sortBy(_._2): _*)
        csvwrite(new File(fileName), mat)
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
      case (10, dist) if dist <= 1600 =>
        ratingMapDShort
      case (10, _) =>
        ratingMapDLong
      case (11, dist) if dist <= 1600 =>
        ratingMapSShort
      case (11, _) =>
        ratingMapSLong
    }


  def makeRaceType(vector: DenseVector[Double]): Long =
    100000 + vector(2).toLong * 10000 + vector(4).toLong
}