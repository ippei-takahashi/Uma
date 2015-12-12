import java.io._
import scala.collection.parallel.immutable.ParSeq
import scala.util.Random
import breeze.linalg._
import breeze.numerics._
import breeze.stats._

object Main {

  case class Data(x: DenseVector[Double], time: Double, raceId: Long, raceType: Long)

  case class PredictData(horseId: Double, raceType: Long, rank: Double, odds: Double, prevDataList: Seq[Data])

  case class CompetitionData(raceType: Long, horseData1: CompetitionHorseData, horseData2: CompetitionHorseData)

  case class CompetitionHorseData(no: Int, time: Double)

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

    val dataMap = array.groupBy(_ (0)).map {
      case (horseId, arr) => horseId -> arr.map { d =>
        val raceId = d(data.cols - 1).toLong
        val x = d(1 until data.cols - 2)
        val raceType = makeRaceType(x)
        new Data(x, d(data.cols - 2), raceId, raceType)
      }.toList.filter {
        case Data(x, _, _, _) =>
          x(4) == 1.0
      }
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
          val races = arr2.filter(x => x(10) == 1.0).map { d =>
            val x = d(2 until data.cols - 2)
            val raceType = makeRaceType(x)
            new Data(x, d(data.cols - 2), d(0).toLong, raceType)
          }.toList
          val subList = subListBeforeRaceId(raceId.toLong, races)
          subList match {
            case head :: tail =>
              Some(PredictData(
                horseId = horseId, raceType = head.raceType, rank = -1, odds = arr2.head(arr2.head.length - 1), prevDataList = tail)
              )
            case _ =>
              None
          }
      }.toArray.collect {
        case Some(x) => x
      }
    }

    val outFile = new File("result.csv")
    val pw = new PrintWriter(outFile)

    try {
      raceMap.filter {
        case (raceId, seq) => seq.length > 0
      }.foreach {
        case (raceId, horses) =>

          val allCompetitions = makeAllCompetitions(horses)
          val ratings = horses.map(_ => DEFAULT_RATE)
          val raceType = horses.head.raceType

          val sortedCompetitions = allCompetitions.sortBy(competitionData => -Math.abs(competitionData.raceType - raceType))

          sortedCompetitions.groupBy(_.raceType).foreach {
            case (thisRaceType, seq) =>
              val ratingUpdates = horses.map(_ => 0.0)
              seq.foreach {
                case CompetitionData(_, CompetitionHorseData(no1, time1), CompetitionHorseData(no2, time2)) =>
                  val e1 = 1.0 / (1.0 + Math.pow(10.0, (ratings(no2) - ratings(no1)) / 400.0))
                  val e2 = 1.0 / (1.0 + Math.pow(10.0, (ratings(no1) - ratings(no2)) / 400.0))
                  val k = Math.max(4, 16 - Math.abs(thisRaceType - raceType) / 50)
                  //val k = 16
                  if (time1 < time2) {
                    ratingUpdates(no1) += k * (1.0 - e1)
                    ratingUpdates(no2) -= k * e2
                  } else if (time1 > time2) {
                    ratingUpdates(no1) -= k * e1
                    ratingUpdates(no2) += k * (1.0 - e2)
                  }
                case _ =>
              }
              ratings.indices.foreach {
                case index =>
                  ratings(index) += ratingUpdates(index)
              }
          }

          val sortedRatings = ratings.sortBy(-_)

          val m = mean(sortedRatings)
          val s = stddev(sortedRatings)

          val head = sortedRatings.head
          val second = sortedRatings(1)

          val ratingTopIndex = ratings.zipWithIndex.maxBy(_._1)._2
          val ratingSecondIndex = ratings.zipWithIndex.filter(_._2 != ratingTopIndex).maxBy(_._1)._2
          val oddsTopIndex = horses.zipWithIndex.minBy(_._1.odds)._2
          val ratingTop = horses(ratingTopIndex)

          val directWin = allCompetitions.map {
            case CompetitionData(_, CompetitionHorseData(no1, time1), CompetitionHorseData(no2, time2)) =>
              if (no1 == ratingTopIndex && no2 == ratingSecondIndex)
                if (time1 < time2) 1 else -1
              else if (no1 == ratingSecondIndex && no2 == ratingTopIndex)
                if (time1 < time2) 1 else -1
              else
                0
          }.sum

          val directWinToOdds = allCompetitions.map {
            case CompetitionData(_, CompetitionHorseData(no1, time1), CompetitionHorseData(no2, time2)) =>
              if (no1 == ratingTopIndex && no2 == oddsTopIndex)
                if (time1 < time2) 1 else -1
              else if (no1 == oddsTopIndex && no2 == ratingTopIndex)
                if (time1 < time2) 1 else -1
              else
                0
          }.sum

          if (allCompetitions.length > 50 && ratingTop.odds > 1.2 && directWin > 0 && directWinToOdds > 0) {
            println("%10d, %f, %10d".format(raceId.toLong, ratingTop.odds, ratingTop.horseId.toLong))
            for {
              res <- ratings.zip(horses).sortBy(_._1)
            } {
              println("%f, %10d".format(res._1, res._2.horseId.toLong))
            }
            pw.println("%10d, %f, %10d".format(raceId.toLong, ratingTop.odds, ratingTop.horseId.toLong))
          }
      }
    } catch {
      case ex: Exception =>
        ex.printStackTrace()
    } finally {
      pw.close()
    }
  }


  def makeAllCompetitions(horses: Array[PredictData]): Array[CompetitionData] =
    for {
      raceType <- raceTypeArray
      i <- 0 until (horses.length - 1)
      time1 <- horses(i).prevDataList.filter(_.raceType == raceType).map(_.time).sorted.headOption.toSeq
      j <- (i + 1) until horses.length
      time2 <- horses(j).prevDataList.filter(_.raceType == raceType).map(_.time).sorted.headOption.toSeq
    } yield {
      val horseData1 = CompetitionHorseData(i, time1)
      val horseData2 = CompetitionHorseData(j, time2)
      CompetitionData(raceType, horseData1, horseData2)
    }


  def subListBeforeRaceId(raceId: Long, list: List[Data]): List[Data] = list match {
    case x :: xs if x.raceId == raceId =>
      x :: xs
    case _ :: xs =>
      subListBeforeRaceId(raceId, xs)
    case _ =>
      Nil
  }

  def makeRaceType(vector: DenseVector[Double]): Long =
    100000 + vector(1).toLong * 10000 + vector(3).toLong
}