import java.io._
import scala.collection.parallel.immutable.ParSeq
import scala.util.Random
import breeze.linalg._
import breeze.numerics._
import breeze.stats._

object Main {
  type Gene = DenseMatrix[Double]

  private[this] val NUM_OF_DISTANCE_TYPE = 4
  private[this] val NUM_OF_RACE_TYPE = 23

  private[this] val NUM_OF_LEARNING_LOOP = 200000
  private[this] val TRAIN_RATE = 0.8
  private[this] val LEARNING_RATE = 0.00001
  private[this] val MOMENTUM_RATE = 0.9
  private[this] val BATCH_SIZE = 30

  case class Data(x: DenseVector[Double], y: Double)

  case class PredictData(umaId: Double, data: Data, std: DenseVector[Double])

  def main(args: Array[String]) {
    val r = new Random()

    val dataCSV = new File("data.csv")
    val stdCSV = new File("std.csv")

    val data: DenseMatrix[Double] = csvread(dataCSV)
    val size = data.rows

    val array = Array.ofDim[DenseVector[Double]](size)
    for (i <- 0 until size) {
      array(i) = data(i, ::).t
    }

    val stdData: DenseMatrix[Double] = csvread(stdCSV)
    val stdSize = stdData.rows

    val stdArray = Array.ofDim[DenseVector[Double]](stdSize)
    for (i <- 0 until stdSize) {
      stdArray(i) = stdData(i, ::).t
    }
    val stdMap = stdArray.zipWithIndex.groupBy(_._1(0)).map {
      case (id, arr) =>
        id ->(arr.head._1(1), arr.head._2)
    }

    val allData = array.groupBy(_ (0)).map {
      case (id, arr) => id -> arr.map { d =>
        new Data(d(1 until data.cols - 2), d(data.cols - 2))
      }.toList.filter {
        case Data(x, _) =>
          (x(4) == 1.0 || x(5) == 1.0) && stdMap.get(makeRaceIdSoft(x)).isDefined
      }
    }.filter {
      case (_, list) =>
        list.nonEmpty
    }

    val timeMap: Map[Double, (Double, Double)] = allData.values.flatten.groupBy {
      case Data(x, _) =>
        makeRaceIdSoft(x)
    }.map {
      case (idx, arr) =>
        val times = arr.map(_.y)
        idx ->(mean(times), stddev(times))
    }


    val outFile = new File("coefficient.csv")


    val stdList = allData.map {
      case (umaId, dataList) =>
        val (head :: tail) = dataList

        val dataListShort = tail.filter {
          case Data(x, _) =>
            isShort(x)
        }

        val dataListMiddle = tail.filter {
          case Data(x, _) =>
            isMiddle(x)
        }

        val dataListSemiLong = tail.filter {
          case Data(x, _) =>
            isSemiLong(x)
        }

        val dataListLong = tail.filter {
          case Data(x, _) =>
            isLong(x)
        }

        val (scoresShort, _, _) =
          calcDataListCost(timeMap, dataListShort, (x, y, z) => {
            val std = stdMap.get(makeRaceIdSoft(x)).map(_._1)
            if (std.isEmpty)
              (0, 0.0)
            else
              (1, Math.abs(y - z) / std.get)
          })

        val (scoresMiddle, _, _) =
          calcDataListCost(timeMap, dataListMiddle, (x, y, z) => {
            val std = stdMap.get(makeRaceIdSoft(x)).map(_._1)
            if (std.isEmpty)
              (0, 0.0)
            else
              (1, Math.abs(y - z) / std.get)
          })

        val (scoresSemiLong, _, _) =
          calcDataListCost(timeMap, dataListSemiLong, (x, y, z) => {
            val std = stdMap.get(makeRaceIdSoft(x)).map(_._1)
            if (std.isEmpty)
              (0, 0.0)
            else
              (1, Math.abs(y - z) / std.get)
          })

        val (scoresLong, _, _) =
          calcDataListCost(timeMap, dataListLong, (x, y, z) => {
            val std = stdMap.get(makeRaceIdSoft(x)).map(_._1)
            if (std.isEmpty)
              (0, 0.0)
            else
              (1, Math.abs(y - z) / std.get)
          })

        val stdShort = calcStd(scoresShort, head)
        val stdMiddle = calcStd(scoresMiddle, head)
        val stdSemiLong = calcStd(scoresSemiLong, head)
        val stdLong = calcStd(scoresLong, head)

        PredictData(umaId = umaId, data = head, DenseVector(stdShort, stdMiddle, stdSemiLong, stdLong))
    }.toList

    val coefficient: Gene = DenseMatrix.rand[Double](NUM_OF_DISTANCE_TYPE, NUM_OF_RACE_TYPE)
    val momentum: Gene = DenseMatrix.zeros[Double](NUM_OF_DISTANCE_TYPE, NUM_OF_RACE_TYPE)

    val trainSize = (TRAIN_RATE * stdList.length).toInt
    val shuffledStdList = Random.shuffle(stdList)
    val trainData = shuffledStdList.slice(0, trainSize)
    val testData = shuffledStdList.slice(trainSize, stdList.length)

    var shuffledTrainData = Random.shuffle(trainData)
    var index = 0

    for {
      i <- 0 until NUM_OF_LEARNING_LOOP
    } {
      if (index + BATCH_SIZE > trainData.length) {
        index = 0
        shuffledTrainData = Random.shuffle(trainData)
      }

      val currentData = shuffledTrainData.slice(index, index + BATCH_SIZE)

      val grad = currentData.map { predictData =>
        println(calcGrad(predictData, coefficient, stdMap))

        //        calcGrad(
        //          predictData,
        //          coefficient
        //        )._2
        //      }.reduce { (grad1, grad2) =>
        //        grad1.zip(grad2).map {
        //          case (x, y) => x + y
        //        }
      }
    }
  }

  def findNearest(timeMap: Map[Double, (Double, Double)], vector: DenseVector[Double]): (Double, Double) = {
    val raceId = makeRaceIdSoft(vector)
    timeMap.minBy {
      case (idx, value) =>
        Math.abs(raceId - idx)
    }._2
  }

  def calcStdScore(timeMap: Map[Double, (Double, Double)], d: Data): Double = {
    val (m, s) = timeMap.getOrElse(makeRaceIdSoft(d.x), findNearest(timeMap, d.x))
    if (s == 0.0) {
      0
    } else {
      (d.y - m) / s
    }
  }

  def calcStd(prevScores: List[(Double, DenseVector[Double])],
              d: Data): Double = {
    val p = prevScores.foldLeft((0.0, 0.0)) {
      case ((scores, weights), (s, vector)) =>
        val distInv = 1.0 / vectorDistance(d.x, vector)
        (scores + s * distInv, weights + distInv)
    }
    if (prevScores.isEmpty) 0.0 else p._1 / p._2
  }


  def prePredict(timeMap: Map[Double, (Double, Double)], stdScore: Double, vector: DenseVector[Double]): Double = {
    val (m, s) = timeMap.getOrElse(makeRaceIdSoft(vector), findNearest(timeMap, vector))
    stdScore * s + m
  }

  def predict(prevScores: List[(Double, DenseVector[Double])],
              timeMap: Map[Double, (Double, Double)],
              d: Data): (List[(Double, DenseVector[Double])], Double) = {
    val score = (calcStdScore(timeMap, d), d.x) :: prevScores
    val p = prevScores.foldLeft((0.0, 0.0)) {
      case ((scores, weights), (s, vector)) =>
        val distInv = 1.0 / vectorDistance(d.x, vector)
        (scores + s * distInv, weights + distInv)
    }
    val out = prePredict(timeMap, if (prevScores.isEmpty) 0.0 else p._1 / p._2, d.x)

    (score, out)
  }

  def calcGrad(predictData: PredictData,
               coefficient: DenseMatrix[Double],
               stdMap: Map[Double, (Double, Int)],
               timeMap: Map[Double, (Double, Double)]): Double = {
    val filter: DenseVector[Double] = makeFilter(predictData.data.x, stdMap)
    val stdScore: Double = (predictData.std.t * coefficient) * filter
    val (m, s) = timeMap.getOrElse(makeRaceIdSoft(predictData.data.x), findNearest(timeMap, predictData.data.x))
    val out = stdScore * s + m
      out
  }

  def calcDataListCost(timeMap: Map[Double, (Double, Double)],
                       dataList: List[Data],
                       costFunction: (DenseVector[Double], Double, Double) => (Int, Double)) = {
    dataList.foldLeft((Nil: List[(Double, DenseVector[Double])], 0, 0.0)) {
      case ((prevScores, prevCount, prevCost), d) =>
        val (scores, out) = predict(prevScores, timeMap, d)
        val (count, cost) = costFunction(d.x, d.y, out)

        val newCount = prevCount + count
        val newCost = prevCost + cost

        val newScores = if (count > 0) scores else prevScores

        (newScores, newCount, newCost)
    }
  }

  def makeFilter(vector: DenseVector[Double], stdMap: Map[Double, (Double, Int)]): DenseVector[Double] = {
    val filter = DenseVector.zeros[Double](NUM_OF_RACE_TYPE)
    filter(stdMap.get(makeRaceIdSoft(vector)).map(_._2).get) = 1.0
    filter
  }


  def isShort(vector: DenseVector[Double]) =
    vector(3) <= 1400

  def isMiddle(vector: DenseVector[Double]) =
    vector(3) > 1400 && vector(3) <= 1600

  def isSemiLong(vector: DenseVector[Double]) =
    vector(3) > 1600 && vector(3) <= 2200

  def isLong(vector: DenseVector[Double]) =
    vector(3) > 2200


  def vectorDistance(
                      vector1: DenseVector[Double],
                      vector2: DenseVector[Double]): Double = {
    100 + Math.abs(vector1(0) - vector2(0)) * 250
  }

  def makeRaceIdSoft(vector: DenseVector[Double]): Double =
    vector(3) * 1000 + vector(1) * 100
}