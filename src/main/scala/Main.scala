import java.io._
import scala.util.Random
import breeze.linalg._
import breeze.numerics._
import breeze.stats._

object Main {
  val NUM_OF_LOOP = 100001

  val LEARNING_RATE = 0.01

  val DATA_RATE = 0.8

  val LAMBDA = 0.003

  val BATCH_SIZE = 30

  val NETWORK_SHAPE = Array(1 -> 15, 1 -> 6, 1 -> 6, 1 -> 15, 1 -> 6, 1 -> 6, 6 -> 15, 6 -> 6, 1 -> 15, 1 -> 6, 1 -> 6, 1 -> 7)
  val NUM_OF_MAT = NETWORK_SHAPE.length


  class Data(val x: DenseVector[Double], val y: Double)

  def main(args: Array[String]) {

    val r = new Random()

    val dataCSV = new File("data.csv")

    val data: DenseMatrix[Double] = csvread(dataCSV)
    val dataT = data(::, 1 until data.cols).t
    val dataMean: DenseVector[Double] = mean(dataT(*, ::))
    val dataStd: DenseVector[Double] = stddev(dataT(*, ::))
    val dataNorm: DenseMatrix[Double] = dataT.copy
    val size = data.rows
    for (i <- 0 until size) {
      dataNorm(::, i) := (dataNorm(::, i) :- dataMean) :/ dataStd
    }

    val newData = DenseMatrix.horzcat(data(::, 0).toDenseMatrix.t, dataNorm.t)

    val yMean = dataMean(dataT.rows - 1)
    val yStd = dataStd(dataT.rows - 1)

    val array = Array.ofDim[DenseVector[Double]](size)
    for (i <- 0 until size) {
      array(i) = newData(i, ::).t
    }

    val group = Random.shuffle(array.groupBy(_(0)).values.toList.map(_.map { d =>
      new Data(DenseVector.vertcat(DenseVector.ones[Double](1), d(1 until data.cols - 16)), d(data.cols - 1))
    }.toList)).par

    val groupSize = group.length
    val trainSize = (DATA_RATE * groupSize).toInt
    val testSize = groupSize - trainSize

    val trainData = group.slice(0, trainSize)
    val testData = group.slice(trainSize, groupSize)

    val theta = Array.ofDim[DenseMatrix[Double]](NUM_OF_MAT)

    for (i <- 0 until NUM_OF_MAT) {
      theta(i) = DenseMatrix.rand[Double](NETWORK_SHAPE(i)._1, NETWORK_SHAPE(i)._2)
    }

    for (i <- 0 until NUM_OF_LOOP) {
      val currentData = Random.shuffle(trainData.toList).slice(0, BATCH_SIZE).par

      var costAndGrad = currentData.map { dataArray =>
        dataArray.foldLeft((DenseVector.zeros[Double](NETWORK_SHAPE(0)._1), DenseVector.zeros[Double](NETWORK_SHAPE(0)._1), 0.0)) {
          case ((zBefore, sBefore, _), d) =>
            val mu = theta(0) * d.x + theta(2) * zBefore

            val gFt = sigmoid(theta(3) * d.x + theta(4) * zBefore + sBefore)
            val gIt = sigmoid(theta(5) * d.x + theta(6) * zBefore + sBefore)

            val s = gFt :* sBefore + gIt :* sigmoid(mu)
            val gOt = sigmoid(theta(7) * d.x + theta(8) * zBefore + s)

            val z2 = gOt :* sigmoid(s)
            val a2 = DenseVector.vertcat(DenseVector.ones[Double](1), z2)
            val z3 = theta(1) * a2

            val hx = z3(0)

            var cost = Math.pow((hx - d.y) * yStd, 2.0) / 2.0

            for (j <- 0 until NUM_OF_MAT) {
              cost += LAMBDA / 2 * sum(theta(j)(::, 1 until NETWORK_SHAPE(j)._2) :^ 2.0)
            }

            (z2, s, cost)
        }._3 / BATCH_SIZE
      }.sum



      //      if (i % 100 == 0) {
      //        println(s"LOOP$i: cost = ${cost}")
      //      }
      //      if (i % 500 == 0) {
      //        val errors = testData.map { dataArray =>
      //          dataArray.foldLeft((DenseVector.zeros[Double](NETWORK_SHAPE(0)._1), DenseVector.zeros[Double](NETWORK_SHAPE(0)._1), 0.0)) {
      //            case ((zBefore, sBefore, _), d) =>
      //              val mu = elites.head(0) * d.x + elites.head(2) * zBefore
      //
      //              val gFt = sigmoid(elites.head(3) * d.x + elites.head(4) * zBefore + sBefore)
      //              val gIt = sigmoid(elites.head(5) * d.x + elites.head(6) * zBefore + sBefore)
      //
      //              val s = gFt :* sBefore + gIt :* sigmoid(mu)
      //              val gOt = sigmoid(elites.head(7) * d.x + elites.head(8) * zBefore + s)
      //
      //              val z2 = gOt :* sigmoid(s)
      //              val a2 = DenseVector.vertcat(DenseVector.ones[Double](1), z2)
      //              val z3 = elites.head(1) * a2
      //
      //              val hx = z3(0)
      //              (z2, s, Math.abs(hx - d.y) * yStd)
      //          }._3
      //        }.toArray
      //
      //        val errorMean = mean(errors)
      //        val errorStd = stddev(errors)
      //        println(s"ErrorMean = $errorMean, ErrorStd = $errorStd")
      //      }
    }
  }

  def updateTheta(zBefore: DenseVector[Double],
                  sBefore: DenseVector[Double],
                  dataList: List[Data],
                  theta: Array[DenseMatrix[Double]]): (Double, Double, Double, Double) =
    dataList match {
      case d :: xs =>
        val wIi = theta(0).copy
        val wIh = theta(1).copy
        val wIc = theta(2).copy
        val wFi = theta(3).copy
        val wFh = theta(4).copy
        val wFc = theta(5).copy
        val wCi = theta(6).copy
        val wCh = theta(7).copy
        val wOi = theta(8).copy
        val wOh = theta(9).copy
        val wOc = theta(10).copy
        val wout = theta(11).copy

        val aI = wIi * d.x + wIh * zBefore + wIc * sBefore
        val bI = sigmoid(aI(0))
        val aF = wFi * d.x + wFh * zBefore + wFc * sBefore
        val bF = sigmoid(aF(0))
        val aC = wCi * d.x + wCh * zBefore
        val bC = sigmoid(aC)
        val s = bF * sBefore + bI * bC
        val aO = wOi * d.x + wOh * zBefore + wOc * s
        val bO = sigmoid(aO(0))
        val out = bO * s

        val a = DenseVector.vertcat(DenseVector.ones[Double](1), out)
        val nu = wout * a

        val hx = nu(0)

        val (dIh, dFh, dCh, dOh) = updateTheta(out, s, xs, theta)

        val dout = hx - d.y
        val w11 = (dout * a).toDenseMatrix
        theta(11) :-= LEARNING_RATE * w11

        val epsCMat = dout * wout(::, 1 until wout.cols) + dIh * wIh + dFh * wFh + dCh * wCh + dOh * wOh
        val epsC: DenseVector[Double] = epsCMat(0, ::).t
        val dO = (1.0 - sigmoid(aO(0))) * sigmoid(aO(0)) * sum(s :* epsC)
        val w8 = dO * d.x
        val w9 = dO * zBefore
        val w10 = dO * s
        theta(8) :-= LEARNING_RATE * w8
        theta(9) :-= LEARNING_RATE * w9
        theta(10) :-= LEARNING_RATE * w10

        val epsS = bO * DenseVector.ones[Double](NETWORK_SHAPE(0)._1)

        (0.0, 0.0, 0.0, 0.0)

      case Nil =>
        (0.0, 0.0, 0.0, 0.0)
    }
}
