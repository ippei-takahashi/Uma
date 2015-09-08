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

  val NETWORK_SHAPE = Array(6 -> 15, 6 -> 6, 6 -> 15, 6 -> 6, 6 -> 15, 6 -> 6, 6 -> 15, 6 -> 6, 1 -> 7)
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

  def updateTheta(zBefore: DenseVector[Double], sBefore: DenseVector[Double],
                  dataList: List[Data], theta: Array[DenseMatrix[Double]]): DenseVector[Double] = dataList match {
    case d :: xs =>
      val win = theta(0).copy
      val w = theta(1).copy
      val wFin = theta(2).copy
      val wF = theta(3).copy
      val wIin = theta(4).copy
      val wI = theta(5).copy
      val wOin = theta(6).copy
      val wO = theta(7).copy
      val wout = theta(8).copy

      val mu = win * d.x + w * zBefore

      val gFt = sigmoid(wFin * d.x + wF * zBefore + sBefore)
      val gIt = sigmoid(wIin * d.x + wI * zBefore + sBefore)

      val s = gFt :* sBefore + gIt :* sigmoid(mu)
      val muOt = wOin * d.x + wO * zBefore + s
      val gOt = sigmoid(muOt)

      val z = gOt :* sigmoid(s)
      val a = DenseVector.vertcat(DenseVector.ones[Double](1), z)
      val nu = wout * a

      val hx = nu(0)

      val deltaPlus = updateTheta(z, s, xs, theta)

      val dout = hx - d.y
      val w8 = (dout * a).toDenseMatrix
      theta(8) :-= LEARNING_RATE * w8

      val eps: DenseVector[Double] = dout * wout(::, 1 until wout.cols) + deltaPlus.toDenseMatrix * theta(1)
      val dOt = (1.0 - sigmoid(muOt)) :* sigmoid(muOt) :* sigmoid(s) :* eps.t
      theta(6) :-= LEARNING_RATE * d.x

      d3

    case Nil => DenseVector.zeros[Double](NETWORK_SHAPE(0)._1)
  }
}
