import java.io._
import scala.util.Random
import breeze.linalg._
import breeze.numerics._
import breeze.stats._

object Main {
  private[this] val NUM_OF_GENE_LOOP = 1000001

  private[this] val NUM_OF_GENE = 100
  private[this] val NUM_OF_ELITE = 5

  private[this] val LEARNING_RATE = 0.001
  private[this] val DATA_RATE = 0.8

  private[this] val CROSSING_RATE = 0.85
  private[this] val MUTATION_RATE = 0.15
  private[this] val ALPHA = 0.5

  private[this] val BATCH_SIZE = 100
  private[this] val BATCH_UPDATE_PERIOD = 500

  private[this] val INPUT_SIZE = 15
  private[this] val HIDDEN_SIZE = 6
  private[this] val OUTPUT_SIZE = 1

  private[this] val NETWORK_SHAPE = Array(
    HIDDEN_SIZE -> INPUT_SIZE, HIDDEN_SIZE -> (HIDDEN_SIZE + 1), OUTPUT_SIZE -> (HIDDEN_SIZE + 1)
  )
  private[this] val NUM_OF_MAT = NETWORK_SHAPE.length

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

    val yStd = dataStd(dataT.rows - 1)

    val array = Array.ofDim[DenseVector[Double]](size)
    for (i <- 0 until size) {
      array(i) = newData(i, ::).t
    }

    val group = Random.shuffle(array.groupBy(_(0)).values.toList.map(_.reverseMap { d =>
      new Data(DenseVector.vertcat(DenseVector.ones[Double](1), d(1 until data.cols - 16)), d(data.cols - 1))
    }.toList)).par

    val groupSize = group.length
    val trainSize = (DATA_RATE * groupSize).toInt

    val trainDataPar = group.slice(0, trainSize)
    val trainData = trainDataPar.toList
    val testData = group.slice(trainSize, groupSize)

    val thetaArray = Array.ofDim[DenseMatrix[Double]](NUM_OF_GENE, NUM_OF_MAT)

    for (i <- 0 until NUM_OF_GENE) {
      val theta = Array.ofDim[DenseMatrix[Double]](NUM_OF_MAT)

      for (j <- 0 until NUM_OF_MAT) {
        theta(j) = DenseMatrix.rand[Double](NETWORK_SHAPE(j)._1, NETWORK_SHAPE(j)._2)
      }
      thetaArray(i) = theta
    }

    var currentData = Random.shuffle(trainData).slice(0, BATCH_SIZE).par

    for (loop <- 0 until NUM_OF_GENE_LOOP) {
      if (loop % BATCH_UPDATE_PERIOD == 0) {
        currentData = Random.shuffle(trainData).slice(0, BATCH_SIZE).par
      }

      val costs = Array.ofDim[Double](NUM_OF_GENE)

      for (i <- 0 until NUM_OF_GENE) {
        val theta = thetaArray(i)
        costs(i) = currentData.map { dataArray =>
          dataArray.foldLeft((DenseVector.zeros[Double](HIDDEN_SIZE + 1), 0.0)) {
            case ((zPrev, _), d) =>
              val (out, hx) = getHx(zPrev, d, theta)
              val cost = Math.pow(d.y - hx, 2.0) * (Math.pow(yStd, 2.0) / BATCH_SIZE)
              (out, cost)
          }._2
        }.sum
      }

      val (elites, sortedCosts) = getElites(costs, thetaArray)

      val tmpThetaArray = selectionTournament(r, costs, thetaArray)

      // 交叉
      for (j <- 0 until NUM_OF_GENE / 2; k <- 0 until NUM_OF_MAT) {
        if (r.nextDouble() < CROSSING_RATE) {
          val minMat = min(tmpThetaArray(j * 2)(k), tmpThetaArray(j * 2 + 1)(k)) - ALPHA * abs(tmpThetaArray(j * 2)(k) - tmpThetaArray(j * 2 + 1)(k))
          val maxMat = max(tmpThetaArray(j * 2)(k), tmpThetaArray(j * 2 + 1)(k)) + ALPHA * abs(tmpThetaArray(j * 2)(k) - tmpThetaArray(j * 2 + 1)(k))

          tmpThetaArray(j * 2)(k) := minMat + (DenseMatrix.rand[Double](NETWORK_SHAPE(k)._1, NETWORK_SHAPE(k)._2) :* (maxMat - minMat))
          tmpThetaArray(j * 2 + 1)(k) := minMat + (DenseMatrix.rand[Double](NETWORK_SHAPE(k)._1, NETWORK_SHAPE(k)._2) :* (maxMat - minMat))
        }
      }

      // 突然変異
      for (j <- 0 until NUM_OF_GENE; k <- 0 until NUM_OF_MAT) {
        if (r.nextDouble() < MUTATION_RATE) {
          val x = r.nextInt(NETWORK_SHAPE(k)._1)
          val y = r.nextInt(NETWORK_SHAPE(k)._2)
          tmpThetaArray(j)(k)(x, y) += r.nextDouble() - 0.5
        }
      }


      for (j <- 0 until NUM_OF_ELITE) {
        tmpThetaArray(j) = elites(j)
      }

      for (j <- 0 until NUM_OF_GENE) {
        thetaArray(j) = tmpThetaArray(j)
      }

      if (loop % 50 == 0) {
        val theta = elites.head
        val errors = testData.map { dataArray =>
          dataArray.foldLeft((DenseVector.zeros[Double](HIDDEN_SIZE + 1), 0.0)) {
            case ((zPrev, _), d) =>
              val (out, hx) = getHx(zPrev, d, theta)
              val error = Math.abs(d.y - hx) * yStd
              (out, error)
          }._2
        }.toArray

        val errorMean = mean(errors)
        val errorStd = stddev(errors)

        val cost1 = sortedCosts.head
        val cost2 = sortedCosts(2)
        val cost3 = sortedCosts(3)
        val cost4 = sortedCosts(4)

        println(s"LOOP$loop: ErrorMean = $errorMean, ErrorStd = $errorStd, cost1 = $cost1, cost2 = $cost2, cost3 = $cost3, cost4 = $cost4")
      }
    }
  }

  def getHx(zPrev: DenseVector[Double],
            d: Data,
            theta: Array[DenseMatrix[Double]]) = {
    val wI = theta(0)
    val wH = theta(1)
    val wO = theta(2)

    val out = wI * d.x + wH * zPrev

    val a = DenseVector.vertcat(DenseVector.ones[Double](1), out)
    val nu = wO * a

    (DenseVector.vertcat(DenseVector(d.y), out), nu(0))
  }

  def calcGrad(zPrev: DenseVector[Double],
               dataList: List[Data],
               theta: Array[DenseMatrix[Double]]): (DenseVector[Double], Array[DenseMatrix[Double]]) =
    dataList match {
      case d :: xs =>
        val wI = theta(0)
        val wH = theta(1)
        val wO = theta(2)

        val out = sigmoid(wI * d.x + wH * zPrev)

        val a = DenseVector.vertcat(DenseVector.ones[Double](1), out)
        val nu = wO * a

        val hx = nu(0)

        val (dHNext, thetaGrads) = calcGrad(DenseVector.vertcat(DenseVector(d.y), out), xs, theta)

        val dO = hx - d.y
        val w2 = (dO * a).toDenseMatrix
        thetaGrads(2) :+= LEARNING_RATE * w2

        val epsH: DenseMatrix[Double] = dO * wO(::, 1 until (HIDDEN_SIZE + 1)) + dHNext.toDenseMatrix * wH(::, 1 until (HIDDEN_SIZE + 1))
        val dH = ((1.0 - sigmoid(out)) :* sigmoid(out)).toDenseMatrix :* epsH

        val w0 = dH.t * d.x.toDenseMatrix
        val w1 = dH.t * zPrev.toDenseMatrix
        thetaGrads(0) :+= LEARNING_RATE * w0
        thetaGrads(1) :+= LEARNING_RATE * w1

        (dH.toDenseVector, thetaGrads)

      case Nil => {
        val array = Array.ofDim[DenseMatrix[Double]](NUM_OF_MAT)
        for (i <- theta.indices) {
          array(i) = DenseMatrix.zeros[Double](NETWORK_SHAPE(i)._1, NETWORK_SHAPE(i)._2)
        }
        (DenseVector.zeros[Double](HIDDEN_SIZE), array)
      }
    }

  def getElites(costs: Array[Double], thetaArray: Array[Array[DenseMatrix[Double]]]): (Array[Array[DenseMatrix[Double]]], Array[Double]) = {
    val sorted = costs.zipWithIndex.sortBy {
      case (c, _) => c
    }.map {
      case (c, index) => thetaArray(index) -> c
    }

    val elites = sorted.slice(0, NUM_OF_ELITE).map(_._1.clone().map(_.copy))
    val sortedCosts = sorted.map(_._2)

    (elites, sortedCosts)
  }

  def selectionTournament(r: Random, costs: Array[Double], thetaArray: Array[Array[DenseMatrix[Double]]]): Array[Array[DenseMatrix[Double]]] = {
    val tmpThetaArray = Array.ofDim[DenseMatrix[Double]](NUM_OF_GENE, NUM_OF_MAT)

    for (j <- 0 until NUM_OF_GENE) {
      val a = r.nextInt(NUM_OF_GENE)
      val b = r.nextInt(NUM_OF_GENE)
      if (costs(a) < costs(b)) {
        tmpThetaArray(j) = thetaArray(a).clone().map(_.copy)
      } else {
        tmpThetaArray(j) = thetaArray(b).clone().map(_.copy)
      }
    }
    tmpThetaArray
  }
}
