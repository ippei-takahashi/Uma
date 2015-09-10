import java.io._
import scala.util.Random
import breeze.linalg._
import breeze.numerics._
import breeze.stats._

object Main {
  private[this] val NUM_OF_GENE_LOOP = 10001
  private[this] val NUM_OF_LEARNING_LOOP = 500

  private[this] val NUM_OF_GENE = 16
  private[this] val NUM_OF_ELITE = 1

  private[this] val LEARNING_RATE = 0.001
  private[this] val MOMENTUM_RATE = 0.7
  private[this] val DATA_RATE = 0.8

  private[this] val CROSSING_RATE = 0.9
  private[this] val MUTATION_RATE = 0.05
  private[this] val ALPHA = 0.35

  private[this] val BATCH_SIZE = 30

  private[this] val INPUT_SIZE = 15
  private[this] val STATE_SIZE = 6
  private[this] val OUTPUT_SIZE = 1

  private[this] val NETWORK_SHAPE = Array(
    OUTPUT_SIZE -> INPUT_SIZE, OUTPUT_SIZE -> STATE_SIZE, OUTPUT_SIZE -> STATE_SIZE,
    OUTPUT_SIZE -> INPUT_SIZE, OUTPUT_SIZE -> STATE_SIZE, OUTPUT_SIZE -> STATE_SIZE,
    STATE_SIZE -> INPUT_SIZE, STATE_SIZE -> STATE_SIZE,
    OUTPUT_SIZE -> INPUT_SIZE, OUTPUT_SIZE -> STATE_SIZE, OUTPUT_SIZE -> STATE_SIZE,
    OUTPUT_SIZE -> (STATE_SIZE + 1))
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

    val trainData = group.slice(0, trainSize).toList
    val testData = group.slice(trainSize, groupSize)

    val thetas = Array.ofDim[DenseMatrix[Double]](NUM_OF_GENE, NUM_OF_MAT)
    val momentums = Array.ofDim[DenseMatrix[Double]](NUM_OF_GENE, NUM_OF_MAT)

    for (i <- 0 until NUM_OF_GENE) {
      val theta = Array.ofDim[DenseMatrix[Double]](NUM_OF_MAT)
      val momentum = Array.ofDim[DenseMatrix[Double]](NUM_OF_MAT)

      for (j <- 0 until NUM_OF_MAT) {
        theta(j) = DenseMatrix.rand[Double](NETWORK_SHAPE(j)._1, NETWORK_SHAPE(j)._2)
        momentum(j) = DenseMatrix.zeros[Double](NETWORK_SHAPE(j)._1, NETWORK_SHAPE(j)._2)
      }
      thetas(i) = theta
      momentums(i) = momentum
    }
    val genes = thetas.zip(momentums).par

    var shuffledTrainData = Random.shuffle(trainData)
    var index = 0

    for (loop <- 0 until NUM_OF_GENE_LOOP) {
      for (i <- 0 until NUM_OF_LEARNING_LOOP) {
        if (index + BATCH_SIZE > trainData.length) {
          index = 0
          shuffledTrainData = Random.shuffle(trainData)
        }
        val currentData = shuffledTrainData.slice(index, index + BATCH_SIZE)
        index += BATCH_SIZE

        genes.foreach { case (theta, momentum) =>
          val grads = currentData.map { dataArray =>
            val (_, _, _, _, _, _, grad) = calcGrad(
              DenseVector.zeros[Double](STATE_SIZE),
              DenseVector.zeros[Double](STATE_SIZE),
              dataArray,
              theta
            )
            grad
          }.reduce { (grad1, grad2) =>
            grad1.zip(grad2).map {
              case (x, y) => x + y
            }
          }

          for (j <- theta.indices) {
            theta(j) :-= (grads(j) + MOMENTUM_RATE * momentum(j)) :/ BATCH_SIZE.toDouble
            momentum(j) = grads(j)
          }
        }
      }
      val costs = Array.ofDim[Double](NUM_OF_GENE)
      val thetaArray = genes.map(_._1.clone().map(_.copy)).toArray
      val trainDataPar = trainData.par

      for (i <- 0 until NUM_OF_GENE) {
        val theta = thetaArray(i)
        costs(i) = trainDataPar.map { dataArray =>
          dataArray.foldLeft((DenseVector.zeros[Double](STATE_SIZE), DenseVector.zeros[Double](STATE_SIZE), 0.0)) {
            case ((zPrev, sPrev, _), d) =>
              val (out, s, hx) = getHx(zPrev, sPrev, d, theta)
              val cost = Math.pow(d.y - hx, 2.0)
              (out, s, cost)
          }._3
        }.sum
      }

      val (elites, _, eliteCount) = getElites(costs, thetaArray)

      val tmpThetaArray = selectionTournament(r, costs, thetaArray)
      for (j <- 0 until NUM_OF_GENE; k <- 0 until NUM_OF_MAT) {
        thetaArray(j)(k) = tmpThetaArray(j)(k)
      }

      // 交叉
      for (j <- 0 until NUM_OF_GENE / 2; k <- 0 until NUM_OF_MAT) {
        if (r.nextDouble() < CROSSING_RATE) {
          val minMat = min(thetaArray(j * 2)(k), thetaArray(j * 2 + 1)(k)) - ALPHA * abs(thetaArray(j * 2)(k) - thetaArray(j * 2 + 1)(k))
          val maxMat = max(thetaArray(j * 2)(k), thetaArray(j * 2 + 1)(k)) + ALPHA * abs(thetaArray(j * 2)(k) - thetaArray(j * 2 + 1)(k))

          thetaArray(j * 2)(k) := minMat + (DenseMatrix.rand[Double](NETWORK_SHAPE(k)._1, NETWORK_SHAPE(k)._2) :* (maxMat - minMat))
          thetaArray(j * 2 + 1)(k) := minMat + (DenseMatrix.rand[Double](NETWORK_SHAPE(k)._1, NETWORK_SHAPE(k)._2) :* (maxMat - minMat))
        }
      }

      // 突然変異
      for (j <- 0 until NUM_OF_GENE; k <- 0 until NUM_OF_MAT) {
        if (r.nextDouble() < MUTATION_RATE) {
          val x = r.nextInt(NETWORK_SHAPE(k)._1)
          val y = r.nextInt(NETWORK_SHAPE(k)._2)
          thetaArray(j)(k)(x, y) += r.nextDouble() - 0.5
        }
      }


      for (j <- 0 until eliteCount) {
        thetaArray(j) = elites(j)
      }

      for (j <- 0 until NUM_OF_GENE) {
        genes.update(j, (thetaArray(j), genes(j)._2))
      }

      if (loop % 10 == 0) {
        val theta = elites.head
        val errors = testData.map { dataArray =>
          dataArray.foldLeft((DenseVector.zeros[Double](STATE_SIZE), DenseVector.zeros[Double](STATE_SIZE), 0.0)) {
            case ((zPrev, sPrev, _), d) =>
              val (out, s, hx) = getHx(zPrev, sPrev, d, theta)
              val error = Math.abs(d.y - hx) * yStd
              (out, s, error)
          }._3
        }.toArray

        val errorMean = mean(errors)
        val errorStd = stddev(errors)

        val costMin = min(costs)
        val costMean = mean(costs)
        val costMax = max(costs)

        println(s"LOOP$loop: ErrorMean = $errorMean, ErrorStd = $errorStd, costMin = $costMin, costMean = $costMean, costMax = $costMax")
      }
    }
  }

  def getHx(zPrev: DenseVector[Double],
            sPrev: DenseVector[Double],
            d: Data,
            theta: Array[DenseMatrix[Double]]) = {
    val wIi = theta(0)
    val wIh = theta(1)
    val wIc = theta(2)
    val wFi = theta(3)
    val wFh = theta(4)
    val wFc = theta(5)
    val wCi = theta(6)
    val wCh = theta(7)
    val wOi = theta(8)
    val wOh = theta(9)
    val wOc = theta(10)
    val wout = theta(11)

    val aI = wIi * d.x + wIh * zPrev + wIc * sPrev
    val bI = sigmoid(aI(0))
    val aF = wFi * d.x + wFh * zPrev + wFc * sPrev
    val bF = sigmoid(aF(0))
    val aC = wCi * d.x + wCh * zPrev
    val bC = sigmoid(aC)
    val s = bF * sPrev + bI * tanh(aC)
    val aO = wOi * d.x + wOh * zPrev + wOc * s
    val bO = sigmoid(aO(0))
    val out = bO * s

    val a = DenseVector.vertcat(DenseVector.ones[Double](1), out)
    val nu = wout * a

    (out, s, nu(0))
  }

  def calcGrad(zPrev: DenseVector[Double],
               sPrev: DenseVector[Double],
               dataList: List[Data],
               theta: Array[DenseMatrix[Double]]): (Double, Double, DenseVector[Double], Double, Double, DenseVector[Double], Array[DenseMatrix[Double]]) =
    dataList match {
      case d :: xs =>
        val wIi = theta(0)
        val wIh = theta(1)
        val wIc = theta(2)
        val wFi = theta(3)
        val wFh = theta(4)
        val wFc = theta(5)
        val wCi = theta(6)
        val wCh = theta(7)
        val wOi = theta(8)
        val wOh = theta(9)
        val wOc = theta(10)
        val wout = theta(11)

        val aI = wIi * d.x + wIh * zPrev + wIc * sPrev
        val bI = sigmoid(aI(0))
        val aF = wFi * d.x + wFh * zPrev + wFc * sPrev
        val bF = sigmoid(aF(0))
        val aC = wCi * d.x + wCh * zPrev
        val bC = sigmoid(aC)
        val s = bF * sPrev + bI * tanh(aC)
        val aO = wOi * d.x + wOh * zPrev + wOc * s
        val bO = sigmoid(aO(0))
        val out = bO * s

        val a = DenseVector.vertcat(DenseVector.ones[Double](1), out)
        val nu = wout * a

        val hx = nu(0)

        val (dINext, dFNext, dCNext, dONext, bFNext, epsSNext, thetaGrads) = calcGrad(out, s, xs, theta)

        val dout = hx - d.y

        val w11 = (dout * a).toDenseMatrix
        thetaGrads(11) :+= LEARNING_RATE * w11

        val dwC: DenseMatrix[Double] = dCNext.toDenseMatrix * wCh
        val epsCMat = dout * wout(::, 1 until wout.cols) + dINext * wIh + dFNext * wFh + dwC + dONext * wOh
        val epsC: DenseVector[Double] = epsCMat(0, ::).t
        val dO = (1.0 - sigmoid(aO(0))) * sigmoid(aO(0)) * sum(s :* epsC)

        val w8 = dO * d.x
        val w9 = dO * zPrev
        val w10 = dO * s
        thetaGrads(8) :+= LEARNING_RATE * w8.toDenseMatrix
        thetaGrads(9) :+= LEARNING_RATE * w9.toDenseMatrix
        thetaGrads(10) :+= LEARNING_RATE * w10.toDenseMatrix

        val epsS = bO * DenseVector.ones[Double](STATE_SIZE) :* epsC + bFNext * epsSNext + dFNext * wFc.toDenseVector + dO * wOc.toDenseVector

        val dC = bI * (1.0 - (tanh(aC) :* tanh(aC))) :* epsS
        val w6: DenseMatrix[Double] = dC.toDenseMatrix.t * d.x.toDenseMatrix
        val w7: DenseMatrix[Double] = dC.toDenseMatrix.t * zPrev.toDenseMatrix
        thetaGrads(6) :+= LEARNING_RATE * w6
        thetaGrads(7) :+= LEARNING_RATE * w7

        val dI = (1.0 - sigmoid(aI(0))) * sigmoid(aI(0)) * sum(sPrev :* epsS)
        val w3 = dI * d.x
        val w4 = dI * zPrev
        val w5 = dI * sPrev
        thetaGrads(3) :+= LEARNING_RATE * w3.toDenseMatrix
        thetaGrads(4) :+= LEARNING_RATE * w4.toDenseMatrix
        thetaGrads(5) :+= LEARNING_RATE * w5.toDenseMatrix

        val dF = (1.0 - sigmoid(aF(0))) * sigmoid(aF(0)) * sum(tanh(aC) :* epsS)
        val w0 = dF * d.x
        val w1 = dF * zPrev
        val w2 = dF * sPrev
        thetaGrads(0) :+= LEARNING_RATE * w0.toDenseMatrix
        thetaGrads(1) :+= LEARNING_RATE * w1.toDenseMatrix
        thetaGrads(2) :+= LEARNING_RATE * w2.toDenseMatrix

        (dI, dF, dC, dO, bF, epsS, thetaGrads)

      case Nil => {
        val array = Array.ofDim[DenseMatrix[Double]](NUM_OF_MAT)
        for (i <- theta.indices) {
          array(i) = DenseMatrix.zeros[Double](NETWORK_SHAPE(i)._1, NETWORK_SHAPE(i)._2)
        }
        (0.0, 0.0, DenseVector.zeros[Double](STATE_SIZE), 0.0, 0.0, DenseVector.zeros[Double](STATE_SIZE), array)
      }
    }

  def getElites(costs: Array[Double], thetaArray: Array[Array[DenseMatrix[Double]]]): (Array[Array[DenseMatrix[Double]]], Array[Double], Int) = {
    val sorted = costs.zipWithIndex.sortBy {
      case (c, _) => c
    }.map {
      case (c, index) => thetaArray(index) -> c
    }
    val elites = Array.ofDim[DenseMatrix[Double]](NUM_OF_ELITE, NUM_OF_MAT)
    val eliteCosts = Array.ofDim[Double](NUM_OF_ELITE)

    elites(0) = sorted.head._1.clone().map(_.copy)
    eliteCosts(0) = sorted.head._2
    var eliteCount = 1
    var j = 1
    while (j < NUM_OF_GENE && eliteCount < NUM_OF_ELITE) {
      if (!sorted(j)._1.sameElements(elites(eliteCount - 1))) {
        elites(eliteCount) = sorted(j)._1.clone().map(_.copy)
        eliteCosts(eliteCount) = sorted(j)._2
        eliteCount += 1
      }
      j += 1
    }
    (elites, eliteCosts, eliteCount)
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
