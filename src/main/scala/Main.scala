import java.io._
import scala.collection.parallel.immutable.ParSeq
import scala.util.Random
import breeze.linalg._
import breeze.numerics._
import breeze.stats._

object Main {
  private[this] val NUM_OF_GENE_LOOP = 1000001
  private[this] val NUM_OF_LEARNING_LOOP = 200

  private[this] val START_LEARNING = 5000
  private[this] val FIRST_TRAIN_SIZE = 500

  private[this] val NUM_OF_GENE = 80
  private[this] val NUM_OF_ELITE = 4

  private[this] val LEARNING_PERIOD = 1

  private[this] val DATA_RATE = 0.8

  private[this] val LAMBDA = 0.01
  private[this] val LEARNING_RATE = 0.00001
  private[this] val MOMENTUM_RATE = 0.9

  private[this] val CROSSING_RATE = 0.0
  private[this] val FIRST_CROSSING_RATE = 0.9
  private[this] val MUTATION_RATE = 0.05
  private[this] val FIRST_MUTATION_RATE = 0.15
  private[this] val ALPHA = 0.5

  private[this] val BATCH_SIZE = 30

  private[this] val MAX_LENGTH = 5

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
    }.slice(0, MAX_LENGTH).toList)).par

    val groupSize = group.length
    val trainSize = (DATA_RATE * groupSize).toInt

    val trainDataPar = group.slice(0, trainSize)
    val trainData = trainDataPar.toList
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

    val genes = thetas.zip(momentums).zipWithIndex.par

    var shuffledTrainData = Random.shuffle(trainData)
    var index = 0

    val firstTrainData = shuffledTrainData.slice(0, FIRST_TRAIN_SIZE).par

    for (loop <- 0 until NUM_OF_GENE_LOOP) {
      val (numOfGene, numOfElite) = if (loop >= START_LEARNING)
        (1, 1)
      else
        (NUM_OF_GENE, NUM_OF_ELITE)


      val newGenes = genes.map { case ((theta, momentum), geneIndex) =>
        ((theta.clone().map(_.copy), momentum), geneIndex)
      }

      val notLearnedThetaArray = genes.map(_._1._1.clone().map(_.copy)).toArray

      val thetaArray = if (loop >= START_LEARNING && loop % LEARNING_PERIOD == 0) {
        for (i <- 0 until NUM_OF_LEARNING_LOOP) {
          if (index + BATCH_SIZE * numOfGene > trainData.length) {
            index = 0
            shuffledTrainData = Random.shuffle(trainData)
          }

          newGenes.foreach { case ((theta, momentum), geneIndex) =>
            val currentData = shuffledTrainData.slice(index + BATCH_SIZE * geneIndex, index + BATCH_SIZE * (geneIndex + 1))

            val grads = currentData.map { dataArray =>
              calcGrad(
                DenseVector.zeros[Double](HIDDEN_SIZE + 1),
                dataArray,
                theta
              )._2
            }.reduce { (grad1, grad2) =>
              grad1.zip(grad2).map {
                case (x, y) => x + y
              }
            }

            for (j <- theta.indices) {
              val w = MOMENTUM_RATE * momentum(j) - LEARNING_RATE * grads(j) :/ BATCH_SIZE.toDouble
              theta(j) :+= w
              momentum(j) = w
            }
          }

          index += BATCH_SIZE * numOfGene
        }

        val learnedThetaArray = newGenes.map(_._1._1.clone().map(_.copy)).toArray
        notLearnedThetaArray ++ learnedThetaArray
      } else {
        notLearnedThetaArray
      }

      val costs = if (loop < START_LEARNING)
        calcCost(yStd, firstTrainData, FIRST_TRAIN_SIZE, thetaArray)
      else
        calcCost(yStd, trainDataPar, trainSize, thetaArray)

      val (newCosts, newThetaArray) = costs.zip(thetaArray).collect {
        case (cost, theta) if !cost.equals(Double.NaN) => (cost, theta)
      }.unzip

      val (elites, sortedCosts) = getElites(numOfElite, newCosts, newThetaArray)

      val tmpThetaArray = selectionTournament(numOfGene, r, newCosts, newThetaArray)

      // 交叉
      val (crossingRate, mutationRate) =
        if (loop < START_LEARNING)
          (FIRST_CROSSING_RATE, FIRST_MUTATION_RATE)
        else
          (CROSSING_RATE, MUTATION_RATE)

      for (j <- 0 until numOfGene / 2; k <- 0 until NUM_OF_MAT) {
        if (r.nextDouble() < crossingRate) {
          val minMat = min(tmpThetaArray(j * 2)(k), tmpThetaArray(j * 2 + 1)(k)) - ALPHA * abs(tmpThetaArray(j * 2)(k) - tmpThetaArray(j * 2 + 1)(k))
          val maxMat = max(tmpThetaArray(j * 2)(k), tmpThetaArray(j * 2 + 1)(k)) + ALPHA * abs(tmpThetaArray(j * 2)(k) - tmpThetaArray(j * 2 + 1)(k))

          tmpThetaArray(j * 2)(k) := minMat + (DenseMatrix.rand[Double](NETWORK_SHAPE(k)._1, NETWORK_SHAPE(k)._2) :* (maxMat - minMat))
          tmpThetaArray(j * 2 + 1)(k) := minMat + (DenseMatrix.rand[Double](NETWORK_SHAPE(k)._1, NETWORK_SHAPE(k)._2) :* (maxMat - minMat))
        }
      }

      // 突然変異
      for (j <- 0 until numOfGene; k <- 0 until NUM_OF_MAT) {
        if (r.nextDouble() < mutationRate) {
          val x = r.nextInt(NETWORK_SHAPE(k)._1)
          val y = r.nextInt(NETWORK_SHAPE(k)._2)
          tmpThetaArray(j)(k)(x, y) += r.nextDouble() - 0.5
        }
      }


      for (j <- 0 until numOfElite) {
        tmpThetaArray(j) = elites(j)
      }

      for (j <- 0 until numOfGene) {
        val momentum = Array.ofDim[DenseMatrix[Double]](NUM_OF_MAT)

        for (k <- 0 until NUM_OF_MAT) {
          momentum(k) = DenseMatrix.zeros[Double](NETWORK_SHAPE(k)._1, NETWORK_SHAPE(k)._2)
        }

        genes.update(j, ((tmpThetaArray(j), momentum), genes(j)._2))
      }

      if (loop >= START_LEARNING && loop % LEARNING_PERIOD == 0) {
        println(costs(numOfGene) - costs(0))
      }

      if (loop % 50 == 0 || loop >= START_LEARNING && loop % 5 == 0) {
        val theta = elites.head
        val errorsOne = testData.filter(_.length == 1).map { dataArray =>
          dataArray.foldLeft((DenseVector.zeros[Double](HIDDEN_SIZE + 1), 0.0)) {
            case ((zPrev, _), d) =>
              val (out, hx) = getHx(zPrev, d, theta)
              val error = Math.abs(d.y - hx) * yStd
              (out, error)
          }._2
        }.toArray

        val errors = testData.filter(_.length != 1).map { dataArray =>
          dataArray.foldLeft((DenseVector.zeros[Double](HIDDEN_SIZE + 1), 0.0)) {
            case ((zPrev, _), d) =>
              val (out, hx) = getHx(zPrev, d, theta)
              val error = Math.abs(d.y - hx) * yStd
              (out, error)
          }._2
        }.toArray

        val errorOneMean = mean(errorsOne)
        val errorOneStd = stddev(errorsOne)

        val errorMean = mean(errors)
        val errorStd = stddev(errors)

        val cost1 = sortedCosts.head
        val cost2 = sortedCosts(0)
        val cost3 = sortedCosts(0)
        val cost4 = sortedCosts(0)

        println(s"LOOP$loop: ErrorMean = $errorMean, ErrorStd = $errorStd, ErrorOneMean = $errorOneMean, ErrorOneStd = $errorOneStd, cost1 = $cost1, cost2 = $cost2, cost3 = $cost3, cost4 = $cost4")
      }
    }
  }

  def getHx(zPrev: DenseVector[Double],
            d: Data,
            theta: Array[DenseMatrix[Double]]) = {
    val wI = theta(0)
    val wH = theta(1)
    val wO = theta(2)

    val out = sigmoid(wI * d.x + wH * zPrev)

    val a = DenseVector.vertcat(DenseVector.ones[Double](1), out)
    val nu = wO * a

    (DenseVector.vertcat(sigmoid(DenseVector(d.y)), out), nu(0))
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

        val (dHNext, thetaGrads) = calcGrad(DenseVector.vertcat(sigmoid(DenseVector(d.y)), out), xs, theta)

        val dO = hx - d.y
        val w2 = (dO * a).toDenseMatrix

        val epsH: DenseMatrix[Double] = dO * wO(::, 1 until (HIDDEN_SIZE + 1)) + dHNext.toDenseMatrix * wH(::, 1 until (HIDDEN_SIZE + 1))
        val dH: DenseMatrix[Double] = ((1.0 - sigmoid(out)) :* sigmoid(out)).toDenseMatrix :* epsH

        val w0: DenseMatrix[Double] = dH.t * d.x.toDenseMatrix
        val w1: DenseMatrix[Double] = dH.t * zPrev.toDenseMatrix

        thetaGrads(0) :+= w0 :+ LAMBDA * wI
        thetaGrads(1) :+= w1 :+ LAMBDA * wH
        thetaGrads(2) :+= w2 + LAMBDA * wO

        (dH.toDenseVector, thetaGrads)

      case Nil =>
        val array = Array.ofDim[DenseMatrix[Double]](NUM_OF_MAT)
        for (i <- theta.indices) {
          array(i) = DenseMatrix.zeros[Double](NETWORK_SHAPE(i)._1, NETWORK_SHAPE(i)._2)
        }
        (DenseVector.zeros[Double](HIDDEN_SIZE), array)
    }

  def calcCost(yStd: Double,
               trainData: ParSeq[List[Data]],
               trainSize: Double,
               thetaArray: Array[Array[DenseMatrix[Double]]]): Array[Double] =
    thetaArray.map { theta =>
      trainData.map { dataArray =>
        dataArray.foldLeft((DenseVector.zeros[Double](HIDDEN_SIZE + 1), 0.0)) {
          case ((zPrev, _), d) =>
            val (out, hx) = getHx(zPrev, d, theta)
            val cost = Math.pow(d.y - hx, 2.0) * Math.pow(yStd, 2.0)
            val reg = LAMBDA / 2 * theta.map(x => sum(x :^ 2.0)).sum
            (out, cost + reg)
        }._2
      }.sum / trainSize
    }


  def getElites(numOfElite: Int, costs: Array[Double], thetaArray: Array[Array[DenseMatrix[Double]]]): (Array[Array[DenseMatrix[Double]]], Array[Double]) = {
    val sorted = costs.zipWithIndex.sortBy {
      case (c, _) => c
    }.map {
      case (c, index) => thetaArray(index) -> c
    }

    val elites = sorted.slice(0, numOfElite).map(_._1.clone().map(_.copy))
    val sortedCosts = sorted.map(_._2)

    (elites, sortedCosts)
  }

  def selectionTournament(numOfGene: Int, r: Random, costs: Array[Double], thetaArray: Array[Array[DenseMatrix[Double]]]): Array[Array[DenseMatrix[Double]]] = {
    val tmpThetaArray = Array.ofDim[DenseMatrix[Double]](numOfGene, NUM_OF_MAT)

    for (j <- 0 until numOfGene) {
      val a = r.nextInt(thetaArray.length)
      val b = r.nextInt(thetaArray.length)
      if (costs(a) < costs(b)) {
        tmpThetaArray(j) = thetaArray(a).clone().map(_.copy)
      } else {
        tmpThetaArray(j) = thetaArray(b).clone().map(_.copy)
      }
    }
    tmpThetaArray
  }
}
