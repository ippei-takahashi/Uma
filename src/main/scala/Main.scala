import java.io._
import scala.collection.parallel.immutable.ParSeq
import scala.util.Random
import breeze.linalg._
import breeze.numerics._
import breeze.stats._

object Main {
  private[this] val NUM_OF_GENE_LOOP = 1000001
  private[this] val NUM_OF_LEARNING_LOOP = 20

  private[this] val NUM_OF_GENE = 50
  private[this] val NUM_OF_ELITE = 2

  private[this] val DATA_RATE = 0.8

  private[this] val LAMBDA = 0.01

  private[this] val CROSSOVER_RATE = 0.6
  private[this] val MUTATION_RATE = 0.001

  private[this] val INNER_CROSSOVER_RATE = 0.8
  private[this] val INNER_MUTATION_RATE = 0.005

  private[this] val ALPHA = 0.5

  private[this] val BATCH_SIZE = 500

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

    for (i <- 0 until NUM_OF_GENE) {
      val theta = Array.ofDim[DenseMatrix[Double]](NUM_OF_MAT)
      for (j <- 0 until NUM_OF_MAT) {
        theta(j) = 2.0 * DenseMatrix.rand[Double](NETWORK_SHAPE(j)._1, NETWORK_SHAPE(j)._2) - 1.0
      }
      thetas(i) = theta
    }

    val genes = thetas.par

    var shuffledTrainData = Random.shuffle(trainData)
    var index = 0

    for (loop <- 0 until NUM_OF_GENE_LOOP) {
      if ((index + 1) * BATCH_SIZE > trainData.length) {
        index = 0
        shuffledTrainData = Random.shuffle(trainData)
      }
      val currentData = shuffledTrainData.slice(index * BATCH_SIZE, (index + 1) * BATCH_SIZE).par

      val thetaArray = {
        val notLearnedThetaArray = genes.map(_.clone().map(_.copy)).toArray

        val newGenes = genes.map {
          _.clone().map(_.copy)
        }.toArray

        for (i <- 0 until NUM_OF_LEARNING_LOOP) {
          val costs = calcCost(yStd, currentData, BATCH_SIZE, newGenes)

          val (elites, _) = getElites(costs, newGenes)

          val tmpThetaArray = selectionTournament(r, costs, newGenes)

          crossover(r, INNER_CROSSOVER_RATE, tmpThetaArray)
          mutation(INNER_MUTATION_RATE, tmpThetaArray)

          for (j <- 0 until NUM_OF_ELITE) {
            tmpThetaArray(j) = elites(j)
          }

          for (j <- 0 until NUM_OF_GENE) {
            newGenes(j) = tmpThetaArray(j)
          }


          index += BATCH_SIZE * NUM_OF_GENE
        }


        val learnedThetaArray = newGenes.map(_.clone().map(_.copy))
        notLearnedThetaArray ++ learnedThetaArray
      }

      val costs = calcCost(yStd, trainDataPar, trainSize, thetaArray)

      val (newCosts, newThetaArray) = costs.zip(thetaArray).collect {
        case (cost, theta) if !cost.equals(Double.NaN) => (cost, theta)
      }.unzip

      val (elites, sortedCosts) = getElites(newCosts, newThetaArray)

      val tmpThetaArray = selectionTournament(r, newCosts, newThetaArray)

      crossover(r, CROSSOVER_RATE, tmpThetaArray)
      mutation(MUTATION_RATE, tmpThetaArray)

      for (j <- 0 until NUM_OF_ELITE) {
        tmpThetaArray(j) = elites(j)
      }

      for (j <- 0 until NUM_OF_GENE) {
        genes(j) = tmpThetaArray(j)
      }

      if (loop % 1 == 0) {
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
        val cost2 = sortedCosts(4)
        val cost3 = sortedCosts(10)
        val cost4 = sortedCosts(20)

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

    val out = wI * d.x + wH * zPrev

    val a = DenseVector.vertcat(DenseVector.ones[Double](1), out)
    val nu = wO * a

    (DenseVector.vertcat(DenseVector(d.y), out), nu(0))
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

  def crossover(r: Random, crossoverRate: Double, tmpThetaArray: Array[Array[DenseMatrix[Double]]]) {
    for (j <- 0 until NUM_OF_GENE / 2; k <- 0 until NUM_OF_MAT) {
      if (r.nextDouble() < crossoverRate) {
        val minMat = min(tmpThetaArray(j * 2)(k), tmpThetaArray(j * 2 + 1)(k)) - ALPHA * abs(tmpThetaArray(j * 2)(k) - tmpThetaArray(j * 2 + 1)(k))
        val maxMat = max(tmpThetaArray(j * 2)(k), tmpThetaArray(j * 2 + 1)(k)) + ALPHA * abs(tmpThetaArray(j * 2)(k) - tmpThetaArray(j * 2 + 1)(k))

        tmpThetaArray(j * 2)(k) := minMat + (DenseMatrix.rand[Double](NETWORK_SHAPE(k)._1, NETWORK_SHAPE(k)._2) :* (maxMat - minMat))
        tmpThetaArray(j * 2 + 1)(k) := minMat + (DenseMatrix.rand[Double](NETWORK_SHAPE(k)._1, NETWORK_SHAPE(k)._2) :* (maxMat - minMat))
      }
    }
  }

  def mutation(mutationRate: Double, tmpThetaArray: Array[Array[DenseMatrix[Double]]]) {
    for (j <- 0 until NUM_OF_GENE; k <- 0 until NUM_OF_MAT) {
      val mask = DenseMatrix.rand[Double](NETWORK_SHAPE(k)._1, NETWORK_SHAPE(k)._2).map { x =>
        if (x < mutationRate) 1.0 else 0.0
      }
      val update = 2.0 * DenseMatrix.rand[Double](NETWORK_SHAPE(k)._1, NETWORK_SHAPE(k)._2) - 1.0

      tmpThetaArray(j)(k) += mask :* update
    }
  }
}