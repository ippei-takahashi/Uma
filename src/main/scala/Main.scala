import java.io._
import scala.util.Random
import breeze.linalg._
import breeze.numerics._
import breeze.stats._

object Main {
  val NUM_OF_INDIVIDUAL = 100
  val NUM_OF_GENERATION = 100001
  val NUM_OF_ELITE = 4
  val NUM_OF_LAYER_MAT = 3

  val DATA_RATE = 0.8

  val CROSSING_RATE = 0.85
  val MUTATION_RATE = 0.15
  val ALPHA = 0.5
  val LAMBDA = 0.0003

  val BATCH_SIZE = 1000

  val NETWORK_SHAPE = Array(6 -> 30, 1 -> 7, 6 -> 6)

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

    class Data(val x: DenseVector[Double], val y: Double)
    val group = Random.shuffle(array.groupBy(_(0)).values.toList.map(_.map { d =>
      new Data(DenseVector.vertcat(DenseVector.ones[Double](1), d(1 until data.cols - 1)), d(data.cols - 1))
    })).par

    val groupSize = group.length
    val trainSize = (DATA_RATE * groupSize).toInt
    val testSize = groupSize - trainSize

    val trainData = group.slice(0, trainSize)
    val testData = group.slice(trainSize, groupSize)

    var currentData = Random.shuffle(trainData.toList).slice(0, BATCH_SIZE).par

    val individuals = Array.ofDim[DenseMatrix[Double]](NUM_OF_INDIVIDUAL, NUM_OF_LAYER_MAT)

    for (i <- 0 until NUM_OF_INDIVIDUAL; j <- 0 until NUM_OF_LAYER_MAT) {
      individuals(i)(j) = DenseMatrix.rand(NETWORK_SHAPE(j)._1, NETWORK_SHAPE(j)._2) - 0.5
    }

    for (i <- 0 until NUM_OF_GENERATION) {
      val costs = Array.ofDim[Double](NUM_OF_INDIVIDUAL)
      if (i % 200 == 0) {
        currentData = Random.shuffle(trainData.toList).slice(0, BATCH_SIZE).par
      }

      for (j <- 0 until NUM_OF_INDIVIDUAL) {
        costs(j) = currentData.map { dataArray =>
          dataArray.foldLeft((DenseVector.zeros[Double](NETWORK_SHAPE(0)._1), 0.0)) { case ((s, _), d) =>
            val z2 = individuals(j)(0) * d.x + individuals(j)(2) * s
            val a2 = DenseVector.vertcat(DenseVector.ones[Double](1), z2)
            val z3 = individuals(j)(1) * a2

            val hx = z3(0)
            (sigmoid(z2), Math.pow((hx - d.y) * yStd, 2.0) / 2.0)
          }._2 / BATCH_SIZE
        }.sum

        //costs(j) += LAMBDA / 2 * sum(individuals(j)(k)(::, 1 until NETWORK_SHAPE(k)._2) :^ 2.0)
      }

      val (elites, _, eliteCount) = getElites(costs, individuals)

      if (i % 10 == 0) {
        println(s"LOOP$i: min = ${costs.min}, average = ${costs.sum / costs.length}, max = ${costs.max}")
      }
      if (i % 50 == 0) {
        val errors = testData.map { dataArray =>
          dataArray.foldLeft((DenseVector.zeros[Double](NETWORK_SHAPE(0)._1), 0.0)) { case ((s, _), d) =>
            val z2 = elites.head(0) * d.x + elites.head(2) * s
            val a2 = DenseVector.vertcat(DenseVector.ones[Double](1), z2)
            val z3 = elites.head(1) * a2

            val hx = z3(0)
            (sigmoid(z2), Math.abs(hx - d.y) * yStd)
          }._2
        }.toArray

        val errorMean = mean(errors)
        val errorStd = stddev(errors)
        println(s"ErrorMean = $errorMean, ErrorStd = $errorStd")

        //        for (j <- 0 until NUM_OF_LAYER_MAT) {
        //          csvwrite(new File(s"result_${i}_$j.csv"), elites.head(j))
        //        }
      }


      val tmpIndividuals = selectionTournament(r, costs, individuals)
      for (j <- 0 until NUM_OF_INDIVIDUAL; k <- 0 until NUM_OF_LAYER_MAT) {
        individuals(j)(k) = tmpIndividuals(j)(k)
      }

      // 交叉
      for (j <- 0 until NUM_OF_INDIVIDUAL / 2; k <- 0 until NUM_OF_LAYER_MAT) {
        if (r.nextDouble() < CROSSING_RATE) {
          val minMat = min(individuals(j * 2)(k), individuals(j * 2 + 1)(k)) - ALPHA * abs(individuals(j * 2)(k) - individuals(j * 2 + 1)(k))
          val maxMat = max(individuals(j * 2)(k), individuals(j * 2 + 1)(k)) + ALPHA * abs(individuals(j * 2)(k) - individuals(j * 2 + 1)(k))

          individuals(j * 2)(k) := minMat + (DenseMatrix.rand[Double](NETWORK_SHAPE(k)._1, NETWORK_SHAPE(k)._2) :* (maxMat - minMat))
          individuals(j * 2 + 1)(k) := minMat + (DenseMatrix.rand[Double](NETWORK_SHAPE(k)._1, NETWORK_SHAPE(k)._2) :* (maxMat - minMat))
        }
      }

      // 突然変異
      for (j <- 0 until NUM_OF_INDIVIDUAL; k <- 0 until NUM_OF_LAYER_MAT) {
        if (r.nextDouble() < MUTATION_RATE) {
          val x = r.nextInt(NETWORK_SHAPE(k)._1)
          val y = r.nextInt(NETWORK_SHAPE(k)._2)
          individuals(j)(k)(x, y) += r.nextDouble() - 0.5
        }
      }

      for (j <- 0 until eliteCount) {
        individuals(j) = elites(j)
      }
    }
  }

  def getElites(costs: Array[Double], individuals: Array[Array[DenseMatrix[Double]]]): (Array[Array[DenseMatrix[Double]]], Array[Double], Int) = {
    val sorted = costs.zipWithIndex.sortBy {
      case (c, _) => c
    }.map {
      case (c, index) => individuals(index) -> c
    }
    val elites = Array.ofDim[DenseMatrix[Double]](NUM_OF_ELITE, NUM_OF_LAYER_MAT)
    val eliteCosts = Array.ofDim[Double](NUM_OF_ELITE)

    elites(0) = sorted.head._1.clone().map(_.copy)
    eliteCosts(0) = sorted.head._2
    var eliteCount = 1
    var j = 1
    while (j < NUM_OF_INDIVIDUAL && eliteCount < NUM_OF_ELITE) {
      if (!sorted(j)._1.sameElements(elites(eliteCount - 1))) {
        elites(eliteCount) = sorted(j)._1.clone().map(_.copy)
        eliteCosts(eliteCount) = sorted(j)._2
        eliteCount += 1
      }
      j += 1
    }
    (elites, eliteCosts, eliteCount)
  }

  def selectionTournament(r: Random, costs: Array[Double], individuals: Array[Array[DenseMatrix[Double]]]): Array[Array[DenseMatrix[Double]]] = {
    val tmpIndividuals = Array.ofDim[DenseMatrix[Double]](NUM_OF_INDIVIDUAL, NUM_OF_LAYER_MAT)

    for (j <- 0 until NUM_OF_INDIVIDUAL) {
      val a = r.nextInt(NUM_OF_INDIVIDUAL)
      val b = r.nextInt(NUM_OF_INDIVIDUAL)
      if (costs(a) < costs(b)) {
        tmpIndividuals(j) = individuals(a).clone().map(_.copy)
      } else {
        tmpIndividuals(j) = individuals(b).clone().map(_.copy)
      }
    }
    tmpIndividuals
  }
}


