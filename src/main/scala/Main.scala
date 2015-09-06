import java.io._
import scala.util.Random
import breeze.linalg._
import breeze.numerics._
import breeze.stats._

object Main {
  val NUM_OF_INDIVIDUAL = 100
  val NUM_OF_GENERATION = 100001
  val NUM_OF_ELITE = 4
  val NUM_OF_SAVED_ELITE = 4
  val NUM_OF_LAYER_MAT = 2

  val DATA_RATE = 0.8

  val CROSSING_RATE = 0.85
  val MUTATION_RATE = 0.15
  val ALPHA = 0.5
  val LAMBDA = 0.0003

  val NETWORK_SHAPE = Array(6 -> 30, 1 -> 7)

  def main(args: Array[String]) {

    val r = new Random()

    val dataCSV = new File("data.csv")

    val data = csvread(dataCSV)

    val size = data.rows

    val xx = data(::, 0 until data.cols - 1)
    val xt = xx.t
    val xMean: DenseVector[Double] = mean(xt(*, ::))
    val xStd: DenseVector[Double] = stddev(xt(*, ::))

    val xNorm : DenseMatrix[Double] = xt.copy

    for (i <- 0 until size) {
      xNorm(::, i) := (xNorm(::, i) :- xMean) :/ xStd
    }

    val yy: DenseVector[Double] = data(::, data.cols - 1)
    val yMean: Double = mean(yy)
    val yStd: Double  = stddev(yy)

    val x = DenseMatrix.horzcat(DenseMatrix.ones[Double](size, 1), xNorm.t)
    val y: DenseVector[Double] = (yy - yMean) / yStd

    val xrand = DenseMatrix.zeros[Double](x.rows, x.cols)
    val yrand = DenseVector.zeros[Double](y.length)

    val randomArray = Random.shuffle(0 to x.rows - 1).toArray

    for (i <- randomArray.indices) {
      xrand(i, ::) := x(randomArray(i), ::)
      yrand(i) = y(randomArray(i))
    }

    val dataSize = (DATA_RATE * x.rows).toInt
    val valSize = size - dataSize

    val a1 = xrand(0 until dataSize, ::)
    val a1val = xrand(dataSize until size, ::)

    val y1 = yrand(0 until dataSize)
    val y1val = yrand(dataSize until size)

    val individuals = Array.ofDim[DenseMatrix[Double]](NUM_OF_INDIVIDUAL, NUM_OF_LAYER_MAT)

    for (i <- 0 until NUM_OF_INDIVIDUAL; j <- 0 until NUM_OF_LAYER_MAT) {
      individuals(i)(j) = DenseMatrix.rand(NETWORK_SHAPE(j)._1, NETWORK_SHAPE(j)._2) - 0.5
    }

    for (i <- 0 until NUM_OF_GENERATION) {
      val costs = Array.ofDim[Double](NUM_OF_INDIVIDUAL)

      for (j <- 0 until NUM_OF_INDIVIDUAL) {
        val z2: DenseMatrix[Double] = a1 * individuals(j)(0).t
        val a2: DenseMatrix[Double] = DenseMatrix.horzcat(DenseMatrix.ones[Double](dataSize, 1), z2)
        val z3: DenseMatrix[Double] = a2 * individuals(j)(1).t

        val hx: DenseVector[Double] = z3(::, 0)

        costs(j) += ((hx - y1).t * (hx - y1)) * yStd / dataSize / 2.0

        for (k <- 0 until NUM_OF_LAYER_MAT) {
          costs(j) += LAMBDA / 2 * sum(individuals(j)(k)(::, 1 until NETWORK_SHAPE(k)._2) :^ 2.0)
        }
      }

      // 優秀な数匹は次の世代に持ち越し
      val (elites, _, eliteCount) = getElites(costs, individuals)

      if (i % 50 == 0) {
        println(s"LOOP$i: min = ${costs.min}, average = ${costs.sum / costs.length}, max = ${costs.max}")
      }
      if (i % 500 == 0) {
        val z2val: DenseMatrix[Double] = a1val * elites.head(0).t
        val a2val: DenseMatrix[Double] = DenseMatrix.horzcat(DenseMatrix.ones[Double](valSize, 1), z2val)
        val z3val: DenseMatrix[Double] = a2val * elites.head(1).t

        val hxval: DenseVector[Double] = z3val(::, 0)
        val s: DenseVector[Double] = sqrt(((hxval * yStd) - (y1val * yStd)) :^ 2.0)
        val errorMean = mean(s)
        val errorStd = stddev(s)

        println(s"ErrorMean = $errorMean, ErrorStd = $errorStd")

        for (j <- 0 until NUM_OF_LAYER_MAT) {
          csvwrite(new File(s"result_${i}_$j.csv"), elites.head(j))
        }
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
