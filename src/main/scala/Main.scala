import java.io._
import scala.util.Random
import breeze.linalg._
import breeze.numerics._
import breeze.stats.mean

object Main {
  val NUM_OF_INDIVIDUAL = 100
  val NUM_OF_GENERATION = 3001
  val NUM_OF_LABELS = 10
  val NUM_OF_ELITE = 4
  val NUM_OF_SAVED_ELITE = 4
  val NUM_OF_LAYER = 3

  val USE_BATCH = false
  val BATCH_SIZE = 500
  val DATA_SIZE = 4000
  val VAL_SIZE = 1000
  val SIZE = if (USE_BATCH) BATCH_SIZE else DATA_SIZE

  val CROSSING_RATE = 0.85
  val MUTATION_RATE = 0.15
  val ALPHA = 0.5
  val LAMBDA = 0.0003

  val NETWORK_SHAPE = Array(25 -> 401, 10 -> 26)

  def main(args: Array[String]) {

    val r = new Random()

    val xcsv = new File("x.csv")
    val ycsv = new File("y.csv")

    val data = csvread(xcsv)
    val ymat: DenseMatrix[Int] = csvread(ycsv).map(_.asInstanceOf[Int]) :% 10

    val x: DenseMatrix[Double] = DenseMatrix.horzcat(DenseMatrix.ones[Double](data.rows, 1), data)
    val y: DenseVector[Int] = ymat(::, 0)

    val xrand = DenseMatrix.zeros[Double](x.rows, x.cols)
    val yrand = DenseVector.zeros[Int](y.length)

    val randomArray = Random.shuffle(0 to x.rows - 1).toArray

    for (i <- randomArray.indices) {
      xrand(i, ::) := x(randomArray(i), ::)
      yrand(i) =  y(randomArray(i))
    }

    val a1 = xrand(0 until 4000, ::)
    val a1val = xrand(4000 until 5000, ::)

    val y1 = yrand(0 until 4000)
    val y1val = yrand(4000 until 5000)

    var a1_ = DenseMatrix.zeros[Double](BATCH_SIZE, 401)
    var y1_ = DenseVector.zeros[Int](BATCH_SIZE)

    var periodOfBatch = 500
    var nextChange = 0
    var currentElite: Option[Array[DenseMatrix[Double]]] = None
    var currentMinCost = Double.MaxValue
    var previousMinCost = Double.MaxValue
    val savedEliteQueue = scala.collection.mutable.Queue[Array[DenseMatrix[Double]]]()

    val individuals = Array.ofDim[DenseMatrix[Double]](NUM_OF_INDIVIDUAL, NUM_OF_LAYER - 1)

    for (i <- 0 until NUM_OF_INDIVIDUAL) {
      individuals(i)(0) = DenseMatrix.rand(25, 401) - 0.5
      individuals(i)(1) = DenseMatrix.rand(10, 26) - 0.5
    }

    for (i <- 0 until NUM_OF_GENERATION) {
      val costs = Array.ofDim[Double](NUM_OF_INDIVIDUAL)

      val (newa, newy) = {
        if (USE_BATCH && i >= nextChange) {
          val a1_ = DenseMatrix.zeros[Double](BATCH_SIZE, 401)
          val y1_ = DenseVector.zeros[Int](BATCH_SIZE)

          if (currentMinCost > previousMinCost) {
            periodOfBatch += 50
          }
          previousMinCost = currentMinCost
          currentMinCost = Double.MaxValue
          nextChange += periodOfBatch
          println(s"period = $periodOfBatch")

          currentElite.foreach { e =>
            savedEliteQueue.enqueue(e)
            if (savedEliteQueue.length > NUM_OF_SAVED_ELITE) {
              savedEliteQueue.dequeue()
            }
          }
          currentElite = None

          for (j <- 0 until BATCH_SIZE) {
            val n = r.nextInt(4000)
            a1_(j, ::) := a1(n, ::)
            y1_(j) = y1(n)
          }

          (a1_, y1_)
        } else if (USE_BATCH) {
          (a1_, y1_)
        } else {
          (a1, y1)
        }
      }
      a1_ = newa
      y1_ = newy

      for (j <- 0 until NUM_OF_INDIVIDUAL) {
        val z2 = a1_ * individuals(j)(0).t
        val a2 = DenseMatrix.horzcat(DenseMatrix.ones[Double](SIZE, 1), sigmoid(z2))
        val z3 = a2 * individuals(j)(1).t

        val hx: DenseMatrix[Double] = sigmoid(z3)

        for (k <- 0 until NUM_OF_LABELS) {
          val yk: DenseVector[Double] = (y1_ :== k).map(x => if (x) 1.0d else 0)
          val hxk = hx(::, k)

          val costk1: DenseVector[Double] = yk :* log(hxk.map(x => if (x == 0.0d) Double.MinPositiveValue else x))
          val costk2: DenseVector[Double] = (1.0d - yk) :* log((1.0d - hxk).map(x => if (x == 0.0d) Double.MinPositiveValue else x))
          val costk = 1.0d / SIZE * sum(-costk1 - costk2)

          costs(j) += costk
        }
        for (k <- 0 until NUM_OF_LAYER / 2) {
          costs(j) += LAMBDA / 2 * sum(individuals(j)(k)(::, 1 until NETWORK_SHAPE(k)._2) :^ 2.0)
        }
      }

      // 優秀な数匹は次の世代に持ち越し
      val (elites, _, eliteCount) = getElites(costs, individuals)

      if (i % 50 == 0) {
        println(s"LOOP$i: min = ${costs.min}, average = ${costs.sum / costs.length}, max = ${costs.max}")
      }
      if (i % 500 == 0) {
        val z2val = a1val * elites.head(0).t
        val a2val = DenseMatrix.horzcat(DenseMatrix.ones[Double](VAL_SIZE, 1), sigmoid(z2val))
        val z3val = a2val * elites.head(1).t

        val hxval: DenseMatrix[Double] = sigmoid(z3val)
        val pred = argmax(hxval(*, ::))
        val eq = pred :== y1val
        val m: Double = mean(eq.map(x => if (x) 1.0d else 0))

        println(s"Accuracy = ${m * 100}")
      }


      val tmpIndividuals = selectionTournament(r, costs, individuals)
      for (j <- 0 until NUM_OF_INDIVIDUAL; k <- 0 until NUM_OF_LAYER - 1) {
        individuals(j)(k) = tmpIndividuals(j)(k)
      }

      // 交叉
      for (j <- 0 until NUM_OF_INDIVIDUAL / 2; k <- 0 until NUM_OF_LAYER - 1) {
        if (r.nextDouble() < CROSSING_RATE) {
          val minMat = min(individuals(j * 2)(k), individuals(j * 2 + 1)(k)) - ALPHA * abs(individuals(j * 2)(k) - individuals(j * 2 + 1)(k))
          val maxMat = max(individuals(j * 2)(k), individuals(j * 2 + 1)(k)) + ALPHA * abs(individuals(j * 2)(k) - individuals(j * 2 + 1)(k))

          individuals(j * 2)(k) := minMat + (DenseMatrix.rand[Double](NETWORK_SHAPE(k)._1, NETWORK_SHAPE(k)._2) :* (maxMat - minMat))
          individuals(j * 2 + 1)(k) := minMat + (DenseMatrix.rand[Double](NETWORK_SHAPE(k)._1, NETWORK_SHAPE(k)._2) :* (maxMat - minMat))
        }
      }

      // 突然変異
      for (j <- 0 until NUM_OF_INDIVIDUAL; k <- 0 until NUM_OF_LAYER - 1) {
        if (r.nextDouble() < MUTATION_RATE) {
          val x = r.nextInt(NETWORK_SHAPE(k)._1)
          val y = r.nextInt(NETWORK_SHAPE(k)._2)
          individuals(j)(k)(x, y) += r.nextDouble() - 0.5

        }
      }


      for (j <- 0 until eliteCount) {
        individuals(j) = elites(j)
      }
      for (j <- savedEliteQueue.indices) {
        individuals(j + eliteCount) = savedEliteQueue(j).clone().map(_.copy)
      }

    }
  }

  def getElites(costs: Array[Double], individuals: Array[Array[DenseMatrix[Double]]]): (Array[Array[DenseMatrix[Double]]], Array[Double], Int) = {
    val sorted = costs.zipWithIndex.sortBy {
      case (c, _) => c
    }.map {
      case (c, index) => individuals(index) -> c
    }
    val elites = Array.ofDim[DenseMatrix[Double]](NUM_OF_ELITE, NUM_OF_LAYER - 1)
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
    val tmpIndividuals = Array.ofDim[DenseMatrix[Double]](NUM_OF_INDIVIDUAL, NUM_OF_LAYER - 1)

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
