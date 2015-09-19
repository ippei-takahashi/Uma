import java.io._
import scala.collection.parallel.immutable.ParSeq
import scala.util.Random
import breeze.linalg._
import breeze.numerics._
import breeze.stats._

object Main {
  private[this] val NUM_OF_GENE_LOOP = 1001

  private[this] val NUM_OF_GENE = 50
  private[this] val NUM_OF_ELITE = 2

  private[this] val DATA_RATE = 0.8

  private[this] val CROSSOVER_RATE = 0.8
  private[this] val MUTATION_RATE = 0.3

  private[this] val ALPHA = 0.5

  private[this] val STATE_SIZE = 8

  type Gene = DenseVector[Double]

  case class Data(x: DenseVector[Double], y: Double)

  def main(args: Array[String]) {
    val r = new Random()

    val dataCSV = new File("data.csv")

    val data: DenseMatrix[Double] = csvread(dataCSV)
    val dataT = data(::, 1 until data.cols - 1).t
    val dataMean: DenseVector[Double] = mean(dataT(*, ::))
    val dataStd: DenseVector[Double] = stddev(dataT(*, ::))
    val dataNorm: DenseMatrix[Double] = dataT.copy
    val size = data.rows
    for (i <- 0 until size) {
      dataNorm(::, i) := (dataNorm(::, i) :- dataMean) :/ dataStd
    }

    val newData: DenseMatrix[Double] =
      DenseMatrix.horzcat(data(::, 0).toDenseMatrix.t, dataNorm.t, data(::, data.cols - 1).toDenseMatrix.t)

    val array = Array.ofDim[DenseVector[Double]](size)
    for (i <- 0 until size) {
      array(i) = newData(i, ::).t
    }

    val newArray = array.groupBy {
      vector => vector(4)
    }.collect {
      case (_, vectors) if vectors.length > 300 =>
        vectors
    }.flatten.toArray

    val group = Random.shuffle(newArray.groupBy(_(0)).values.toList.map(_.reverseMap { d =>
      new Data(d(1 until data.cols - 1), d(data.cols - 1))
    }.toList)).par

    val groupSize = group.length
    val trainSize = (DATA_RATE * groupSize).toInt

    val trainDataPar = group.slice(0, trainSize)
    val trainData = trainDataPar.toList
    val testData = group.slice(trainSize, groupSize)

    val raceMap: Map[Double, (Double, Double)] = trainData.flatten.groupBy {
      case Data(x, y) =>
        makeRaceId(x)
    }.map {
      case (idx, arr) =>
        val times = arr.map(_.y)
        idx ->(mean(times), stddev(times))
    }

    val genes = Array.ofDim[Gene](NUM_OF_GENE)

    for (i <- 0 until NUM_OF_GENE) {
      genes(i) = makeRandomGene(r)
    }

    for (loop <- 0 until NUM_OF_GENE_LOOP) {
      val costs = calcCost(dataStd, raceMap, trainDataPar, trainSize, genes)

      val (newCosts, newGenes) = costs.zip(genes).collect {
        case (cost, gene) if !cost.equals(Double.NaN) => (cost, gene)
      }.unzip

      val (elites, sortedCosts) = getElites(newCosts, newGenes)

      val tmpThetaArray = selectionTournament(r, newCosts, newGenes)

      crossover(r, CROSSOVER_RATE, tmpThetaArray)
      mutation(r, MUTATION_RATE, tmpThetaArray)

      for (j <- 0 until NUM_OF_ELITE) {
        tmpThetaArray(j) = elites(j)
      }

      for (j <- 0 until NUM_OF_GENE) {
        genes(j) = tmpThetaArray(j)
      }

      if (loop % 10 == 0) {
        val gene = elites.head
        val errorsOne = testData.filter(_.length == 1).map { dataList =>
          calcDataListCost(dataStd, raceMap, dataList, (x, y) => Math.abs(x - y), gene)
        }.toArray

        val errors = testData.filter(_.length > 1).groupBy { x =>
          makeRaceId(x.last.x :* dataStd :+ dataMean)
        }.collect {
          case (_, vector) if vector.length > 10 =>
            vector
        }.flatten.toArray.map { dataList =>
          calcDataListCost(dataStd, raceMap, dataList, (x, y) => Math.abs(x - y), gene)
        }

//        if (loop % 5 == 0) {
//          testData.filter(_.length > 1).groupBy { x =>
//            makeRaceId(x.last.x :* dataStd :+ dataMean)
//          }.foreach { case (idx, data) =>
//            val errors = data.map {
//              dataList =>
//                calcDataListCost(dataStd, raceMap, dataList, (x, y) => Math.abs(x - y), gene)
//            }.toArray
//
//            val errorMean = mean(errors)
//            val errorStd = stddev(errors)
//
//            println(s"idx$idx: ErrorMean = $errorMean, ErrorStd = $errorStd")
//
//          }
//        }

        val errorOneMean = mean(errorsOne)
        val errorOneStd = stddev(errorsOne)

        val errorMean = mean(errors)
        val errorStd = stddev(errors)

        val cost1 = sortedCosts.head
        val cost2 = sortedCosts(4)
        val cost3 = sortedCosts(10)
        val cost4 = sortedCosts(20)

        println(s"LOOP$loop: ErrorMean = $errorMean, ErrorStd = $errorStd, ErrorOneMean = $errorOneMean, ErrorOneStd = $errorOneStd, cost1 = $cost1, cost2 = $cost2, cost3 = $cost3, cost4 = $cost4")

        if (loop % 100 == 0) {
          csvwrite(new File(s"result_$loop.csv"), elites.head.toDenseMatrix)
        }
      }
    }
  }

  def findNearest(raceMap: Map[Double, (Double, Double)], vector: DenseVector[Double]): (Double, Double) = {
    val raceId = makeRaceId(vector)
    raceMap.minBy {
      case (idx, value) =>
        Math.abs(raceId - idx)
    }._2
  }

  def prePredict(raceMap: Map[Double, (Double, Double)], stdScore: Double, vector: DenseVector[Double]): Double = {
    val (m, s) = raceMap.getOrElse(makeRaceId(vector), findNearest(raceMap, vector))
    stdScore * s + m
  }

  def calcStdScore(raceMap: Map[Double, (Double, Double)], d: Data): Double = {
    val (m, s) = raceMap.getOrElse(makeRaceId(d.x), findNearest(raceMap, d.x))
    if (s == 0.0) {
      0
    } else {
      (d.y - m) / s
    }
  }

  def predict(prevScores: List[(Double, DenseVector[Double])],
              dataStd: DenseVector[Double],
              raceMap: Map[Double, (Double, Double)],
              d: Data,
              gene: Gene): (List[(Double, DenseVector[Double])], Double) = {
    val score = (calcStdScore(raceMap, d), d.x) :: prevScores
    val p = prevScores.foldLeft((0.0, 0.0)) {
      case ((scores, weights), (s, vector)) =>
        val distInv = 1.0 / vectorDistance(dataStd, d.x, vector, gene)
        (scores + s * distInv, weights + distInv)
    }
    val out = prePredict(raceMap, if (prevScores.isEmpty) 0.0 else p._1 / p._2, d.x)

    (score, out)
  }

  def calcDataListCost(dataStd: DenseVector[Double],
                       raceMap: Map[Double, (Double, Double)],
                       dataList: List[Data],
                       costFunction: (Double, Double) => Double,
                       gene: Gene) =
    dataList.foldLeft((Nil: List[(Double, DenseVector[Double])], 0.0)) {
      case ((prevScores, _), d) =>
        val (scores, out) = predict(prevScores, dataStd, raceMap, d, gene)
        val cost = costFunction(d.y, out)
        (scores, cost)
    }._2


  def calcCost(dataStd: DenseVector[Double],
               raceMap: Map[Double, (Double, Double)],
               trainData: ParSeq[List[Data]],
               trainSize: Double,
               genes: Array[Gene]): Array[Double] =
    genes.map { gene =>
      trainData.map { dataList =>
        calcDataListCost(dataStd, raceMap, dataList, (x, y) => Math.pow(x - y, 2.0), gene)
      }.sum / trainSize
    }


  def getElites(costs: Array[Double], genes: Array[Gene]): (Array[Gene], Array[Double]) = {
    val sorted = costs.zipWithIndex.sortBy {
      case (c, _) => c
    }.map {
      case (c, index) => genes(index) -> c
    }

    val elites = sorted.slice(0, NUM_OF_ELITE).map(_._1.copy)
    val sortedCosts = sorted.map(_._2)

    (elites, sortedCosts)
  }

  def selectionTournament(r: Random, costs: Array[Double], genes: Array[Gene]): Array[Gene] = {
    val tmpGenes = Array.ofDim[Gene](NUM_OF_GENE)

    for (j <- 0 until NUM_OF_GENE) {
      val a = r.nextInt(genes.length)
      val b = r.nextInt(genes.length)
      if (costs(a) < costs(b)) {
        tmpGenes(j) = genes(a).copy
      } else {
        tmpGenes(j) = genes(b).copy
      }
    }
    tmpGenes
  }

  def crossover(r: Random, crossoverRate: Double, tmpGenes: Array[Gene]) {
    for (j <- 0 until NUM_OF_GENE / 2) {
      if (r.nextDouble() < crossoverRate) {
        val minMat = max(min(tmpGenes(j * 2), tmpGenes(j * 2 + 1)) - ALPHA * abs(tmpGenes(j * 2) - tmpGenes(j * 2 + 1)), DenseVector.zeros[Double](STATE_SIZE))
        val maxMat = max(tmpGenes(j * 2), tmpGenes(j * 2 + 1)) + ALPHA * abs(tmpGenes(j * 2) - tmpGenes(j * 2 + 1))

        tmpGenes(j * 2) := minMat + (DenseVector.rand[Double](STATE_SIZE) :* (maxMat - minMat))
        tmpGenes(j * 2 + 1) := minMat + (DenseVector.rand[Double](STATE_SIZE) :* (maxMat - minMat))
      }
    }
  }

  def mutation(r: Random, mutationRate: Double, tmpGenes: Array[Gene]) {
    for (j <- 0 until NUM_OF_GENE) {
      val mask = DenseVector.rand[Double](STATE_SIZE).map { x =>
        if (x < mutationRate) 1.0 else 0.0
      }
      val update = 2.0 * DenseVector.rand[Double](STATE_SIZE) - 1.0

      val newGenes = max(tmpGenes(j) :+ mask :* update, DenseVector.zeros[Double](STATE_SIZE))

      tmpGenes(j) = newGenes
    }
  }

  def makeRandomGene(r: Random): Gene =
    DenseVector.rand[Double](STATE_SIZE)

  def vectorDistance(
                      dataStd: DenseVector[Double],
                      vector1: DenseVector[Double],
                      vector2: DenseVector[Double],
                      gene: Gene): Double = {
    0.01 +
      Math.abs(vector1(3) - vector2(3)) * gene(0) +
      Math.abs(vector1(0) - vector2(0)) * gene(1) +
      Math.abs(vector1(12) - vector2(12)) * gene(2) +
      Math.abs(vector1(13) - vector2(13)) * gene(3) +
      (if (vector1(15) != vector2(15) || vector1(16) != vector2(16) || vector1(17) != vector2(17) || vector1(18) != vector2(18)) 1.0 else 0.0) * gene(4) +
      (if (vector1(1) != vector2(1) || vector1(2) != vector2(2)) 1.0 else 0.0) * gene(5) +
      (if (vector1(4) != vector2(4) || vector1(5) != vector2(5) || vector1(6) != vector2(6) || vector1(7) != vector2(7)) 1.0 else 0.0) * gene(6) +
      (if (vector1(8) != vector2(8) || vector1(9) != vector2(9) || vector1(10) != vector2(10) || vector1(11) != vector2(11)) 1.0 else 0.0) * gene(7)
  }

  def makeRaceId(vector: DenseVector[Double]): Double =
    vector(3) * 1000 + vector(1) * 100 + vector(4) * 30 + vector(5) * 20 + vector(6) * 10 + vector(8) * 3 + vector(9) * 2 + vector(10)

}