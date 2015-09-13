import java.io._
import scala.collection.parallel.immutable.ParSeq
import scala.util.Random
import breeze.linalg._
import breeze.numerics._
import breeze.stats._

object Main {
  private[this] val NUM_OF_GENE_LOOP = 1001

  private[this] val NUM_OF_GENE = 21
  private[this] val NUM_OF_ELITE = 1

  private[this] val DATA_RATE = 0.8

  private[this] val CROSSOVER_RATE = 0.6
  private[this] val MUTATION_RATE = 0.001

  private[this] val ALPHA = 0.5

  private[this] val BATCH_SIZE = 3000

  def main(args: Array[String]) {
    case class Data(x: DenseVector[Double], y: Double)

    trait Node[A] {
      def eval(state: Double, env: DenseVector[Double]): A
    }

    trait DoubleNode extends Node[Double] {
      def eval(state: Double, env: DenseVector[Double]): Double
    }

    trait BooleanNode extends Node[Boolean] {
      def eval(state: Double, env: DenseVector[Double]): Boolean
    }

    val r = new Random()

    val dataCSV = new File("data.csv")

    val data: DenseMatrix[Double] = csvread(dataCSV)

    val size = data.rows

    val array = Array.ofDim[DenseVector[Double]](size)
    for (i <- 0 until size) {
      array(i) = data(i, ::).t
    }

    val newArray = array.groupBy {
      vector => vector(4)
    }.collect {
      case (_, vectors) if vectors.length > 300 =>
        vectors
    }.flatten.toArray

    val group = Random.shuffle(newArray.groupBy(_(0)).values.toList.map(_.reverseMap { d =>
      new Data(d(1 until data.cols - 16), d(data.cols - 1))
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

    val genes = Array.ofDim[DoubleNode](NUM_OF_GENE)

    for (i <- 0 until NUM_OF_GENE) {
      genes(i) = makeRandomDoubleNode
    }

    for (loop <- 0 until NUM_OF_GENE_LOOP) {
      val costs = calcCost(trainDataPar, trainSize, genes)

      val (newCosts, newGenes) = costs.zip(genes).collect {
        case (cost, gene) if !cost.equals(Double.NaN) => (cost, gene)
      }.unzip

      val (elites, sortedCosts) = getElites(newCosts, newGenes)

      val tmpThetaArray = selectionTournament(r, newCosts, newGenes)

      //      crossover(r, CROSSOVER_RATE, tmpThetaArray)
      //      mutation(MUTATION_RATE, tmpThetaArray)
      //
      //      for (j <- 0 until NUM_OF_ELITE) {
      //        tmpThetaArray(j) = elites(j)
      //      }
      //
      //      for (j <- 0 until NUM_OF_GENE) {
      //        genes(j) = tmpThetaArray(j)
      //      }
      //
      if (loop % 1 == 0) {
        val gene = elites.head
        val errorsOne = testData.filter(_.length == 1).map { dataArray =>
          dataArray.foldLeft((Nil: List[Double], 0.0)) {
            case ((prevState, _), d) =>
              val (state, out) = predict(prevState, d, gene)
              val cost = Math.abs(d.y - out)
              (state, cost)
          }._2
        }.toArray

        val errors = testData.filter(_.length != 1).map { dataArray =>
          dataArray.foldLeft((Nil: List[Double], 0.0)) {
            case ((prevState, _), d) =>
              val (state, out) = predict(prevState, d, gene)
              val cost = Math.abs(d.y - out)
              (state, cost)
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

    def findNearest(vector: DenseVector[Double]): (Double, Double) = {
      val raceId = makeRaceId(vector)
      raceMap.minBy {
        case (idx, value) =>
          Math.abs(raceId - idx)
      }._2
    }

    def prePredict(stdScore: Double, vector: DenseVector[Double]): Double = {
      val (m, s) = raceMap.getOrElse(makeRaceId(vector), findNearest(vector))
      stdScore * s + m
    }

    def calcStdScore(d: Data): Double = {
      val (m, s) = raceMap.getOrElse(makeRaceId(d.x), findNearest(d.x))
      if (s == 0.0) {
        0
      } else {
        (d.y - m) / s
      }
    }

    def predict(prevState: List[Double],
                d: Data,
                gene: DoubleNode): (List[Double], Double) = {
      val state = calcStdScore(d) :: prevState
      val out = gene.eval(if (prevState.isEmpty) 0.0 else mean(prevState), d.x)

      (state, out)
    }

    def calcCost(trainData: ParSeq[List[Data]],
                 trainSize: Double,
                 genes: Array[DoubleNode]): Array[Double] =
      genes.map { gene =>
        trainData.map { dataArray =>
          dataArray.foldLeft((Nil: List[Double], 0.0)) {
            case ((prevState, _), d) =>
              val (state, out) = predict(prevState, d, gene)
              val cost = Math.pow(d.y - out, 2.0)
              (state, cost)
          }._2
        }.sum / trainSize
      }


    def getElites(costs: Array[Double], genes: Array[DoubleNode]): (Array[DoubleNode], Array[Double]) = {
      val sorted = costs.zipWithIndex.sortBy {
        case (c, _) => c
      }.map {
        case (c, index) => genes(index) -> c
      }

      val elites = sorted.slice(0, NUM_OF_ELITE).map(_._1) // clone
      val sortedCosts = sorted.map(_._2)

      (elites, sortedCosts)
    }

    def selectionTournament(r: Random, costs: Array[Double], genes: Array[DoubleNode]): Array[DoubleNode] = {
      val tmpGenes = Array.ofDim[DoubleNode](NUM_OF_GENE)

      for (j <- 0 until NUM_OF_GENE) {
        val a = r.nextInt(genes.length)
        val b = r.nextInt(genes.length)
        if (costs(a) < costs(b)) {
          tmpGenes(j) = genes(a) // clone
        } else {
          tmpGenes(j) = genes(b) // clone
        }
      }
      tmpGenes
    }

    //  def crossover(r: Random, crossoverRate: Double, tmpThetaArray: Array[Array[DoubleNode]]) {
    //    for (j <- 0 until NUM_OF_GENE / 2; k <- 0 until NUM_OF_MAT) {
    //      if (r.nextDouble() < crossoverRate) {
    //        val minMat = min(tmpThetaArray(j * 2)(k), tmpThetaArray(j * 2 + 1)(k)) - ALPHA * abs(tmpThetaArray(j * 2)(k) - tmpThetaArray(j * 2 + 1)(k))
    //        val maxMat = max(tmpThetaArray(j * 2)(k), tmpThetaArray(j * 2 + 1)(k)) + ALPHA * abs(tmpThetaArray(j * 2)(k) - tmpThetaArray(j * 2 + 1)(k))
    //
    //        tmpThetaArray(j * 2)(k) := minMat + (DenseMatrix.rand[Double](NETWORK_SHAPE(k)._1, NETWORK_SHAPE(k)._2) :* (maxMat - minMat))
    //        tmpThetaArray(j * 2 + 1)(k) := minMat + (DenseMatrix.rand[Double](NETWORK_SHAPE(k)._1, NETWORK_SHAPE(k)._2) :* (maxMat - minMat))
    //      }
    //    }
    //  }
    //
    //  def mutation(mutationRate: Double, tmpThetaArray: Array[Array[DoubleNode]]) {
    //    for (j <- 0 until NUM_OF_GENE; k <- 0 until NUM_OF_MAT) {
    //      val mask = DenseMatrix.rand[Double](NETWORK_SHAPE(k)._1, NETWORK_SHAPE(k)._2).map { x =>
    //        if (x < mutationRate) 1.0 else 0.0
    //      }
    //      val update = 2.0 * DenseMatrix.rand[Double](NETWORK_SHAPE(k)._1, NETWORK_SHAPE(k)._2) - 1.0
    //
    //      tmpThetaArray(j)(k) += mask :* update
    //    }
    //  }

    case class DoubleValue(value: Double) extends DoubleNode {
      def eval(state: Double, env: DenseVector[Double]): Double = value
    }


    case object Predict extends DoubleNode {
      def eval(state: Double, env: DenseVector[Double]): Double =
        prePredict(state, env)
    }


    case object Age extends DoubleNode {
      def eval(state: Double, env: DenseVector[Double]): Double = env(0)
    }

    case object BasisWeight extends DoubleNode {
      def eval(state: Double, env: DenseVector[Double]): Double = env(12)
    }

    case object Weight extends DoubleNode {
      def eval(state: Double, env: DenseVector[Double]): Double = env(13)
    }


    case object IsTurf extends BooleanNode {
      def eval(state: Double, env: DenseVector[Double]): Boolean = env(1) > 0
    }

    case object IsDirt extends BooleanNode {
      def eval(state: Double, env: DenseVector[Double]): Boolean = env(2) > 0
    }


    case object IsGood extends BooleanNode {
      def eval(state: Double, env: DenseVector[Double]): Boolean = env(4) > 0
    }

    case object IsSlightlyHeavy extends BooleanNode {
      def eval(state: Double, env: DenseVector[Double]): Boolean = env(5) > 0
    }

    case object isHeavy extends BooleanNode {
      def eval(state: Double, env: DenseVector[Double]): Boolean = env(6) > 0
    }

    case object IsBad extends BooleanNode {
      def eval(state: Double, env: DenseVector[Double]): Boolean = env(7) > 0
    }


    case object IsSunny extends BooleanNode {
      def eval(state: Double, env: DenseVector[Double]): Boolean = env(8) > 0
    }

    case object IsCloudy extends BooleanNode {
      def eval(state: Double, env: DenseVector[Double]): Boolean = env(9) > 0
    }

    case object IsLightRainy extends BooleanNode {
      def eval(state: Double, env: DenseVector[Double]): Boolean = env(10) > 0
    }

    case object IsRainy extends BooleanNode {
      def eval(state: Double, env: DenseVector[Double]): Boolean = env(11) > 0
    }


    case class IfElse(cond: BooleanNode, ifThen: Double, elseThen: Double) extends DoubleNode {
      def eval(state: Double, env: DenseVector[Double]): Double =
        if (cond.eval(state, env))
          ifThen
        else
          elseThen
    }


    case class Plus(x: DoubleNode, y: DoubleNode) extends DoubleNode {
      def eval(state: Double, env: DenseVector[Double]): Double = x.eval(state, env) + y.eval(state, env)
    }

    case class Multiply(x: DoubleNode, y: DoubleNode) extends DoubleNode {
      def eval(state: Double, env: DenseVector[Double]): Double = x.eval(state, env) * y.eval(state, env)
    }

    case class Power(x: DoubleNode, y: Double) extends DoubleNode {
      def eval(state: Double, env: DenseVector[Double]): Double = Math.pow(Math.abs(x.eval(state, env)), y)
    }


    case class Minus(x: DoubleNode) extends DoubleNode {
      def eval(state: Double, env: DenseVector[Double]): Double = -x.eval(state, env)
    }

    case class Inv(x: DoubleNode) extends DoubleNode {
      def eval(state: Double, env: DenseVector[Double]): Double = {
        val xValue = x.eval(state, env)
        if (xValue == 0.0) Double.MaxValue else 1.0 / xValue
      }
    }


    case class And(x: BooleanNode, y: BooleanNode) extends BooleanNode {
      def eval(state: Double, env: DenseVector[Double]): Boolean = x.eval(state, env) && y.eval(state, env)
    }

    case class Or(x: BooleanNode, y: BooleanNode) extends BooleanNode {
      def eval(state: Double, env: DenseVector[Double]): Boolean = x.eval(state, env) || y.eval(state, env)
    }


    case class Not(x: BooleanNode) extends BooleanNode {
      def eval(state: Double, env: DenseVector[Double]): Boolean = !x.eval(state, env)
    }

    def makeRandomDoubleNode: DoubleNode = {
      r.nextInt(11) match {
        case 0 => DoubleValue(r.nextDouble() * 2.0 - 1.0)
        case 1 => Predict
        case 2 => Age
        case 3 => BasisWeight
        case 4 => Weight
        case 5 => IfElse(makeRandomBooleanNode, r.nextDouble() * 2.0 - 1.0, r.nextDouble() * 2.0 - 1.0)
        case 6 => Plus(makeRandomDoubleNode, makeRandomDoubleNode)
        case 7 => Multiply(makeRandomDoubleNode, makeRandomDoubleNode)
        case 8 => Power(makeRandomDoubleNode, r.nextDouble() * 2.0 - 1.0)
        case 9 => Minus(makeRandomDoubleNode)
        case 10 => Inv(makeRandomDoubleNode)
      }
      Predict
    }

    def makeRandomBooleanNode: BooleanNode = {
      r.nextInt(13) match {
        case 0 => IsTurf
        case 1 => IsDirt
        case 2 => IsGood
        case 3 => IsSlightlyHeavy
        case 4 => isHeavy
        case 5 => IsBad
        case 6 => IsSunny
        case 7 => IsCloudy
        case 8 => IsLightRainy
        case 9 => IsRainy
        case 10 => And(makeRandomBooleanNode, makeRandomBooleanNode)
        case 11 => Or(makeRandomBooleanNode, makeRandomBooleanNode)
        case 12 => Not(makeRandomBooleanNode)
      }
    }
  }

  def makeRaceId(vector: DenseVector[Double]): Double = {
    vector(3) * 1000 + vector(1) * 100 + vector(4) * 30 + vector(5) * 20 + vector(6) * 10 + vector(8) * 3 + vector(9) * 2 + vector(10)
  }
}