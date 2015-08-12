import scala.util.Random

object Main {
  def main(args: Array[String]) {

    val NUM_OF_INDIVIDUAL = 1000
    val NUM_OF_GENERATION = 100
    val NUM_OF_ELITE = 5
    val CROSSING_RATE = 0.95
    val MUTATION_RATE = 0.10

    val C_LEN = 3

    val r = new Random()

    val individuals = Array.ofDim[Double](NUM_OF_INDIVIDUAL, C_LEN)
    val c = Array.ofDim[Int](C_LEN)

    for {i <- 0 until NUM_OF_INDIVIDUAL; j <- 0 until C_LEN} {
      individuals(i)(j) = r.nextDouble() * 100
    }

    for {i <- 0 until C_LEN} {
      c(i) = r.nextInt() % 10
    }

    println(s"${c(2)}x^3 + ${c(1)}y^2 + ${c(0)}z = 0")


    for (i <- 0 until NUM_OF_GENERATION) {
      val scores = Array.ofDim[Double](NUM_OF_INDIVIDUAL)

      for (j <- 0 until NUM_OF_INDIVIDUAL; k <- 0 until C_LEN) {
        scores(j) += c(k) * Math.pow(individuals(j)(k), k + 1)
      }

      for (j <- 0 until NUM_OF_INDIVIDUAL) {
        scores(j) =
          if (scores(j) == 0)
            Double.MaxValue
          else
            1.0 / Math.abs(scores(j))
      }

      val total = scores.sum

      // 優秀な数匹は次の世代に持ち越し
      val elites = scores.zipWithIndex.sortBy {
        case ((s, _)) => -s
      }.map {
        case ((_, index)) => individuals(index).clone()
      }.slice(0, NUM_OF_ELITE)

      println(s"elite$i: value = ${scores.max}, x = ${elites(0)(2)}, y = ${elites(0)(1)}, z = ${elites(0)(0)}")

      val tmpindividuals = Array.ofDim[Double](NUM_OF_INDIVIDUAL, C_LEN)

      // 自然淘汰
      for (j <- 0 until NUM_OF_INDIVIDUAL) {
        val threshold = r.nextDouble() * total
        val num = scores.zipWithIndex.foldLeft((None: Option[Int]) -> 0.0) {
          case ((None, acc), (s, k)) =>
            if (acc + s > threshold)
              Some(k) -> acc
            else
              None -> (acc + s)
          case ((opt, acc), _) =>
            opt -> acc
        }._1.getOrElse(0)

        for (k <- 0 until C_LEN) {
          tmpindividuals(j)(k) = individuals(num)(k)
        }
      }

      for (j <- 0 until NUM_OF_INDIVIDUAL; k <- 0 until C_LEN) {
        individuals(j)(k) = tmpindividuals(j)(k)
      }

      // 交叉
      for (j <- 0 until NUM_OF_INDIVIDUAL / 2) {
        if (r.nextDouble() < CROSSING_RATE) {
          for (k <- r.nextInt(C_LEN) until C_LEN) {
            val temp = individuals(j * 2)(k)
            individuals(j * 2)(k) = individuals(j * 2 + 1)(k)
            individuals(j * 2 + 1)(k) = temp

          }
        }
      }

      // 突然変異
      for (j <- 0 until NUM_OF_INDIVIDUAL; k <- 0 until C_LEN) {
        if (r.nextDouble() < CROSSING_RATE) {
          individuals(j)(k) = r.nextDouble() * 100
        }
      }

      for (j <- 0 until NUM_OF_ELITE; k <- 0 until C_LEN) {
        individuals(j)(k) = elites(j)(k)
      }
    }
  }

}
