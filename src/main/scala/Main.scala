import java.io._

import scala.concurrent.Await
import scala.concurrent.duration._

import dispatch._
import dispatch.Defaults._

object Main {

  val YEAR_START = 2008
  val YEAR_END = 2015

  val MONTH_START = 1
  val MONTH_END = 12

  val DAY_START = 1
  val DAY_END = 31

  val BABA_START = 1
  val BABA_END = 40

  val RACE_START = 1
  val RACE_END = 12

  def main(args: Array[String]): Unit = {
    for {
      year <- YEAR_START to YEAR_END
      month <- MONTH_START to MONTH_END
      day <- DAY_START to DAY_END
    } {
      val resF = Future.sequence {
        for {
          baba <- BABA_START to BABA_END
          race <- RACE_START to RACE_END
        } yield {
          val urlString =
            "http://www.keiba.go.jp/KeibaWeb/TodayRaceInfo/RaceList?k_raceDate=%04d/%02d/%02d&k_raceNo=%d&k_babaCode=%d".
              format(year, month, day, race, baba)
          val request = url(urlString)
          Http(request OK (r => r)).map {
            res => (baba, race, res)
          }
        }
      }
      val res = Await.result(resF, 60 seconds)
      res.foreach {
        case (baba, race, response) =>
        val file = new File("raceLocal/%04d%02d%02d-%d-%d.html".format(year, month, day, baba, race))
        val pw = new PrintWriter(new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file), "UTF-8")))
        try {
          pw.write(response.getResponseBody())
        } finally {
          pw.close()
        }
      }
    }
  }
}
/*
http://www.keiba.go.jp/KeibaWeb/DataRoom/RaceHorseList?k_flag=1&k_pageNum=0&k_horseName=&k_horsebelong=*&k_birthYear=2008&k_fatherHorse=&k_motherHorse=&k_activeCode=2&k_dataKind=1
 */