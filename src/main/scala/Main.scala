import java.io._

import scala.concurrent.Await
import scala.concurrent.duration._
import scala.xml._
import scala.xml.parsing.NoBindingFactoryAdapter

import nu.validator.htmlparser.sax.HtmlParser
import nu.validator.htmlparser.common.XmlViolationPolicy
import org.xml.sax.InputSource

import dispatch._
import dispatch.Defaults._

object Main {

  val YEAR_START = 2013
  val YEAR_END = 2013

  val PAGE_START = 0
  val PAGE_END = 39

  val BASE_URL = "http://www.keiba.go.jp"

  def main(args: Array[String]): Unit = {
    import XmlFilter._

    for {
      year <- YEAR_START to YEAR_END
      page <- PAGE_START to PAGE_END
    } {
      val urlString =
        s"$BASE_URL/KeibaWeb/DataRoom/RaceHorseList?k_flag=1&k_pageNum=%d&k_horseName=&k_horsebelong=*&k_birthYear=%d&k_fatherHorse=&k_motherHorse=&k_activeCode=2&k_dataKind=1".
          format(page, year)
      val request = url(urlString)
      val responseF = Http(request OK (r => r))

      val response = Await.result(responseF, 60 seconds)

      val reader = new BufferedReader(new InputStreamReader(response.getResponseBodyAsStream))
      val xml = toNode(reader)

      val tr = xml \\@("tr", "class", "dbnote")
      val hrefs = tr.toList.map(_ \ "td").map(_(1)).map(x => (x \\ "a")(0).attribute("href")).collect {
        case Some(seq) =>
          seq.head.text
      }

      val seqF = Future.sequence {
        hrefs.map {
          href =>
            val fileName = "raceLocal/" + href.split("k_lineageLoginCode=")(1).split("&").head + ".html"
            val req = url(s"$BASE_URL$href")

            Http(req OK (r => r)).map(fileName -> _)
        }
      }.map {
        _.foreach {
          case (fileName, res) =>
            val file = new File(fileName)
            val pw = new PrintWriter(new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file), "UTF-8")))

            try {
              pw.write(res.getResponseBody())
            } finally {
              pw.close()
            }
        }
      }

      val result = Await.result(seqF, 300 seconds)
      println(result)
    }
  }

  def toNode(reader: Reader): Node = {
    val hp = new HtmlParser
    hp.setNamePolicy(XmlViolationPolicy.ALLOW)
    hp.setCommentPolicy(XmlViolationPolicy.ALLOW)

    val saxer = new NoBindingFactoryAdapter
    hp.setContentHandler(saxer)
    hp.parse(new InputSource(reader))

    saxer.rootElem
  }

  object XmlFilter {
    implicit def nodeSeqToMyXmlFilter(nodeSeq: NodeSeq): XmlFilter =
      new XmlFilter(nodeSeq)
  }

  class XmlFilter(that: NodeSeq) {

    import XmlFilter._

    def attrFilter(name: String, value: String): NodeSeq = {
      that filter (_ \ ("@" + name) exists (_.text == value))
    }

    def \\@(nodeName: String, attrName: String, value: String): NodeSeq = {
      that \\ nodeName attrFilter(attrName, value)
    }

    def \@(nodeName: String, attrName: String, value: String): NodeSeq = {
      that \ nodeName attrFilter(attrName, value)
    }
  }

}