import java.io._
import scala.xml._
import scala.xml.parsing.NoBindingFactoryAdapter
import nu.validator.htmlparser.sax.HtmlParser
import nu.validator.htmlparser.common.XmlViolationPolicy
import org.xml.sax.InputSource

object Main {

  def main(args: Array[String]) {

    import XmlFilter._

    val raceDir = new File("race")
    val raceXMLs = raceDir.
      listFiles.
      toList.
      filter(_.getName.endsWith("html"))

    val outFile = new File("race.csv")
    val pw = new PrintWriter(outFile)

    try {
      raceXMLs.foreach { file =>
        val (name, xml) = file.getName.split("\\.").head -> toNode(file)
        val meta = xml \\@ ("p", "id", "raceTitMeta")
        val tr = (xml \\@ ("table", "id", "resultLs") \\ "tr").filter(node => (node \ "td").length > 0)
        val td = tr.toList.map(_ \ "td")
        val resultYen =  (xml \\@ ("table", "class", "resultYen") \\ "tr").filter(node => (node \ "td").length > 0)
        val yenTh = resultYen.toList.map(_ \ "th")
        val yenTd = resultYen.toList.map(_ \ "td")
        val yenT = yenTh.zip(yenTd)
        for {
          tan <- yenT.find(x => x._1.length > 0 && x._1(0).text == "単勝")
          fukWithIndex <- yenT.zipWithIndex.find(x => x._1._1.length > 0 && x._1._1(0).text == "複勝")
          fuk1 = fukWithIndex._1
          fuk2 = yenT(fukWithIndex._2 + 1)
          fuk3 = yenT(fukWithIndex._2 + 2)
          ren <- yenT.find(x => x._1.length > 0 && x._1(0).text == "馬連")
          reg = "\\d+".r
          tanYen <- reg.findFirstIn(tan._2(1).text.replaceAll(",", "").replaceAll("円", "")).map(_.toDouble / 100.0)
          fuk1Yen <- reg.findFirstIn(fuk1._2(1).text.replaceAll(",", "").replaceAll("円", "")).map(_.toDouble / 100.0)
          fuk2Yen <- reg.findFirstIn(fuk2._2(1).text.replaceAll(",", "").replaceAll("円", "")).map(_.toDouble / 100.0)
          fuk3Yen <- reg.findFirstIn(fuk3._2(1).text.replaceAll(",", "").replaceAll("円", "")).map(_.toDouble / 100.0)
          renYen <- reg.findFirstIn(ren._2(1).text.replaceAll(",", "").replaceAll("円", "")).map(_.toDouble / 100.0)
        } {
          td.foreach { td =>
            val a = td(3) \ "a"
            val rank = td(0).text.trim
            val num = td(2).text.trim
            val fukYen = num match {
              case n if n == fuk1._2(0).text => fuk1Yen
              case n if n == fuk2._2(0).text => fuk2Yen
              case n if n == fuk3._2(0).text => fuk3Yen
              case _ => 0.0
            }
            if (a.nonEmpty && !meta(0).text.startsWith("障害") && reg.findFirstIn(rank).nonEmpty) {
              val list = List(name, td(0).text.trim, a.head.attribute("href").get.text.split("/")(3), td(13).text.trim,
                tanYen, fukYen, renYen)
              pw.println(list.mkString(","))
            }
          }
        }
      }
    } catch {
      case ex: Exception =>
       ex.printStackTrace()
    } finally {
      pw.close()
    }
  }

  def toNode(file: File): Node = {
    val hp = new HtmlParser
    hp.setNamePolicy(XmlViolationPolicy.ALLOW)
    hp.setCommentPolicy(XmlViolationPolicy.ALLOW)

    val saxer = new NoBindingFactoryAdapter
    hp.setContentHandler(saxer)
    hp.parse(new InputSource(new FileReader(file)))

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
      that \\ nodeName attrFilter (attrName, value)
    }

    def \@(nodeName: String, attrName: String, value: String): NodeSeq = {
      that \ nodeName attrFilter (attrName, value)
    }
  }
}