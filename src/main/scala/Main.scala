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
        if (meta(0).text.split(" ")(1).split("m").isEmpty) return
        val distance = meta(0).text.split(" ")(1).split("m")(0).toInt * 1000.0
        val course = meta(0).text.split(" ")(0).split("・")(0) match {
          case "芝" => 100
          case "ダート" => 0
          case _ => Double.NaN
        }
        val weather = (meta \\ "img")(0).attribute("alt").get.text match {
          case "晴" => 30
          case "曇" => 20
          case "小雨" => 10
          case "雨" => 0
          case _ => Double.NaN
        }
        val field = (meta \\ "img")(1).attribute("alt").get.text match {
          case "良" => 3
          case "稍重" => 2
          case "重" => 1
          case "不良" => 0
          case _ => Double.NaN
        }
        val id = distance + course + weather + field
        val tr = (xml \\@ ("table", "id", "resultLs") \\ "tr").filter(node => (node \ "td").length > 0)
        val td = tr.toList.map(_ \ "td")
        td.foreach { td =>
          val a = td(3) \ "a"
          if (a.nonEmpty && !id.equals(Double.NaN)) {
            val list = List(name, td(0).text.trim, a.head.attribute("href").get.text.split("/")(3), td(13).text.trim)
            pw.println(list.mkString(","))
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