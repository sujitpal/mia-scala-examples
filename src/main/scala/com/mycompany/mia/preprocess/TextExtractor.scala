package com.mycompany.mia.preprocess

import java.io.{File, FileInputStream, FileWriter, InputStream, PrintWriter}

import scala.collection.JavaConversions.asScalaIterator

import org.apache.commons.io.{FileUtils, FilenameUtils, IOUtils}
import org.apache.commons.io.filefilter.{DirectoryFileFilter, WildcardFileFilter}
import org.apache.tika.exception.TikaException
import org.apache.tika.metadata.Metadata
import org.apache.tika.parser.{AutoDetectParser, ParseContext, Parser}
import org.apache.tika.parser.audio.AudioParser
import org.apache.tika.parser.html.HtmlParser
import org.apache.tika.parser.image.ImageParser
import org.apache.tika.parser.microsoft.OfficeParser
import org.apache.tika.parser.opendocument.OpenOfficeParser
import org.apache.tika.parser.pdf.PDFParser
import org.apache.tika.parser.rtf.RTFParser
import org.apache.tika.parser.txt.TXTParser
import org.apache.tika.parser.xml.XMLParser
import org.apache.tika.sax.WriteOutContentHandler

object TextExtractor extends App {
  val extractor = new TextExtractor()
  val idir = new File("/path/to/raw/files")
  val odir = new File("/path/to/text/files")
  extractor.extractDirToFiles(idir, odir, null)
}

class TextExtractor {

  object FileType extends Enumeration {
    type FileType = Value
    val Text, Html, Xml, Pdf, Rtf, OOText,
    MsExcel, MsWord, MsPowerpoint, MsOutlook, Visio,
    Png, Jpeg, Mp3, Undef = Value
  }
  object DocPart extends Enumeration {
    type DocPart = Value
    val Title, Author, Body, Error = Value
  }
  
  val parsers = Map[FileType.Value,Parser](
    (FileType.Text, new TXTParser()),
    (FileType.Html, new HtmlParser()),
    (FileType.Xml, new XMLParser()),
    (FileType.Pdf, new PDFParser()),
    (FileType.Rtf, new RTFParser()),
    (FileType.OOText, new OpenOfficeParser()),
    (FileType.MsExcel, new OfficeParser()),
    (FileType.MsWord, new OfficeParser()),
    (FileType.MsPowerpoint, new OfficeParser()),
    (FileType.MsOutlook, new OfficeParser()),
    (FileType.Visio, new OfficeParser()),
    (FileType.Png, new ImageParser()),
    (FileType.Jpeg, new ImageParser()),
    (FileType.Mp3, new AudioParser()),
    (FileType.Undef, new AutoDetectParser())
  )

  /** Extract single file into map of name value pairs */
  def extract(file: File): Map[DocPart.Value,String] = {
    var istream: InputStream = null
    try {
      istream = new FileInputStream(file)
      val handler = new WriteOutContentHandler(-1)
      val metadata = new Metadata()
      val parser = parsers(detectFileType(file))
      val ctx = new ParseContext()
      parser.parse(istream, handler, metadata, ctx)
      Map[DocPart.Value,String](
        (DocPart.Author, metadata.get(Metadata.CREATOR)),
        (DocPart.Title, metadata.get(Metadata.TITLE)),
        (DocPart.Body, handler.toString))
    } catch {
      case e: TikaException => Map[DocPart.Value,String](
        (DocPart.Error, e.getMessage()))      
    } finally {
      IOUtils.closeQuietly(istream)
    }
  }
  
  /** Detect FileType based on file name suffix */
  def detectFileType(file: File): FileType.Value = {
    val suffix = FilenameUtils.getExtension(file.getName()).
      toLowerCase()
    suffix match {
      case "text" | "txt" => FileType.Text
      case "html" | "htm" => FileType.Html
      case "xml"          => FileType.Xml
      case "pdf"          => FileType.Pdf
      case "rtf"          => FileType.Rtf
      case "odt"          => FileType.OOText
      case "xls" | "xlsx" => FileType.MsExcel
      case "doc" | "docx" => FileType.MsWord
      case "ppt" | "pptx" => FileType.MsPowerpoint
      case "pst"          => FileType.MsOutlook
      case "vsd"          => FileType.Visio
      case "png"          => FileType.Png
      case "jpg" | "jpeg" => FileType.Jpeg
      case "mp3"          => FileType.Mp3
      case _              => FileType.Undef
    }
  }
  
  /** Extract all files in directory with specified file name
      pattern. Accepts a renderer function to convert name-
      value pairs into an output file (or files).*/
  def extract(dir: File, pattern: String, odir: File,
      renderer: (File, File, Map[DocPart.Value,String]) => Unit): 
      Unit = {
    val filefilter = pattern match {
      case null => new WildcardFileFilter("*.*")
      case _ => new WildcardFileFilter(pattern)
    }
    FileUtils.iterateFiles(dir, filefilter, 
        DirectoryFileFilter.DIRECTORY).foreach(file => {
      Console.println("Parsing file: " + file.getName())
      val data = extract(file)
      renderer(file, odir, data)
    })
  }

  /** Convenience method to write out text extracted from a file
      into the specified directory as filename.txt */
  def extractDirToFiles(dir: File, odir: File, pattern: String): Unit = {
    def renderDirToFiles(file: File, odir: File, 
        data: Map[DocPart.Value,String]): Unit = {
      val ofname = file.getName() + ".txt"
      val writer = new PrintWriter(new FileWriter(new File(odir, ofname)), true)
      writer.println(data(DocPart.Title))
      writer.println(data(DocPart.Author))
      writer.println(data(DocPart.Body))
      writer.flush()
      writer.close()
    } 
    extract(dir, pattern, odir, renderDirToFiles)
  }
}