import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.common.Term;


public class ReverseIndex {
	static Segment segment = HanLP.newSegment();
	static JavaSparkContext sc =init(); 	// rdd的配置文件
	String filename;						//  要读取的评论文件  
	Long commentsNum; 		//  评论数
	List<String> vocabulary;   //　获取词汇
	
	public ReverseIndex(String filename){
		this.filename = filename;
		commentsNum = getCommentsNum();
		vocabulary = createDictionary();
	}
	
	public static JavaSparkContext init(){
		//spark参数文件的初始化
		SparkConf conf = new SparkConf();
		conf.set("spark.testing.memory", "3000000000");
		JavaSparkContext sc = new JavaSparkContext("local[4]", "Spark", conf);
		return sc;
	}
	
	public Long getCommentsNum(){
		Long  commentsNum= sc.textFile(filename).count();
		return commentsNum;
	}
	public List<String> createDictionary() {
		JavaRDD<String> comments  = sc.textFile(filename); 		//加载数据
		JavaRDD<String> segRDD = comments.map(new Seg()); 		//分词，并过滤部分实体词
		JavaRDD<String> termsRDD = segRDD.flatMap(new FlatOne()); 		//映射出每一个单词
		JavaPairRDD<String, Integer> wordPair = termsRDD.mapToPair(new MapOne());   // 将每个单词映射成键值对
		JavaPairRDD<String, Integer> reducePair = wordPair.reduceByKey(new Reduce()); 	//相同单词累积加和
		JavaPairRDD<String, Integer> frqPair = reducePair.filter(new FrqFilter(commentsNum));
		JavaRDD<String> vocabularyRDD = frqPair.keys();	//提取出词汇
		List<String> vocabulary = new ArrayList<String>(vocabularyRDD.collect());
		return vocabulary;
	}
	
	public void createReverseIndex(String reverseFile) {
		JavaRDD<String> comments = sc.textFile(filename);
		List<String> commentsList = comments.collect(); // 获取评论
		FileWriter fw = null;
		try {
			fw = new FileWriter(reverseFile);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		PrintWriter pw = new PrintWriter(fw);
		
		Map<String, Set<Long>> reverseIndex = new HashMap<String, Set<Long>>();
		for(String word: vocabulary){
			Set<Long> docid = new HashSet<Long>(); //　记录每个单词的倒排索引
			for(String comment: commentsList){
				if(comment.contains(word)){
					docid.add((long)commentsList.indexOf(comment));
				}
			}
			System.out.println(word + ":" + docid);
			pw.println(word + ":" + docid);
		}
		pw.close();
	}
	
	static class Seg implements Function<String, String>{
		public String call(String sentence) throws Exception {
			String segStr = ""; 	//分词后的拼接,　以｀分割
			List<Term> termList = segment.seg(sentence);  // 分词
			StringBuilder sb = new StringBuilder();
			for(Term term : termList){
				String str = term.toString();
				Boolean tagPos =  ( str.contains("n") ||  str.contains("v") );
				//过滤名，形，动，成语
				if(tagPos){
					sb.append(term.word + "`");
				}
			}
			segStr = sb.toString();
			return segStr;
		}
	}
	
	static class FlatOne implements java.io.Serializable, FlatMapFunction<String, String>{
		public Iterable<String> call(String segStr){
			return Arrays.asList(segStr.split("`"));		
		}
	}
	
	static class MapOne implements PairFunction<String, String , Integer>{
		public Tuple2<String, Integer> call (String terms){
			return new Tuple2<String, Integer>(terms, 1);
		}
	}
	
	static class Reduce implements Function2<Integer, Integer, Integer>{
		public Integer call(Integer i1, Integer i2){
			return i1 + i2;
		}
	} 
	
	static class FrqFilter implements Function<Tuple2<String, Integer>, Boolean>{
		Long commentsNum;
		public FrqFilter(Long commentsNum){
			this.commentsNum = commentsNum;
		}
		public Boolean call(Tuple2<String, Integer> data){
			String term = data._1;
			Long Frequency = (long)data._2;
			Long theta;
			if(commentsNum < 100){
				theta = (long) (commentsNum * 0.01);
			}
			else if(commentsNum < 1000){
				theta = (long) (commentsNum * 0.05);
			}
			else{
				theta = (long) 50;
			}
			Boolean tagWord = ( Frequency >= theta && term.length() > 1); // 过滤出频率为0.1以上，长度大于1的词语
			return tagWord;
		}
	}
	
	static List<String> showAllFiles(File dir) throws Exception{
		List<String> fileList = new ArrayList<String>();
		File[] fs = dir.listFiles();
		for(int i=0;i<fs.length;i++){
//			System.out.println(fs[i].getAbsolutePath());
			fileList.add(fs[i].getAbsolutePath());
		}
		return fileList;
	}
	
	public static void train(String commentsFile){
		String[] tokens = commentsFile.split("/");
		String id = tokens[tokens.length - 1];
		String reverseFile = "/home/quincy1994/实验室项目/影迷关注点分析/实验数据/ReverseIndex/" + id;
//		String reverseFile = "/home/quincy1994/实验室项目/影迷关注点分析/实验数据/ReverseIndex/" + "5645c95a756a5d75424ca124";
		File f = new File(reverseFile);
		if (f.exists()){
			System.out.println(reverseFile);
			return;
		}
		ReverseIndex  reIndex = new ReverseIndex(commentsFile);
		reIndex.createReverseIndex(reverseFile);
	}
	
	public static void main(String[] agrs) throws Exception {
		String fileStr = "/home/quincy1994/实验室项目/影迷关注点分析/实验数据/film_comments_txt";
		File fileDir = new File(fileStr);
		List<String> filedir = showAllFiles(fileDir);
		for(String commentsFile : filedir){
//			String cFile = "/home/quincy1994/实验室项目/影迷关注点分析/实验数据/film_comments_txt/5645c95a756a5d75424ca124.txt";
			train(commentsFile);
//			break;
		}
	}
	
}
