import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
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
import org.apache.spark.mllib.clustering.DistributedLDAModel;
import org.apache.spark.mllib.clustering.GaussianMixture;
import org.apache.spark.mllib.clustering.GaussianMixtureModel;
import org.apache.spark.mllib.clustering.LDA;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.rdd.RDD;

import scala.Tuple2;
import scala.Tuple3;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.common.Term;


public class MYLDA {
	
	static Segment segment=HanLP.newSegment();   //使用了HanLP分词工具
	
	JavaSparkContext sc;  //spark的相关配置对象
	String filename;   // 评论文件，每行对应一条评论
	
	public MYLDA(String filename){
		sc = this.init();
		this.filename = filename;
	}
	
	public JavaSparkContext init(){
		SparkConf conf = new SparkConf();
		conf.set("spark.testing.memory", "3000000000");
		JavaSparkContext sc = new JavaSparkContext("local[4]", "Spark", conf);
		return sc;
	}
		
	public void trainsLDAmodel() throws IOException {
		JavaRDD<String> data = sc.textFile(filename);
		JavaRDD<String> segRDD = data.map(new Seg());  //将每行评论进行分词
		segRDD.cache(); //暂存到内存，以便于二次使用
		JavaRDD<String> words = segRDD.flatMap(new FlatOne()); //扁平化每行评论成一个个的词语
		JavaPairRDD<String, Integer> pairRDD = words.mapToPair(new MapOne());  // 统计每个词语出现的频率
		JavaPairRDD<String, Integer> ones = pairRDD.reduceByKey(new Count()); //抽取出词汇表
		JavaPairRDD<String, Integer> freones = ones.filter(new LongWordFilter());//根据词频情况过滤掉一个低频词语
		JavaRDD<String> vocabularyRDD = freones.keys(); //将剩下的高频词语成为新的词汇表
		List<String> vocabulary = new ArrayList<String>(vocabularyRDD.collect());//将词汇表从RDD转化为普通类型
		for(String word: vocabulary){
			System.out.println(word);
		}
		
		//转化为向量空间模型
			JavaRDD<Vector> vector = segRDD.map(new ToVector(vocabulary));
			vector.cache();
			JavaPairRDD<Long, Vector> corpus = JavaPairRDD.fromJavaRDD(vector.zipWithIndex().map(new VectorMap()));
			LDA ldaModel = new LDA().setK(50).setSeed(50).setDocConcentration(10).setTopicConcentration(10).setMaxIterations(10).setOptimizer("em");
			corpus.cache();
			ldaModel.run(corpus);
		    DistributedLDAModel disldaModel = (DistributedLDAModel) ldaModel.run(corpus);
//
//			//每一个document下的topic分布, 获取热门话题topic
		      Map<Integer , Integer> map = new HashMap<Integer, Integer>();
			  RDD<Tuple3<Object, int[], double[]>> ttpd = disldaModel.topTopicsPerDocument(10);
			  Tuple3<Object, int[], double[]>[] result =  (Tuple3<Object, int[], double[]>[]) ttpd.collect();
			  for(Tuple3<Object, int[], double[]> one : result){
			        int[] topic = one._2();				//Top_topic
			        for(int t: topic){
			        	  if( ! map.containsKey(t)){
					        	map.put(t, 0);
					        }
			        	  int count = map.get(t) +1;
			        	  map.put(t, count);
			        }
			  }
			  Set<Integer> s = map.keySet(); // 获取topic的id, 并排序
			  List<Map.Entry<Integer,Integer>> infolds = new ArrayList<Map.Entry<Integer, Integer>>(map.entrySet());  // 转化为list，然后对值排序
				Collections.sort(infolds, new Comparator<Map.Entry<Integer, Integer>>(){
					public int compare(Map.Entry<Integer,Integer>o1, Map.Entry<Integer, Integer>o2){
						return (o2.getValue() - o1.getValue());
						}
					});
//			
//			//获取前k个的热门话题
			int topicNum = 6;
			int[] hotTopics  = new int[topicNum];
			for(int i=0; i< topicNum;i++){
				hotTopics[i]= infolds.get(i).getKey();
				System.out.print(hotTopics[i] + " ");
			}
			System.out.println();
//		    //每一个topic下的词汇
			Tuple2<int[], double[]>[] dt = disldaModel.describeTopics();
			int topTermNums = 100;	//限制每一个hotTopic下的词汇
			List<int[]> termInTopics = new ArrayList<int[]>(); 		//记录每一个hotTopic下的词汇索引
			int[] tmp = new int[ topTermNums];
			for(int index : hotTopics){
				int count = 0;		//遍历迭代次数
				for(Tuple2<int[], double[]> topic: dt){
					if(count == index){
						int[] wordIndices = topic._1();
						System.out.print("Topic " + index + " : ");
						for(int i = 0 ; i < topTermNums; i++){
							int j = wordIndices[i];
							tmp[i] = j;
							System.out.print(vocabulary.get(j)+ " ");
						}
						System.out.println();
						termInTopics.add(tmp);
						break;
					}
					count += 1;
				}
			}
			
		}
			
			
//			//聚类
//			JavaRDD<String> comments = sc.textFile(filename);
//			List<String> commentsList = comments.collect();
//			
//			FileWriter fw = new FileWriter("reverse.txt");
//			PrintWriter pw = new PrintWriter(fw);
//			
//			
//			//构建倒排索引
//			Map<String, Set<Long>> reverseIndex = new HashMap<String, Set<Long>>();
//			for(String eachword : vocabulary){
//				Set<Long> docid = new HashSet<Long>();
//				for(String comment: commentsList){
//					if(comment.contains(eachword)){
//						docid.add((long) commentsList.indexOf(comment));
//					}
//				}
//				System.out.println(eachword + ":"+ docid);
//				pw.println(eachword + ":"+ docid);
//				reverseIndex.put(eachword, docid);
//			}
//			
//			pw.close();
			//　构建相似度矩阵
//			for(int[] terms: termInTopics){
//				List<Vector> similarityVector = new ArrayList<Vector>();
//				int length = terms.length;
//				double[] termInTopicsWithWeight = new double[length];
//				for(int i =0 ; i< length;i++){
//					for(int j = 0; j< length; j++){
//						int termIndex_i = terms[i];
//						int termIndex_j = terms[j];
//						String term_i = vocabulary.get(termIndex_i);
//						String term_j = vocabulary.get(termIndex_j);
//						Set<Long> reverse_i = reverseIndex.get(term_i);
//						Set<Long> reverse_j = reverseIndex.get(term_j);
////						System.out.println(reverse_i.size());
//						 reverse_i.retainAll(reverse_j);
//						double Freq_ij  = (double)reverse_i.size();
//						double pmi =  Freq_ij;
////						System.out.println(pmi);
//						termInTopicsWithWeight[j] = pmi;
//					}
//					similarityVector.add(Vectors.dense(termInTopicsWithWeight));
//				}
			
			
//			for(int[] terms: termInTopics){
//				List<Vector> similarityVector = new ArrayList<Vector>();
//				int length = vocabulary.size();
//				double[] termInTopicsWithWeight = new double[length];
//				for(int i =0 ; i< length;i++){
//					for(int j = 0; j< length; j++){
//						String term_i = vocabulary.get(i);
//						String term_j = vocabulary.get(j);
//						Set<Long> reverse_i = reverseIndex.get(term_i);
//						Set<Long> reverse_j = reverseIndex.get(term_j);
//						reverse_i.retainAll(reverse_j);
//						double Freq_ij  = (double)reverse_i.size();
//						double pmi =  Freq_ij;
//						termInTopicsWithWeight[j] = pmi;
//					}
//					similarityVector.add(Vectors.dense(termInTopicsWithWeight));
//				}
//				JavaRDD<Vector> parsedData = sc.parallelize(similarityVector);
//				parsedData.cache();
//				GaussianMixtureModel gmm = new GaussianMixture().setK(5).run(parsedData.rdd());
//				JavaRDD<Integer> results = gmm.predict(parsedData);
//			  	List<Integer> groups  = results.collect();
//			  	
//				int num = 0;
//				for(int group: groups){
//				 System.out.println("member " + num + " : " + group);
//				 num += 1;
//				}
//				break;
//			}
			
//		}
		
		static class BothContains implements Function<String, Boolean>{
			String term_i;
			String term_j;
			public BothContains(String term_i , String term_j){
				this.term_i = term_i;
				this.term_j = term_j;
			}
			public Boolean call(String sentence){
				return sentence.contains(term_i) && sentence.contains(term_j);
			}
		}
		
		static class Contains implements Function<String, Boolean>{
			String term;
			public Contains(String term){
				this.term = term;
			}
			public Boolean call(String sentence){
				return sentence.contains(term);
			}
		}
		static class LongWordFilter implements Function<Tuple2<String, Integer>, Boolean>{
			public Boolean call(Tuple2<String, Integer> data){
				int Frequency = data._2;
				String word = data._1;
				Boolean tagWord = ( Frequency > 3 && word.length() > 1);
				return tagWord;
			}
		}
		
		static class VectorMap implements  Function<Tuple2<Vector, Long>, Tuple2<Long, Vector>>{
			public Tuple2<Long, Vector> call(Tuple2<Vector, Long> doc_id){
				return doc_id.swap();
			}
		}
		
		static class ToVector implements Function<String, Vector>{
			
			List<String> vocabulary;
			public ToVector(List<String> vocabulary){
				this.vocabulary = vocabulary;
			}
			
			public Vector call(String sentence){
				double[] values =new double[vocabulary.size()];;
				for(int i =0 ;i < vocabulary.size(); i++){
					values[i] = 0;
				}
				String[] tokens = sentence.split("`");
				for(String word: tokens){
					int index = vocabulary.indexOf(word);
					if (index >=0 && index < vocabulary.size()){
							values[index] += 1;
					}
				}
				Vector vector = Vectors.dense(values);
				return vector;
			}
			
		}
		
		
		static class FlatOne implements java.io.Serializable, FlatMapFunction<String, String> {
			public Iterable<String> call(String s) {
			    return Arrays.asList(s.split("`"));
			}
		}
		static class Seg implements Function<String, String>{
			public String call(String sentence) throws Exception {
				String segStr = "";
				List<Term> termList = segment.seg(sentence);
				StringBuilder sb = new StringBuilder();
				for(Term term: termList){
					String  str= term.toString();
					Boolean tagPos = ( str.contains("n") || str.contains("v") || str.contains("a") || str.contains("i")); 	//保留名词、动词、形容词、成语等
					if(tagPos){
						sb.append(term.word+ "`");
					}
				}
				segStr = sb.toString();
				return segStr;
			}
		}
		
		static class MapOne implements PairFunction<String, String, Integer>{
				/**
			 * 
			 */
			private static final long serialVersionUID = 1L;

				public Tuple2<String, Integer> call (String s){
					return new Tuple2<String, Integer>(s, 1);
				}
		
		}
		
		static class Count implements Function2<Integer, Integer, Integer>{
			public Integer call (Integer i1, Integer i2){
				return i1 + i2;
			}
		}
		
		public static void main(String[] agrs) throws IOException{
			String filename = "DataSet.txt";
			MYLDA  myLDA = new MYLDA(filename);
			myLDA.trainsLDAmodel();
//			myLDA.dataToVector();
		}
}
