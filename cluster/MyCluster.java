import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IDFModel;
import org.apache.spark.ml.feature.Normalizer;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.mllib.clustering.GaussianMixture;
import org.apache.spark.mllib.clustering.GaussianMixtureModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import scala.Tuple2;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.common.Term;


public class MyCluster {
	
	static Segment segment = HanLP.newSegment();
	static int k = 2; //设定有多少个高斯混合模型
	static GaussianMixtureModel gmm;
	
	public static void main(String[] agrs){
		
		//配置spark的初始文件
		SparkConf conf = new SparkConf().setAppName("GMM").setMaster("local");
		conf.set("spark.testing.memory", "2147480000");
		JavaSparkContext sc = new JavaSparkContext(conf);
		SQLContext sqlContext = new SQLContext(sc);
		
		//加载数据
		String filename = "/home/quincy1994/test.txt";
		JavaRDD<String> sentences = sc.textFile(filename);
		JavaRDD<String> segRDD = sentences.map(new Seg());
		JavaRDD<Row> jrdd = segRDD.map(new StringtoRow());
		segRDD.cache();
		
		//数据转换为矩阵
		StructType schema = new StructType(new StructField[]{
				new StructField("sentence", DataTypes.StringType, false, Metadata.empty())
		});
		DataFrame sentenceData = sqlContext.createDataFrame(jrdd, schema);
		Tokenizer tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words");  //tokenizer以简单的空白分割词语
		DataFrame wordsData = tokenizer.transform(sentenceData); // 将句子分割词语
				
		//tfidf模型
		int numFeatures = 20;  //选定抽取前k个特征
		HashingTF hashingTF  = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(numFeatures);
		DataFrame featurizedData = hashingTF.transform(wordsData);
		IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");
		IDFModel idfModel = idf.fit(featurizedData);
		DataFrame result = idfModel.transform(featurizedData);
		
		//归一化处理
		Normalizer normalizer = new Normalizer().setInputCol("features").setOutputCol("normFeatures").setP(1.0);
		DataFrame l1NormData = normalizer.transform(result.select("features"));
		JavaRDD<Vector> normRDD = l1NormData.rdd().toJavaRDD().map(new RowToVector()); //将row转变成为vector
		normRDD.cache();
		
		//使用高斯混合模型进行聚类
		GaussianMixtureModel gmm = new GaussianMixture().setK(k).run(normRDD.rdd());
		normRDD.cache();
		
		//为每个节点标记归属的簇
		RDD<Vector> points = normRDD.rdd();
		JavaRDD<double[]> predictRDD = new JavaRDD(gmm.predictSoft(points), null);
		JavaRDD<Integer> resultRDD = predictRDD.map(new Group());
		resultRDD.cache();
		
		//在每个簇中提取主标签
		Object[] output= resultRDD.collect().toArray();  //得到每个数据点属于的簇
		Object[]  seg = segRDD.collect().toArray();    //得到每个数据点原来的标签词
		//集合不同簇各自的标签词
		List<Tuple2<Integer, String>> list = new ArrayList<Tuple2<Integer, String>>();
		for(int i = 0; i<output.length; i++){
			int group = (Integer) output[i];
			String tags = (String) seg[i];
			Tuple2<Integer, String> one = new Tuple2<Integer, String>(group, tags);
			list.add(one);
		}
		JavaPairRDD<Integer, String>  rddValue = sc.parallelizePairs(list);
		JavaPairRDD<Integer, Iterable<String>> groupRDD = rddValue.groupByKey();  //按簇归类
		JavaRDD<Tuple2<Integer, String>> tagsRDD = groupRDD.map(new ReduceString()); //将不同的标签混合在一块
		JavaRDD<Tuple2<Integer,String>> topKRDD = tagsRDD.map(new TopTag()); //找出前k个具有代表性的标签
		
		//输出结果
		List<Tuple2<Integer, String>> reducelist = topKRDD.collect();
		for(Tuple2<Integer, String> tags: reducelist){
			System.out.println(tags._1() + ":" + tags._2());
		}
		sc.close();
	}
	
	//将row转变为Vector,机器学习模型基本采用vector类型
	static class RowToVector implements Function<Row, Vector>{

		public Vector call(Row r) throws Exception {
			// TODO Auto-generated method stub
		    Vector features = r.getAs(0);    //将row转变成为vector 
		    return features;
		}
	}
	
	//分词类
	static class Seg implements Function<String, String>{
		
		public String call(String sentence) throws Exception{
			String segStr = "";
			List<Term> termList = segment.seg(sentence); //分词
			StringBuilder sb = new StringBuilder();
			for(Term term: termList){
				String word = term.word;
				sb.append(word+ " ");
			}
			segStr = sb.toString().trim();
			return segStr;
		}
	}
	
	//将String的sentence转变为mllib中row数据类型
	static class StringtoRow implements Function<String, Row>{
		
		public Row call(String sentence) throws Exception {
			return RowFactory.create(sentence);
		}
	}
	
	static class Group implements Function<double[], Integer>{

		//我设定归属概率大于0.5的簇，否则当其为噪声
		public Integer call(double[] probabilities) throws Exception {
			double max = 0.5;
			int index = -1;
			for(int i = 0; i < probabilities.length; i++){
				if(max <= probabilities[i]){
					index = i;
					break;
				}
			}
			return index;
		}
	}
	
	static class ReduceString implements Function<Tuple2<Integer, Iterable<String>>, Tuple2<Integer, String>>{
		//合并标签词
		public Tuple2<Integer, String> call(Tuple2<Integer, Iterable<String>> clusterString){
			int key = clusterString._1();
			StringBuffer sb = new StringBuffer();
			Iterable<String> iter = clusterString._2();
			for( String string: iter){
				sb.append(string + " ");
			}
			return new Tuple2(key, sb.toString().trim());
		}
	}
	
	static class TopTag implements Function<Tuple2<Integer, String>, Tuple2<Integer, String>>{
		//将所有的标签收集，排序，找出频率最高的前k个标签词 
		int topK = 3; 
		
		public Tuple2<Integer, String> call(Tuple2<Integer, String> cluster){
			int key = cluster._1();
			String[] taglist = cluster._2().split(" ");
			Map<String, Integer> map = new HashMap<String, Integer>();
			for(String tag: taglist){
				if(!map.containsKey(tag)){
					map.put(tag, 1);
				}
				else{
					int count = map.get(tag);
					map.put(tag, count + 1);
				}
			}
			
			List<Map.Entry<String, Integer>> infolds = new ArrayList<Map.Entry<String, Integer>>(map.entrySet());
			Collections.sort(infolds, new Comparator<Map.Entry<String, Integer>>(){
				public int compare(Map.Entry<String, Integer>o1, Map.Entry<String, Integer>o2){
					return (o2.getValue() - o1.getValue());
				}
			});
			String str = "";
			int num = 0;
			for(Map.Entry<String, Integer> one: infolds){
				str += one.getKey() + " ";
				if(num == topK){
					break;
				}
				num += 1;
			}
			return new Tuple2<Integer, String>(key, str.trim());
		}
	}
}
