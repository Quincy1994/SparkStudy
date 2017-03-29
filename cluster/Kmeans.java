package Cluster;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.feature.CountVectorizer;
import org.apache.spark.ml.feature.CountVectorizerModel;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.mllib.clustering.BisectingKMeans;
import org.apache.spark.mllib.clustering.BisectingKMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.VectorUDT;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.ArrayType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import scala.Tuple2;
import Cluster.FeatureSelection.Prediction2WithIndex;
import Cluster.FeatureSelection.RowToVector;

public class Kmeans {
	 
	 static String path = "data/cluster/shenghua.txt";
	 static String clusterpath = "data/cluster/result.txt";
	 
	 public static void main(String[] args) throws Exception{
		 int K = 10;
		 Map<Integer, String> record = new HashMap<Integer, String>();
		 for(int k=8; k<=K; k++){
			 record = cluster(k, record);
		 }
		 Set<Integer> membership =  record.keySet();
		 PrintWriter pw = new PrintWriter(new FileWriter(clusterpath));
		 for(int member:membership){
			 System.out.println(member);
			 System.out.println(record.get(member));
			 String line = String.valueOf(member) + ":" + record.get(member);
			 pw.println(line);
		 }
		 pw.close();
	 }
	 
	public static Map<Integer, String> cluster(int K, Map<Integer, String> record) throws Exception{
		
		//配置spark环境
		SparkConf conf = new SparkConf();
		conf.set("spark.testing.memory", "2147480000");
		JavaSparkContext sc = new JavaSparkContext("local[*]","Spark", conf);
		JavaRDD<String> lines  = sc.textFile(path);
		JavaRDD<Row> jrdd = lines.map(new LineToRow());
		
		//变成dataFrame
		SQLContext sqlContext = new SQLContext(sc.sc());
		StructType schema = new StructType(new StructField[]{
			new  StructField("text", new ArrayType(DataTypes.StringType, true), false, Metadata.empty())
		});
		DataFrame df = sqlContext.createDataFrame(jrdd, schema);
		
		//转化为countVector模式，注意df的数量
		CountVectorizerModel cvModel = new CountVectorizer().setInputCol("text").setOutputCol("feature").setVocabSize(10000).setMinDF(5).fit(df);
		DataFrame cvdf = cvModel.transform(df);
		
		//转化为向量模式
		JavaRDD<Row> vectorRow = cvdf.javaRDD().map(new GetVector());
		StructField[] fields = {new StructField("features", new VectorUDT(), false, Metadata.empty())};
		StructType schema2 = new StructType(fields);
		DataFrame dataset = sqlContext.createDataFrame(vectorRow, schema2);
		
		//特征标准化
		StandardScaler scaler = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures").setWithStd(true);
		DataFrame scaledData = scaler.fit(dataset).transform(dataset);
		
		JavaRDD<Vector> features = scaledData.toJavaRDD().map(new RowToVector());
		features.cache();
		
		//kmeans聚类，迭代次数为100
		KMeans kmeans = new KMeans().setK(K).setMaxIter(100);
		KMeansModel model = kmeans.fit(scaledData);
		//model.save("./kmeansModel");
		
//		//二分kmeans聚类
		BisectingKMeans bkm = new BisectingKMeans().setK(K);
		BisectingKMeansModel bkmodel = bkm.run(features);
//		
//		//kmeans聚类分组统计
//		JavaPairRDD<Integer, Integer> kmgroup = features.mapToPair(new Prediction(model));
//		JavaPairRDD<Integer, Integer> kmcounts =kmgroup.reduceByKey(new Count()).sortByKey();		
//		List<Tuple2<Integer, Integer>> kmresult = kmcounts.collect();
//		
//		//二分kmeans聚类
//		JavaPairRDD<Integer, Integer> bkgroup = features.mapToPair(new Prediction2(bkmodel));
//		JavaPairRDD<Integer, Integer> bkcounts = bkgroup.reduceByKey(new Count()).sortByKey();		
//		List<Tuple2<Integer, Integer>> bkresult = bkcounts.collect();
//		
//		//kmeans的SSE
//		double WSSSE = model.computeCost(scaledData);
//		System.out.println(WSSSE);
//		
//		//二分kmeans的SSE
//		double cpc = bkmodel.computeCost(features);
//		System.out.println(cpc);
		
		//对每个数据贴上标签
		List<Vector> vectors = features.collect();
		Object[] vectors2 = vectors.toArray();
		List<Row> datas = new ArrayList<Row>();
		for(int i=0;i<vectors2.length;i++){
			Vector vec = (Vector) vectors2[i]; 
			Row row = RowFactory.create(i, vec);
			datas.add(row);
		}
		JavaRDD<Row> dataWithIndex = sc.parallelize(datas);
//		
//		//记录kmeans每个簇的成员
//		JavaPairRDD<Integer, Integer> predictWithIndex = dataWithIndex.mapToPair(new PredictionWithIndex(model));
//		JavaPairRDD<Integer, Iterable<Integer>> groupWithIndex = predictWithIndex.groupByKey();
//		List<Tuple2<Integer, Iterable<Integer>>> groupMember = groupWithIndex.collect();
//		Map<Integer, List<Integer>> kmeansGroupMember = new HashMap<Integer, List<Integer>>();
//		for(Tuple2<Integer, Iterable<Integer>> one : groupMember){
//			List<Integer> membership = new ArrayList<Integer>();
//			int groupNumber = one._1();
//			System.out.println("Group :" + groupNumber);
//			System.out.print("group member: ");
//			for(int member: one._2){
//				System.out.print(member + " ");
//				membership.add(member);
//			}
//			System.out.println();
//			kmeansGroupMember.put(groupNumber, membership);
//		}
		
		
		
		
		//记录bkmeans每个簇的成员
//		JavaPairRDD<Integer, Integer> bkpredictWithIndex = dataWithIndex.mapToPair(new Prediction2WithIndex(bkmodel));
//		JavaPairRDD<Integer, Iterable<Integer>> bkgroupWithIndex = bkpredictWithIndex.groupByKey();
//		List<Tuple2<Integer, Iterable<Integer>>> bkgroupMember = bkgroupWithIndex.collect();
//		Map<Integer, List<Integer>> bkGroupMember = new HashMap<Integer, List<Integer>>();
//		for(Tuple2<Integer, Iterable<Integer>> one : bkgroupMember){
//			List<Integer> membership = new ArrayList<Integer>();
//			int groupNumber = one._1();
//			System.out.println("Group :" + groupNumber);
//			System.out.print("group member: ");
//			for(int member: one._2){
//				System.out.print(member + " ");
//				membership.add(member);
//			}
//			System.out.println();
//			bkGroupMember.put(groupNumber, membership);
//		}
		
		//记录bkmeans下每个对象所在的簇
		JavaPairRDD<Integer, Integer> bkpredictWithIndex = dataWithIndex.mapToPair(new Prediction2WithIndex(bkmodel));
		List<Tuple2<Integer, Integer>> memberWithGroup = bkpredictWithIndex.collect();
		sc.stop();
		for(Tuple2<Integer, Integer> one: memberWithGroup){
			int member = one._2();
			int group = one._1();
			if(!record.containsKey(member)){
				record.put(member, "");
			}
			String value = record.get(member);
			value += String.valueOf(group) + " ";
			record.put(member, value);
		}
		return record;
	}
	
	
	
	//将row转变成vector
	static class GetVector implements Function<Row, Row>{
		
		public Row call(Row r) throws Exception {
			Vector v= r.getAs(1);
			return RowFactory.create(v);
		}
	}
	
	//将一行转变成row变量
	static class LineToRow implements Function<String, Row>{

		public Row call(String line) throws Exception {
			// TODO Auto-generated method stub
			List<String> tags = new ArrayList<String>();
			Pattern pattern;
			Matcher matcher;
			
			//获取年龄
			pattern = Pattern.compile("<age>(.*?)</age>");
			matcher = pattern.matcher(line);
			if(matcher.find()){
				tags.add(matcher.group(1));
			}
			
			//获取性别
			pattern = Pattern.compile("<sex>(.*?)</sex>");
			matcher = pattern.matcher(line);
			if(matcher.find()){
				tags.add(matcher.group(1));
			}
			
			//获取地区
			pattern = Pattern.compile("<area>(.*?)</area>");
			matcher = pattern.matcher(line);
			if(matcher.find()){
				tags.add(matcher.group(1));
			}
			
			//获取加入的群组
			pattern = Pattern.compile("<group>(.*?)</group>");
			matcher = pattern.matcher(line);
			if(matcher.find()){
				String group = matcher.group(1);
				String[] groupTags = group.split(",");
				for(String tag: groupTags){
					tags.add(tag);
				}
			}
			
			//获取收藏
			pattern = Pattern.compile("<collection>(.*?)</collection>");
			matcher = pattern.matcher(line);
			if(matcher.find()){
				String collects = matcher.group(1);
				String[] collectTags = collects.split(",");
				for(String tag: collectTags){
					tags.add(tag);
				}
			}
			return RowFactory.create(tags);
		}
	}
}
