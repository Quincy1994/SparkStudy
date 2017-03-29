package Cluster;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.feature.CountVectorizer;
import org.apache.spark.ml.feature.CountVectorizerModel;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.mllib.clustering.BisectingKMeans;
import org.apache.spark.mllib.clustering.BisectingKMeansModel;
import org.apache.spark.mllib.clustering.GaussianMixtureModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.VectorUDT;
import org.apache.spark.mllib.regression.LabeledPoint;
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



public class FeatureSelection {

	
	public static void main(String[] agrs){
		int K = 11;
		String path = "data/cluster/shenghua.txt";
		SparkConf conf = new SparkConf();
		conf.set("spark.testing.memory", "2147480000");
		JavaSparkContext sc = new JavaSparkContext("local[*]","Spark", conf);
		JavaRDD<String> lines  = sc.textFile(path);
		JavaRDD<Row> jrdd = lines.map(new LineToRow());
		
		SQLContext sqlContext = new SQLContext(sc.sc());
		
		//变成dataFrame
		StructType schema = new StructType(new StructField[]{
			new  StructField("text", new ArrayType(DataTypes.StringType, true), false, Metadata.empty())
		});
		DataFrame df = sqlContext.createDataFrame(jrdd, schema);
		
		CountVectorizerModel cvModel = new CountVectorizer().setInputCol("text").setOutputCol("feature").setVocabSize(10000).setMinDF(5).fit(df);
		DataFrame cvdf = cvModel.transform(df);
	
		
		JavaRDD<Row> vectorRow = cvdf.javaRDD().map(new GetVector());
		vectorRow.collect();
		StructField[] fields = {new StructField("features", new VectorUDT(), false, Metadata.empty())};
		StructType schema2 = new StructType(fields);
		DataFrame dataset = sqlContext.createDataFrame(vectorRow, schema2);
		
//		MinMaxScaler scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures");
//		MinMaxScalerModel scalerModel = scaler.fit(dataset);
//		DataFrame scaledData = scalerModel.transform(dataset);
//		dataset = scaledData;
	//	JavaRDD<Row> rows = cvdf.javaRDD();
		StandardScaler scaler = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures").setWithStd(true);
		DataFrame scaledData = scaler.fit(dataset).transform(dataset);
		
		//kmeans聚类
		KMeans kmeans = new KMeans().setK(K).setMaxIter(100);
		KMeansModel model = kmeans.fit(scaledData);
		
		//聚类分组统计
		JavaRDD<Vector> features = scaledData.toJavaRDD().map(new RowToVector());
		features.cache();
		JavaPairRDD<Integer, Integer> group = features.mapToPair(new Prediction(model));
		JavaPairRDD<Integer, Integer> counts = group.reduceByKey(new Count()).sortByKey();		
		List<Tuple2<Integer, Integer>> result = counts.collect();
		
		//记录每个簇的成员
		List<Vector> vectors = features.collect();
		List<Row> datas = new ArrayList<Row>();
		for(Vector vec: vectors){
			int index = vectors.indexOf(vec);
			Row row = RowFactory.create(index, vec);
			datas.add(row);
		}
		JavaRDD<Row> dataWithIndex = sc.parallelize(datas);
		JavaPairRDD<Integer, Integer> predictWithIndex = dataWithIndex.mapToPair(new PredictionWithIndex(model));
		List<Tuple2<Integer, Integer>> indexWithGroupMember = predictWithIndex.collect();
		
		//	JavaPairRDD<Integer, Iterable<Integer>> groupWithIndex = predictWithIndex.groupByKey();
		//List<Tuple2<Integer, Iterable<Integer>>> groupMember = groupWithIndex.collect();
		
		
		//二分kmeans聚类
		BisectingKMeans bkm = new BisectingKMeans().setK(K);
		BisectingKMeansModel bkmodel = bkm.run(features);
		JavaPairRDD<Integer, Integer> group2 = features.mapToPair(new Prediction2(bkmodel));
		JavaPairRDD<Integer, Integer> counts2 = group2.reduceByKey(new Count()).sortByKey();		
		List<Tuple2<Integer, Integer>> result2 = counts2.collect();
		
		//KMeans
		for(Tuple2<Integer, Integer> tuple: result){
			System.out.println(tuple._1() + ":" + tuple._2());
		}
		double WSSSE = model.computeCost(scaledData);
		System.out.println(WSSSE);
		
		//bisecetingKMeans
		for(Tuple2<Integer, Integer> tuple: result2){
			System.out.println(tuple._1() + ":" + tuple._2());
		}
		double cpc = bkmodel.computeCost(features);
		System.out.println(cpc);
	}
	
	static class Prediction2WithIndex  implements PairFunction<Row, Integer, Integer> {
		
		BisectingKMeansModel model;
		public Prediction2WithIndex(BisectingKMeansModel model){
			this.model = model;
		}
		
		public Tuple2<Integer, Integer> call(Row r) throws Exception {
			int index = r.getAs(0);
			Vector v = r.getAs(1);
			int prediction = model.predict(v);
			return new Tuple2<Integer, Integer>(prediction, index);
		}
		
	}
	
	static class PredictionWithIndex  implements PairFunction<Row, Integer, Integer> {
		
		KMeansModel model ;
		public PredictionWithIndex(KMeansModel model){
			this.model = model;
		}
		
		public Tuple2<Integer, Integer> call(Row r) throws Exception {
			int index = r.getAs(0);
			Vector v = r.getAs(1);
			int prediction = model.predict(v);
			return new Tuple2<Integer, Integer>(prediction, index);
		}
		
	}
	
	static class Both implements Function<LabeledPoint, Tuple2<Double, Double>>{
		
		GaussianMixtureModel gmm;
		public Both(GaussianMixtureModel gmm){
			this.gmm = gmm;
		}
		
		public Tuple2<Double, Double> call(LabeledPoint p) throws Exception{
			Vector v = p.features();
			double group = p.label();
			System.out.println(gmm.predict(v));
			double prediction = gmm.predict(v);
			return new Tuple2<Double, Double>(prediction, group);
		}
	}
	
	static class ToLabeledPoint implements Function<Vector, LabeledPoint>{
		
		KMeansModel model;
		public ToLabeledPoint(KMeansModel model){
			this.model = model;
		}
		
		public LabeledPoint call(Vector v) throws Exception{
			int groupNumber = this.model.predict(v);
			return new LabeledPoint(groupNumber, v);
		}
	}
	
	static class Prediction2 implements PairFunction<Vector, Integer, Integer>{
		
	    BisectingKMeansModel model;
		public Prediction2(BisectingKMeansModel model){
			this.model = model;
		}
		public Tuple2<Integer, Integer> call(Vector v) throws Exception{
			int groupNumber = this.model.predict(v);
			return new Tuple2<Integer, Integer>(groupNumber, 1);
		}
	}
	
	static class Prediction implements PairFunction<Vector, Integer, Integer>{
		
		KMeansModel model;
		public Prediction(KMeansModel model){
			this.model = model;
		}
		public Tuple2<Integer, Integer> call(Vector v) throws Exception{
			int groupNumber = this.model.predict(v);
			return new Tuple2<Integer, Integer>(groupNumber, 1);
		}
	}
	
	static class Count implements Function2<Integer, Integer, Integer>{
		
		public Integer call(Integer i1, Integer i2) throws Exception{
			return i1+i2;
		}
	}
	
	static class RowToVector implements Function<Row, Vector>{
		
		public Vector call(Row r) throws Exception{
			Vector v = r.getAs(0);
			return v;
		}
	}
	
	static class GetVector implements Function<Row, Row>{
		
		public Row call(Row r) throws Exception {
			Vector v= r.getAs(1);
			
		//	System.out.println(v);
			return RowFactory.create(v);
		}
	}
	
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
