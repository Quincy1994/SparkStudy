import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.sql.SQLContext;

import scala.Tuple2;



public class MYNaiveBayes {

	static NaiveBayesModel model;
	
	public static void main(String[] args) throws Exception{
		
		SparkConf conf = new SparkConf().setAppName("GMM").setMaster("local");
		conf.set("spark.testing.memory", "2147480000");
		JavaSparkContext sc = new JavaSparkContext(conf);
		SQLContext sqlContext = new SQLContext(sc);
		
		//获取标签特征词
		List<String> vocabulary = new ArrayList<String>();
		File dir = new File("/home/quincy1994/文档/微脉圈/tags/类别库");
		File[] files = dir.listFiles(); //获取不同类别的标签文件
		StringBuilder sb = new StringBuilder();
		for(File file : files){
			BufferedReader br = new BufferedReader(new FileReader(file));
			String line = null;
			while((line = br.readLine()) != null){
				sb.append(line + "`");  //按“`"分割不同类别的标签
			}
		}
		 String[] tags = sb.toString().trim().split("`");
		 List<String> newTags = new ArrayList<String>();
		 for(String tag: tags){
			 if(tag.length() > 4){
				 newTags.add(tag);  //去除空行标签
			 }
		 }
		 Object[] newtags =  newTags.toArray();
		 List<Tuple2<Integer, String>> list = new ArrayList<Tuple2<Integer,String>>(); //记录每类中的标签
		 for(int i=0; i<newtags.length;i++){
			 Tuple2 <Integer, String> classWithTags = new Tuple2<Integer, String>(i, (String)newtags[i]); 
			 System.out.println(classWithTags);
			 list.add(classWithTags);
			 String[] tokens = ((String)newtags[i]).split("/");
			 for(String tag: tokens){
				 vocabulary.add(tag);
			 }
		 }
	 
	 //获取训练样本
	 JavaPairRDD<Integer, String> trainRDD = sc.parallelizePairs(list); //将每类的标签词转化为RDD
	 JavaPairRDD<Integer, String> trainSetRDD = trainRDD.mapValues(new ToTrainSet(vocabulary)); //将标签词转化为向量模型
	 List<Tuple2<Integer, String>> trainSet = trainSetRDD.collect(); 
	 writeTrainSet(trainSet);  //写成libsvm文件格式，以方便训练
	 System.out.println("trainset is ok");
	 
	 //读取训练集并训练模型
	 String path = "./trainset";
	 JavaRDD<LabeledPoint> trainData = MLUtils.loadLibSVMFile(sc.sc(), path).toJavaRDD();
	 model = NaiveBayes.train(trainData.rdd(), 1.0);  
//	 model.save(sc.sc(), "./model");
	 System.out.println("model is ok");
 
	 //预测新的测试集
	 String testStr = "萌宠 猫狗 ";
	 double[] testArray = sentenceToArrays(vocabulary, testStr);
	 writeTestSet(testArray);
	 String testPath = "./testset";
	 JavaRDD<LabeledPoint> testData = MLUtils.loadLibSVMFile(sc.sc(), testPath).toJavaRDD();
	 
	 //多元分类预测
	 JavaRDD<double[]> resultData = testData.map(new GetProbabilities());
	 List<double[]> result = resultData.collect(); //保存的是每个测试样本所属于不同类别的概率值
	 for(double[] one: result){
		 for(int i=0;i<one.length;i++){
			 System.out.println("class "+ i + ":" + one[i]);
		 }
	 }
	 
	}
	
	public static void writeTestSet(double[] testArray) throws Exception {
		//和writeTrainSet一样
		File file = new File("./testset");
		PrintWriter pr = new PrintWriter(new FileWriter(file));
		pr.print("0" + " ");
		String value = "";
		for(int i=0; i<testArray.length; i++){
			value += (i+1) + ":" + testArray[i] + " ";
		}
		pr.print(value.trim());
		pr.close();
	}
	
	public static void  writeTrainSet( List<Tuple2<Integer, String>> list) throws Exception{
		File file = new File("./trainset");
		PrintWriter pr = new PrintWriter(new FileWriter(file));
		for(Tuple2<Integer, String> one : list){     //将每个训练样本以libsvm格式保存到trainset文件当中
			String label = String.valueOf(one._1);   //训练样本的类别属性
			String vector = one._2();  //训练样本的向量模型
			String[] indexes = vector.split(" ");
			pr.print(label + " ");
			String value = "";
			for(int i = 0; i<indexes.length;i++){
				value += (i+1) + ":" + indexes[i] + " ";   // i+1是因为libsvm文件的index是从1开始
			}
			pr.print(value.trim());
			pr.println();
		}
		pr.close();
	}
	
	 public static double[] sentenceToArrays(List<String> vocabulary, String sentence){
		 double[] vector = new double[vocabulary.size()];
		 for(int i=0; i<vocabulary.size();i++){
			 vector[i] = 0;
		 }
		 String[] tags = sentence.split(" ");
		 for(String tag: tags){
			 if(vocabulary.contains(tag)){
				 int index = vocabulary.indexOf(tag);
				 System.out.println(index);
				 vector[index] += 1;
			 }
		 }
		 return vector;
	 }
	static class CreateLabel implements Function<Tuple2<Integer,Vector>, LabeledPoint>{

		public LabeledPoint call(Tuple2<Integer, Vector> one) throws Exception {
			double label = (double)one._1();
			Vector vector = one._2();
			return new LabeledPoint(label, vector);
		}
		
	}
	
	static class FlatTags implements  Function<String, Iterable<String>>{

		public Iterable<String> call(String tags) throws Exception {
			// TODO Auto-generated method stub
			return Arrays.asList(tags.split("/"));
		}
	}
		
	static class ToTrainSet implements Function<String, String>{
		List<String> vocabulary = null; //标签特征库
		public ToTrainSet(List<String> vocabulary){
			this.vocabulary = vocabulary;
		}
		public String call(String sentence) throws Exception {
			// TODO Auto-generated method stub
			int length = vocabulary.size(); 		//特征维度
			String[] tags = sentence.split("/");	
			List<Integer> tagsindex = new ArrayList<Integer>();
			for(int i =0; i<tags.length; i++){
				tagsindex.add(vocabulary.indexOf(tags[i]));
			}
			String vector = "";  //将特征向量转变为String类，节省空间
			for(int i = 0 ; i < length; i++){
				if(tagsindex.contains(i)){
					vector += String.valueOf(1) + " ";
				}
				else{
					vector += String.valueOf(0) + " ";
				}
			}
			return vector.trim();
		}
	}
	
	static class ToVector implements Function<String, Vector>{
		
		List<String> vocabulary = null;
		public ToVector(List<String> vocabulary){
			this.vocabulary = vocabulary;
		}
		
		public Vector call(String tag) throws Exception {
			// TODO Auto-generated method stub
			int index = vocabulary.indexOf(tag);
			double[] arrays = new double[vocabulary.size()];
			for(int i = 0; i<arrays.length;i++){
				if( i == index){
					arrays[i] = 1;
				}
				else{
					arrays[i] = 0;
				}
			}
			return Vectors.dense(arrays);
		}
	}
	
	static class GetProbabilities implements Function<LabeledPoint, double[]>{
		
		public double[] call(LabeledPoint p){
			Vector predict = model.predictProbabilities(p.features());
			double[] probabilities =predict.toArray();
			return probabilities;
		}
	}
	
	static class Predict implements Function<LabeledPoint, Double> {
		
		public Double call(LabeledPoint p){
			double predict = model.predict(p.features());
			return predict;
		}
	}
}
