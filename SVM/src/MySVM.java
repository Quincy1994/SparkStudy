import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.feature.ChiSqSelector;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.VectorUDT;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
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
import com.hankcs.hanlp.seg.common.Term;


public class MySVM {
	List<String> vocabulary;
	List<String> manVocabulary;
	List<String> womanVocabulary;
	Map<String, Double> words; 
	String dataSetFilename = "data/data.txt";
	String libsvmFilename = "data/libsvmFile.txt";
	SVMModel model;

	
	public void trainModel(){
		try {
			this.loadVocabulary();
			this.writeLibsvmFile();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public void loadVocabulary() throws Exception{
		vocabulary = new ArrayList<String>();
		manVocabulary = new ArrayList<String>();
		womanVocabulary = new ArrayList<String>();
		words = new HashMap<String, Double>();
		String line;
		BufferedReader br;
		
		//加载男标签
		String mantagFile = "data/feature/manTag.txt";
		br = new BufferedReader(new FileReader(new File(mantagFile)));
		while((line=br.readLine())!= null){
			System.out.println(line);
			String[] token = line.split("`");
			String tag = token[0];
			double value = Double.parseDouble(token[1]);
			manVocabulary.add(tag);
			vocabulary.add(tag);
			words.put(tag, value);
		}
		br.close();
		
		//加载女标签
		String womantagFile = "data/feature/womanTag.txt";
		br = new BufferedReader(new FileReader(new File(womantagFile)));
		while((line=br.readLine())!= null){
			String[] token = line.split("`");
			String tag = token[0];
			double value = Double.parseDouble(token[1]);
			womanVocabulary.add(tag);
			vocabulary.add(tag);
			words.put(tag, 1-value);
		}
		br.close();
		
		//加载男群组
		String manGroupFile = "data/feature/manGroup.txt";
		br = new BufferedReader(new FileReader(new File(manGroupFile)));
		while((line=br.readLine())!= null){
			String[] token = line.split("`");
			String tag = token[0];
			double value = Double.parseDouble(token[1]);
			manVocabulary.add(tag);
			vocabulary.add(tag);
			words.put(tag, value);
		}
		br.close();
		
		//加载女群组
		String womanGroupFile = "data/feature/womanGroup.txt";
		br = new BufferedReader(new FileReader(new File(womanGroupFile)));
		while((line=br.readLine())!= null){
			String[] token = line.split("`");
			String tag = token[0];
			double value = Double.parseDouble(token[1]);
			womanVocabulary.add(tag);
			vocabulary.add(tag);
			words.put(tag, 1-value);
		}
		br.close();
		
		//加载男用语习惯
		String manWordFile = "data/feature/wordMan.txt";
		br = new BufferedReader(new FileReader(new File(manWordFile)));
		while((line=br.readLine())!= null){
			System.out.println(line);
			String[] token = line.split("`");
			String tag = token[0];
			double value = Double.parseDouble(token[1]);
			manVocabulary.add(tag);
			vocabulary.add(tag);
			words.put(tag, value);
		}
		br.close();
		
		//加载女用语习惯
		String womanWordFile = "data/feature/wordWoman.txt";
		br = new BufferedReader(new FileReader(new File(womanWordFile)));
		while((line=br.readLine())!= null){
			System.out.println(line);
			String[] token = line.split("`");
			String tag = token[0];
			double value = Double.parseDouble(token[1]);
			womanVocabulary.add(tag);
			vocabulary.add(tag);
			words.put(tag, 1-value);
		}
		br.close();
	}
	
	public void writeLibsvmFile() throws Exception{
		BufferedReader br = new BufferedReader(new FileReader(new File(this.dataSetFilename)));
		String line;
		String libsvmStr = "";
		List<Term> termList;
		while((line = br.readLine())!= null){
			String vectorStr = "";
			
			//添加标签
			if(line.contains("<sex>男</sex>")){
				vectorStr += "1 ";
			}
			else{
				vectorStr += "0 ";
			}
			
			//添加属性
			double[] vector = new double[this.vocabulary.size()];
			for(int i=0; i<this.vocabulary.size();i++){
				vector[i] = 0.0;
			}
			
//			//添加标签属性
			Pattern patternTag = Pattern.compile("<tag>(.*?)</tag>");
			Matcher matcherTag = patternTag.matcher(line);
			String tags = null;
			if(matcherTag.find()){
				tags = matcherTag.group(1);
			}
			String[] token = tags.split(",");
			for(String tag: token){
				if(this.vocabulary.contains(tag)){
					int index = this.vocabulary.indexOf(tag);
					vector[index] = 1;
					if(this.words.containsKey(tag)){
						double value = this.words.get(tag);
						vector[index] = 1;
					}
				}
			}
			
			//添加群组属性
			Pattern patternGroup = Pattern.compile("<group>(.*?)</group>");
			Matcher matcherGroup = patternGroup.matcher(line);
			String groups = null;
			if(matcherGroup.find()){
				groups = matcherGroup.group(1);
			}
			termList = HanLP.segment(groups);
			for(Term term: termList){
				String tag = term.word;
				if(this.vocabulary.contains(tag)){
					int index = this.vocabulary.indexOf(tag);
					vector[index] = 1;
					if(this.words.containsKey(tag)){
						double value = this.words.get(tag);
						vector[index] = 1;
					}
				}
			}
			
			//添加用语属性
			Pattern patternWord = Pattern.compile("<words>(.*?)</words>");
			Matcher matcherWord = patternWord.matcher(line);
			String words = null;
			if(matcherWord.find()){
				words = matcherWord.group(1);
			}
			termList = HanLP.segment(words);
			for(Term term: termList){
				String tag = term.word;
				if(this.vocabulary.contains(tag)){
					int index = this.vocabulary.indexOf(tag);
					vector[index] = 1;
					if(this.words.containsKey(tag)){
						double value = this.words.get(tag);
						vector[index] = value;
					}
				}
			}
			
			//将属性转变成为字符串
			for(int i=0; i<this.vocabulary.size(); i++){
				if(vector[i] > 0){
					vectorStr += String.valueOf(i+1) + ":" + String.valueOf(vector[i]) + " ";
				}
			}
			if(vectorStr.length() < 40){
				continue;
			}
			libsvmStr += vectorStr.trim() + "\n";
		}
		
		//写入文件
		PrintWriter pw = new PrintWriter(new FileWriter(new File(this.libsvmFilename)));
		pw.print(libsvmStr);
		pw.close();
	}
	
	public void trainSVM(){
		
		SparkConf conf = new SparkConf().setAppName("SVM").setMaster("local");
		conf.set("spark.testing.memory", "2147480000");
		SparkContext sc = new SparkContext(conf);
		String path = libsvmFilename;
		JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(sc, path).toJavaRDD();
		
		JavaRDD<Row> jrdd = data.map(new LabelToRow());
		StructType schema = new StructType(new StructField[]{
				new StructField("label", DataTypes.DoubleType, false, Metadata.empty()),
				new StructField("features",new VectorUDT(), false, Metadata.empty()),
		});
		SQLContext jsql = new 	SQLContext(sc);
		DataFrame df2 =jsql.createDataFrame(jrdd, schema);
//		
		StandardScaler  scaler = new StandardScaler().setInputCol("features").setOutputCol("normFeatures").setWithStd(true).setWithMean(false);
		DataFrame df = scaler.fit(df2).transform(df2);
		
//		PCAModel pca = new PCA().setInputCol("features").setOutputCol("pcaFeatures").setK(3000).fit(df);
////		DCT dct = new DCT().setInputCol("features").setOutputCol("dctFeatures").setInverse(false);
//		DataFrame results = pca.transform(df).select("label","pcaFeatures");
//		JavaRDD<Row> rows = results.javaRDD();
//		JavaRDD<LabeledPoint> data2 = rows.map(new RowToLabel());
	
		ChiSqSelector selector = new ChiSqSelector().setNumTopFeatures(5000).setFeaturesCol("normFeatures").setLabelCol("label").setOutputCol("selectedFeatures");
		DataFrame results = selector.fit(df).transform(df).select("label","selectedFeatures");
//		PCAModel pca = new PCA().setK(3000).setInputCol("selectedFeatures").setOutputCol("pcaFeatures").fit(results);
//		DataFrame results2 = pca.transform(results).select("label","pcaFeatures");
		
		JavaRDD<Row> rows = results.javaRDD();
		JavaRDD<LabeledPoint> data2 = rows.map(new RowToLabel());
		
		
		//切分数据
		JavaRDD<LabeledPoint>[] tmp = data2.randomSplit(new double[]{0.6, 0.4}, 12345);
		JavaRDD<LabeledPoint> training = tmp[0];
		training.cache();
		JavaRDD<LabeledPoint> test = tmp[1];
		
		//构建模型
		int numIterations = 200;
		this.model = SVMWithSGD.train(training.rdd(), numIterations);
		model.clearThreshold();
		
		//计算结果
		JavaRDD<Tuple2<Double, Double>> scoreAndLabels = test.map(new Prediction(model));
		double accuracy = 1.0 *scoreAndLabels.filter(new Accuracy()).count() / test.count();
		double accuracyMan = 1.0 *scoreAndLabels.filter(new CountMan()).filter(new Accuracy()).count() / scoreAndLabels.filter(new CountMan()).count();
		double accuracyWoman = 1.0 *scoreAndLabels.filter(new CountWoman()).filter(new Accuracy()).count() / scoreAndLabels.filter(new CountWoman()).count();
		
		List<Tuple2<Double,Double>> result = scoreAndLabels.collect();
//		for(Tuple2<Double, Double> one: result){
//			if(one._1() >= 0 && one._2() == 0){
//				System.out.println("predict: " + one._1() + " label: " + one._2());
//			}
//			else if(one._1() < 0 && one._2() == 1){
//				System.out.println("predict: " + one._1() + " label: " + one._2());
//			}
//		}
//		
		System.out.println("accuracyMan:" + accuracyMan);
		System.out.println("accuracyWoman:" + accuracyWoman);
		System.out.println("accuracy:" + accuracy);
	}
	
	static class Prediction implements Function<LabeledPoint , Tuple2<Double, Double>>{
		SVMModel model;
		public Prediction(SVMModel model){
			this.model = model;
		}
		public Tuple2<Double, Double> call(LabeledPoint p) throws Exception {
			// TODO Auto-generated method stub
			Double score = model.predict(p.features());
			return new Tuple2<Double,Double>(score, p.label());
		}
	}
	
	static class CountMan implements Function<Tuple2<Double, Double>,  Boolean>{
		
		public Boolean call(Tuple2<Double, Double> p){
			if(p._2 == 1.0) return true;
			else return false;
		}
	}
	
	static class CountWoman implements Function<Tuple2<Double, Double>,  Boolean>{
		
		public Boolean call(Tuple2<Double, Double> p){
			if(p._2 == 0.0) return true;
			else return false;
		}
	}
	static class Accuracy implements Function<Tuple2<Double, Double>, Boolean>{
		
		public Boolean call(Tuple2<Double, Double> p){
			if(p._1() > 0&& p._2() == 1.0){
				return true;
			}
			else if(p._1() <= 0  && p._2() == 0.0){
				return true;
			}
			else return false;
		}
	}
	
	static class LabelToRow implements Function<LabeledPoint ,Row>{

		public Row call(LabeledPoint p) throws Exception {
			// TODO Auto-generated method stub
			double label = p.label();
			Vector vector =p.features();
			return RowFactory.create(label, vector);
		}
		
	}
	
	static class RowToLabel implements Function<Row ,LabeledPoint>{

		public LabeledPoint call(Row r) throws Exception {
			// TODO Auto-generated method stub
			Vector features = r.getAs(1);
			double label = r.getDouble(0);
			return new LabeledPoint(label, features);
		}
		
	}
	public static void main(String[] args) throws Exception{
		MySVM svm = new MySVM();
//		svm.loadVocabulary();
//		svm.writeLibsvmFile();
		svm.trainSVM();
	}
}
