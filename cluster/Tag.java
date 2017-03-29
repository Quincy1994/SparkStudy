package Cluster;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
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
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

public class Tag {

	String clusterResult = "";
	String dataSet ="";
	Map<Integer, List<Integer>> group = null;
	List<String> dataRow = null;
	Map<Integer, String> groupmember = null;
	
	public Tag(String clusterResult, String dataSet){
		this.clusterResult = clusterResult;
		this.dataSet = dataSet;
		try {
			this.group = this.getClusterResult();
			this.dataRow = this.readData();
			this.groupmember = this.getGroupMember();
			this.getTag();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public Map<Integer, List<Integer>> getClusterResult() throws Exception{
		Map<Integer,List<Integer>> group = new HashMap<Integer, List<Integer>>();
		BufferedReader br = new BufferedReader(new FileReader(clusterResult));
		String line;
		while((line = br.readLine())!=null){
			String[] tokens = line.split(":");
			int groupNumber = Integer.parseInt(tokens[0]);
			String[] members = tokens[1].split(",");
			List<Integer> memberIndex = new ArrayList<Integer>();
			for(String m: members){
				memberIndex.add(Integer.parseInt(m));
			}
			group.put(groupNumber, memberIndex);
		}
		return group;
	}
	
	public List<String> readData() throws Exception{
		List<String> dataRow = new ArrayList<String>();
		BufferedReader br = new BufferedReader(new FileReader(this.dataSet));
		String line;
		Pattern pattern;
		Matcher matcher;
		while((line = br.readLine())!=null){
			String tags = "";
			//获取年龄
			pattern = Pattern.compile("<age>(.*?)</age>");
			matcher = pattern.matcher(line);
			if(matcher.find()){
				tags += (matcher.group(1)) +",";
			}
			
			//获取性别
			pattern = Pattern.compile("<sex>(.*?)</sex>");
			matcher = pattern.matcher(line);
			if(matcher.find()){
				tags += (matcher.group(1)) +",";
			}
			
			//获取地区
			pattern = Pattern.compile("<area>(.*?)</area>");
			matcher = pattern.matcher(line);
			if(matcher.find()){
				tags += (matcher.group(1)) +",";
			}
			
			//获取加入的群组
			pattern = Pattern.compile("<group>(.*?)</group>");
			matcher = pattern.matcher(line);
			if(matcher.find()){
				String group = matcher.group(1);
				String[] groupTags = group.split(",");
				for(String tag: groupTags){
					tags += tag +",";
				}
			}
			
			//获取收藏
			pattern = Pattern.compile("<collection>(.*?)</collection>");
			matcher = pattern.matcher(line);
			if(matcher.find()){
				String collects = matcher.group(1);
				String[] collectTags = collects.split(",");
				for(String tag: collectTags){
					tags += tag +",";
				}
			}
			dataRow.add(tags);
		}
		return dataRow;
	}
	
	public Map<Integer, String> getGroupMember(){
		Map<Integer, String> groupmember = new HashMap<Integer,String>();
		Set<Integer> group = this.group.keySet();
		System.out.println(group.size());
		for(int number: group){
			String rows = "";
			List<Integer> members = this.group.get(number);
			for(int m:members){
				String row =this.dataRow.get(m);
				rows += row + ",";
			}
			groupmember.put(number, rows);
		}
		return groupmember;
	}
	
	
	public void getTag() throws Exception{	
		Set<Integer> group = this.groupmember.keySet();
		for(int groupNumber: group){
			String groupData = this.groupmember.get(groupNumber);
			groupData = groupData.replace("\n", "");
			String[] data = groupData.split(",");
			for(String d: data){
				System.out.println(d);
			}
			List<String> list = Arrays.asList(data);
			countTag(list, groupNumber);
		}		
	}
	
	public void countTag(List<String> list, int groupNumber) throws Exception{
		SparkConf conf = new SparkConf();
		conf.set("spark.testing.memory", "2147480000");
		JavaSparkContext sc = new JavaSparkContext("local[*]","Spark", conf);
		JavaRDD<String> lines  = sc.parallelize(list);
		JavaPairRDD<String, Integer> ones = lines.mapToPair(new MapOne());
		JavaPairRDD<String, Integer> counts = ones.reduceByKey(new Count());
		List<Tuple2<String,Integer>> result = counts.collect();
		Map<String, Integer> map = new HashMap<String, Integer>();	
		for(Tuple2<String, Integer> tuple: result){
			map.put(tuple._1(), tuple._2());
		}
		List<Map.Entry<String,Integer>> infolds = new ArrayList<Map.Entry<String, Integer>>(map.entrySet());  // 转化为list，然后对值排序
		Collections.sort(infolds, new Comparator<Map.Entry<String, Integer>>(){
			public int compare(Map.Entry<String,Integer>o1, Map.Entry<String, Integer>o2){
				return (o2.getValue() - o1.getValue());
				}
		});
		String filename = String.valueOf(groupNumber) + ".txt";
		PrintWriter pw = new PrintWriter(new FileWriter(filename));
		for(Map.Entry<String,Integer> one : infolds){
			String tag = one.getKey();
			int num = one.getValue();
			pw.println(tag+ ":" + String.valueOf(num));
		}
		pw.close();
		sc.stop();
	}
	
	public static void main(String[] args) throws Exception{
		String clusterResult = "data/cluster/聚类结果.txt";
		String dataSet ="data/cluster/shenghua.txt";
		Tag t = new Tag(clusterResult, dataSet);
	}
	
	static class MapOne implements PairFunction<String, String, Integer>{

		public Tuple2<String, Integer> call(String tag) throws Exception {
			// TODO Auto-generated method stub
			
			return new Tuple2<String, Integer>(tag, 1);
		}
	
	}
	
	static class Count implements Function2<Integer, Integer, Integer>{
		/*
		 * 计算每一个元素
		 */
		public Integer call(Integer i1, Integer i2){
			return i1 + i2;
		}
}
}
