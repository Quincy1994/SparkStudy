import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.seg.common.Term;


public class FeatureSelection {

	Map<String,  Double> manfeature;
	Map<String, Double> womanfeature;
	Map<String, Double> feature; //记录所有特征的集合
	String dataSetFilename = "data/dataSet.txt";
	
	public static void main(String[] args){
		FeatureSelection fs = new FeatureSelection();
//		fs.getTag();
//		fs.getGroup();
		fs.getWord();
	}
	
	public void getWord(){
		manfeature = new HashMap<String, Double>();
		womanfeature = new HashMap<String, Double>();
		Map<String, Double> feature = new HashMap<String, Double>(); //记录所有特征的集合
		try {
			BufferedReader br = new BufferedReader(new FileReader(new File(this.dataSetFilename)));
			String line;
			
			//读取标签，分别记入男女的特征集合中
			while((line=br.readLine())!=null){
				String regexTag = "<words>(.*?)</words>";
				Pattern patternTag = Pattern.compile(regexTag);
				Matcher matcherTag = patternTag.matcher(line);
				String groupTag ="";
				if(matcherTag.find()){
					groupTag = matcherTag.group(1);
				}
				List<Term> tokens = HanLP.segment(groupTag);
				String patternMan = "<sex>男</sex>";
				if(line.contains(patternMan)){
					for(Term term: tokens){
						String tag = term.word;
						if(!manfeature.containsKey(tag)){
							manfeature.put(tag, 0.0);
						}
						double countNum = manfeature.get(tag);
						manfeature.put(tag, countNum+1);
						feature.put(tag, 1.0);
					}
				}
				else{
					for(Term term: tokens){
						String tag = term.word;
						if(!womanfeature.containsKey(tag)){
							womanfeature.put(tag, 0.0);
						}
						double countNum = womanfeature.get(tag);
						womanfeature.put(tag, countNum+1);
						feature.put(tag, 1.0);
					}
				}
			}
			
			//计算每个标签的概率
			Set<String> tagSet = feature.keySet();
			for(String tag: tagSet){
				double tagInman = 0;
				double tagInwoman = 0;
				if(manfeature.containsKey(tag)){
					tagInman = manfeature.get(tag);
				}
				if(womanfeature.containsKey(tag)){
					tagInwoman = womanfeature.get(tag);
				}
				double percent = tagInman / (tagInman + tagInwoman);
				
				//过滤掉低频标签
				if(tagInman + tagInwoman > 20 && tag.length() >=2 ){
					feature.put(tag, percent);
				}
			}
			
			//排序
			List<Map.Entry<String,Double>> infolds = new ArrayList<Map.Entry<String, Double>>(feature.entrySet());
			Collections.sort(infolds, new DescendSort());
			for(Map.Entry<String, Double> one: infolds){
				double value = one.getValue();
				String key = one.getKey();
				if(value>=0.7&& value <1){
					System.out.println(key + "," + value);
				}
			}
		}catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
	}
	
	public void getGroup(){
		manfeature = new HashMap<String, Double>();
		womanfeature = new HashMap<String, Double>();
		Map<String, Double> feature = new HashMap<String, Double>(); //记录所有特征的集合
		try {
			BufferedReader br = new BufferedReader(new FileReader(new File(this.dataSetFilename)));
			String line;
			
			//读取标签，分别记入男女的特征集合中
			while((line=br.readLine())!=null){
				String regexTag = "<group>(.*?)</group>";
				Pattern patternTag = Pattern.compile(regexTag);
				Matcher matcherTag = patternTag.matcher(line);
				String groupTag ="";
				if(matcherTag.find()){
					groupTag = matcherTag.group(1);
				}
				List<Term> tokens = HanLP.segment(groupTag);
				String patternMan = "<sex>男</sex>";
				if(line.contains(patternMan)){
					for(Term term: tokens){
						String tag = term.word;
						if(!manfeature.containsKey(tag)){
							manfeature.put(tag, 0.0);
						}
						double countNum = manfeature.get(tag);
						manfeature.put(tag, countNum+1);
						feature.put(tag, 1.0);
					}
				}
				else{
					for(Term term: tokens){
						String tag = term.word;
						if(!womanfeature.containsKey(tag)){
							womanfeature.put(tag, 0.0);
						}
						double countNum = womanfeature.get(tag);
						womanfeature.put(tag, countNum+1);
						feature.put(tag, 1.0);
					}
				}
			}
			
			//计算每个标签的概率
			Set<String> tagSet = feature.keySet();
			for(String tag: tagSet){
				double tagInman = 0;
				double tagInwoman = 0;
				if(manfeature.containsKey(tag)){
					tagInman = manfeature.get(tag);
				}
				if(womanfeature.containsKey(tag)){
					tagInwoman = womanfeature.get(tag);
				}
				double percent = tagInman / (tagInman + tagInwoman);
				
				//过滤掉低频标签
				if(tagInman + tagInwoman > 20 && tag.length() >=2 ){
					feature.put(tag, percent);
				}
			}
			
			//排序
			List<Map.Entry<String,Double>> infolds = new ArrayList<Map.Entry<String, Double>>(feature.entrySet());
			Collections.sort(infolds, new DescendSort());
			for(Map.Entry<String, Double> one: infolds){
				double value = one.getValue();
				String key = one.getKey();
				if(value>=0 && value <0.4){
					System.out.println(key + "," + value);
				}
			}
		}catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
	}
	
	public void getTag(){
		manfeature = new HashMap<String, Double>();
		womanfeature = new HashMap<String, Double>();
		Map<String, Double> feature = new HashMap<String, Double>(); //记录所有特征的集合
		try {
			BufferedReader br = new BufferedReader(new FileReader(new File(this.dataSetFilename)));
			String line;
			
			//读取标签，分别记入男女的特征集合中
			while((line=br.readLine())!=null){
				String regexTag = "<tag>(.*?)</tag>";
				Pattern patternTag = Pattern.compile(regexTag);
				Matcher matcherTag = patternTag.matcher(line);
				String tags ="";
				if(matcherTag.find()){
					tags = matcherTag.group(1);
				}
				String[] tokens = tags.split(",");
				String patternMan = "<sex>男</sex>";
				if(line.contains(patternMan)){
					for(String tag: tokens){
						if(!manfeature.containsKey(tag)){
							manfeature.put(tag, 0.0);
						}
						double countNum = manfeature.get(tag);
						manfeature.put(tag, countNum+1);
						feature.put(tag, 1.0);
					}
				}
				else{
					for(String tag: tokens){
						if(!womanfeature.containsKey(tag)){
							womanfeature.put(tag, 0.0);
						}
						double countNum = womanfeature.get(tag);
						womanfeature.put(tag, countNum+1);
						feature.put(tag, 1.0);
					}
				}
			}
			
			//计算每个标签的概率
			Set<String> tagSet = feature.keySet();
			for(String tag: tagSet){
				double tagInman = 0;
				double tagInwoman = 0;
				if(manfeature.containsKey(tag)){
					tagInman = manfeature.get(tag);
				}
				if(womanfeature.containsKey(tag)){
					tagInwoman = womanfeature.get(tag);
				}
				double percent = tagInman / (tagInman + tagInwoman);
				
				//过滤掉低频标签
				if(tagInman + tagInwoman > 10){
					feature.put(tag, percent);
				}
			}
			
			//排序
			List<Map.Entry<String,Double>> infolds = new ArrayList<Map.Entry<String, Double>>(feature.entrySet());
			Collections.sort(infolds, new DescendSort());
			for(Map.Entry<String, Double> one: infolds){
				double value = one.getValue();
				String key = one.getKey();
				if(value>=0.8 && value <1){
					System.out.println(key + "," + value);
				}
			}
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	//倒序排序
	static class DescendSort implements Comparator<Map.Entry<String, Double>> {
		
		public int compare(Map.Entry<String, Double> o1, Map.Entry<String, Double> o2) {
			// TODO Auto-generated method stub
			if(o1.getValue() > o2.getValue()) return -1;
			else if( o1.getValue() < o2.getValue()) return 1;
			else return 0;
		}
	}
}
