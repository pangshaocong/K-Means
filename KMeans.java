import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class KMeans {
	
	static int clCnt = 4;		//clusters count = 4
	static int maxIter = 10;		// stop criteria is max iterations count = 10
	static double[][] centroids = new double[clCnt][11];		//2D arrays for centroids, col 1 stores cluster number
	static int iterCnt;
	
	public static void main(String[] args) throws Exception {
		
		//initialize centroids array, vector values are assigned to be same as cluster number
		for (int s = 0; s < clCnt; s++) {			
			for (int t = 0; t < 11; t++) {
				centroids[s][t] = s+1;
			}
		}
		
		iterCnt = 0;
		while (iterCnt < maxIter) {
			iterCnt++;
			Configuration conf = new Configuration();
			Job job = Job.getInstance(conf, "k-means clustering");
			job.setJarByClass(KMeans.class);
			job.setMapperClass(ClusteringMapper.class);
			job.setReducerClass(ClusteringReducer.class);
			job.setMapOutputKeyClass(IntWritable.class);
			job.setMapOutputValueClass(DoubleArrayWritable.class);
			job.setOutputKeyClass(IntWritable.class);
			job.setOutputValueClass(Text.class);
			
			String outDir = args[1] + "iter" + iterCnt;		//setup output folder to store result after each iteration	
			FileInputFormat.addInputPath(job, new Path(args[0]));
			FileSystem fs = FileSystem.get(conf);
			
			//check whether output folder exists. If so, delete it.	
			if (fs.exists(new Path(outDir))) {				
				fs.delete(new Path(outDir),true);
			}
			FileOutputFormat.setOutputPath(job, new Path(outDir));
			job.waitForCompletion(true);
		}
		System.exit(0);
	}
	
	public static class DoubleArrayWritable
				extends ArrayWritable {
		
		public DoubleArrayWritable() {
			super(DoubleWritable.class);		//call for parent constructor, equivalent to ArrayWritable(DoubleWritable.class)
		}
		
		@Override		//override parent toArray() method, which returns array of writable
		public DoubleWritable[] toArray() {
			return (DoubleWritable[])super.toArray();		//type cast returned array to DoubleWritable
		}
	}
	
	public static class ClusteringMapper
				extends Mapper<LongWritable, Text, IntWritable, DoubleArrayWritable> {

		IntWritable clNum = new IntWritable();
		DoubleArrayWritable vecValue = new DoubleArrayWritable();
		
		public void map(LongWritable key, Text value, Context context) 
				throws IOException, InterruptedException {
			
			double minDis = Math.pow(10, 10);
			String strValue = value.toString().replaceAll(",", "\t");
			String[] split = strValue.split("\t");
			DoubleWritable[] vecArray = new DoubleWritable[11];
			
			//store each data point into an array, with index 0 to be point number
			for (int i = 0; i < 11; i++) {			
				vecArray[i]= new DoubleWritable(Double.parseDouble(split[i]));
			}
			
			//calculate distance to each centroid and assign point to centroid with shortest distance
			for (int j = 0; j < clCnt; j++) {			
				double disSumSq = 0;
				for (int k = 1; k < 11; k++) {
					disSumSq += Math.pow((centroids[j][k] - vecArray[k].get()),2);
				}
				if (minDis > Math.sqrt(disSumSq)) {
					minDis = Math.sqrt(disSumSq);
					clNum.set(j+1);
				}
			}
			
			vecValue.set(vecArray);
			context.write(clNum,vecValue);
		}
	}


	public static class ClusteringReducer
				extends Reducer<IntWritable, DoubleArrayWritable, IntWritable, Text> {

		IntWritable outKey = new IntWritable();
		Text outValue = new Text();
		
		public void reduce(IntWritable key, Iterable<DoubleArrayWritable> values, Context context) 
				throws IOException, InterruptedException {
			
			int cnt = 0;		//counter for number of points in each cluster
			int intKey = key.get();
			double[][] sumArray = new double[clCnt][10];			//store sum of each element for each data vector into 2D array
			DoubleWritable[] valArray = new DoubleWritable[11];
			String outString = "";
			
			//initialize sumArray with 0s
			for (int p = 0; p < clCnt; p++) {		
				for (int q = 0; q < 10; q++) {
					sumArray[p][q] = 0;
				}
			}
			
			for (DoubleArrayWritable val : values) {
				valArray = val.toArray();
				for (int m = 1; m < 11; m++) {
					sumArray[intKey-1][m-1] += valArray[m].get();		//update sumArray with sum of elements
				}
				cnt++;		//update data points count in each cluster
				outString += (int)(valArray[0].get()) + ",";		//store data point IDs
			}	
			
			for (int r = 1; r < 11; r++) {
				centroids[intKey-1][r] = sumArray[intKey-1][r-1]/cnt;
			}
			
			//check clustering results for debugging purposes
			System.out.println("Iteration: " + iterCnt + "| Cls: " + intKey + "| Count: " + cnt);
			String[] ctr = new String[4];
			
			for (int x = 0; x < clCnt; x++) {
				ctr[x] = "";
			}
			
			for (int w = 0; w < 11; w++) {
				ctr[intKey-1] += centroids[intKey-1][w] + ", ";
			}
			
			System.out.println(ctr[intKey-1]);

			outKey.set(intKey);
			outValue.set(outString);
			context.write(outKey, outValue);
		}
	}
}
