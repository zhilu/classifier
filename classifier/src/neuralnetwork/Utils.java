package neuralnetwork;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Vector;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class Utils {
	private Utils() {}

	public static double ETA  = 0.1;
	
	public static DoubleMatrix getData(String file) throws Exception {
		DoubleMatrix result = null;

		@SuppressWarnings("resource")
		BufferedReader reader = new BufferedReader(new FileReader(file));
		String line = reader.readLine();
		Vector<double[]> examples = new Vector<double[]>();

		while ((line = reader.readLine()) != null) {
			String[] strs = line.split(",| ");
			double[] point = new double[strs.length];
			for (int i = 0; i < point.length; i++) {
				point[i] = Double.parseDouble(strs[i]);
			}
			examples.add(point);
		}
		result = new DoubleMatrix(examples.size(), examples.firstElement().length);
		for (int i = 0; i < examples.size(); i++) {
			result.putRow(i, new DoubleMatrix(examples.get(i)));
		}
		return result;
	}
	
	public static DoubleMatrix sigmoid(DoubleMatrix x){
		DoubleMatrix result = new DoubleMatrix();
		result = x;
		result = MatrixFunctions.expi(result.mul(-1));
		result = result.add(1);
		result = MatrixFunctions.powi(result, -1);
		return result;
	}
}
