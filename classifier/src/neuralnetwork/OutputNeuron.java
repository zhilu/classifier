package neuralnetwork;

import java.util.Arrays;

import org.jblas.DoubleMatrix;

public class OutputNeuron extends Neuron {

	private HiddenNeuron[] hiddenNeurons;
	
	public DoubleMatrix delta;
	private DoubleMatrix y;
	int nNeuron = 0;
	int ndata = 0;
	
	public OutputNeuron(HiddenNeuron[] hiddenNeurons) {
		this.hiddenNeurons = hiddenNeurons;
		nNeuron = hiddenNeurons.length;
		ndata = NetworkModel.obs;
		weights = DoubleMatrix.rand(nNeuron, 1);
	}

	public double getE(){
		double eplison = Double.MAX_VALUE;
		if(outputs.rows == y.rows && y.columns==1 && outputs.columns==1){
			eplison= y.sub(outputs).mul(y.sub(outputs)).sum();
		}
		return eplison;
		
	}
	
	public void calculateinputs() {
		DoubleMatrix temp = new DoubleMatrix(ndata,nNeuron);
		for (int i = 0; i < hiddenNeurons.length; i++) {
			temp.putColumn(i, hiddenNeurons[i].outputs);;
		}
		inputs=temp.mmul(weights);
		System.out.println();
	}

	public void load() {
		outputs = Utils.sigmoid(inputs);
	}

	public void adjustweights() {
		delta = new DoubleMatrix(ndata, 1);
		delta = y.sub(outputs).mul(outputs).mul(outputs.neg().add(1));
		weights = weights.add(delta.mmul(outputs).mul(Utils.ETA).neg());
	}

	@Override
	public String toString() {
		return "OutputNeuron [hiddenNeurons=" + Arrays.toString(hiddenNeurons) + ", delta=" + delta + ", y=" + y
				+ ", nNeuron=" + nNeuron + ", ndata=" + ndata + "]";
	}

	public void setY(DoubleMatrix y2) {
		this.y = y2;
	}

	
	
}
