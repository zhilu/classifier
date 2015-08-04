package neuralnetwork;

import java.util.Arrays;

import org.jblas.DoubleMatrix;

public class HiddenNeuron extends Neuron {


	private Neuron[] InputNeurons;
	private Neuron[] OutputNeurons;
	private DoubleMatrix delta;

	int nNeuron = 0;
	int ndata = 0;

	public void setOutputNeurons(OutputNeuron [] outputNeurons) {
		OutputNeurons = outputNeurons;
	}

	public HiddenNeuron(InputNeuron[] InputNeurons) {
		this.InputNeurons = InputNeurons;
		nNeuron = InputNeurons.length;
		ndata = NetworkModel.obs;
		weights = DoubleMatrix.rand(nNeuron, 1);

	}

	@Override
	public void calculateinputs() {
		DoubleMatrix temp = new DoubleMatrix(ndata, nNeuron);
		for (int i = 0; i < InputNeurons.length; i++) {
			temp.putColumn(i, InputNeurons[i].outputs);;
		}
		inputs = temp.mmul(weights);

		System.out.println(weights);
	}

	@Override
	public void load() {
		outputs = Utils.sigmoid(inputs);
		System.out.println(outputs);
	}

	@Override
	public void adjustweights() {
		delta = new DoubleMatrix(ndata, 1);
		delta = outputs.mul(outputs.neg().add(1)).mul(weights.mul(OutputNeurons[0].deltas));
		weights = weights.add(delta.mul(outputs).neg().mul(Utils.ETA));
	}

	@Override
	public String toString() {
		return "HiddenNeuron [InputNeurons=" + Arrays.toString(InputNeurons) + ", OutputNeurons="
				+ Arrays.toString(OutputNeurons) + ", delta=" + delta + ", nNeuron=" + nNeuron + ", ndata=" + ndata
				+ "]";
	}
	
	

}
