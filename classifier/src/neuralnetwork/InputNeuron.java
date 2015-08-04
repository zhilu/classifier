package neuralnetwork;

import org.jblas.DoubleMatrix;

/**
 * 输入节点需要输入数据，每个Neuron为一列值
 * 
 * @param data
 */
public class InputNeuron extends Neuron {
	
	public InputNeuron(DoubleMatrix data) {
		outputs = data;
		
	}

	@Override
	public String toString() {
		return "InputNeuron [outputs=" + outputs + "]";
	}

	
	
}
