package neuralnetwork;

import org.jblas.DoubleMatrix;

/**
 * ����ڵ���Ҫ�������ݣ�ÿ��NeuronΪһ��ֵ
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
