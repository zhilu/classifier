package neuralnetwork;

import org.jblas.DoubleMatrix;

public class NetworkModel extends Thread {
	public InputNeuron[] InputNeurons;
	public HiddenNeuron[] HiddenNeurons;
	public OutputNeuron[] OutputNeurons;

	public DoubleMatrix dataset;
	public DoubleMatrix Y;

	public static int obs = 0; // 设置一次，都可以访问

	int hiden = 3; // 默认的隐藏层节点数

	public NetworkModel(DoubleMatrix x, DoubleMatrix y, int n) {
		dataset = x;
		this.Y = y;
		obs = n;
	}

	public static void main(String[] args) throws Exception {

		DoubleMatrix dat = Utils.getData("./src/neuralnetwork/only2.csv");

		DoubleMatrix x = dat.getRange(0, dat.rows, 0, dat.columns - 1);
		DoubleMatrix y = dat.getColumn(dat.columns - 1);

		NetworkModel model = new NetworkModel(x, y, x.rows);

		model.buildNetwork();

		model.forward();
		while (model.OutputNeurons[0].getE() > 0.001) {
			model.backward();
			model.forward();
		}

	}

	public int getHiden() {
		return hiden;
	}

	public void setHiden(int hiden) {
		this.hiden = hiden;
	}

	public void buildNetwork() {

		InputNeurons = new InputNeuron[dataset.columns];
		for (int i = 0; i < InputNeurons.length; i++) {
			InputNeurons[i] = new InputNeuron(dataset.getColumn(i));
		}

		HiddenNeurons = new HiddenNeuron[hiden];
		for (int i = 0; i < HiddenNeurons.length; i++) {
			HiddenNeurons[i] = new HiddenNeuron(InputNeurons);
		}

		OutputNeurons = new OutputNeuron[1];
		OutputNeurons[0] = new OutputNeuron(HiddenNeurons);
		OutputNeurons[0].setY(Y);

		for (int i = 0; i < HiddenNeurons.length; i++) {
			HiddenNeurons[i].setOutputNeurons(OutputNeurons);
		}

	}

	// 重构线程run 方法, 负责读取样本, 控制各层神经元前向学习、误差反传和权值调整
	public void run() {

	}

	public void optimize() {
	}// 优化神经网络结构, 动态增删隐结点

	private void backward() {
		OutputNeurons[0].adjustweights();
		for (int i = 0; i < HiddenNeurons.length; i++) {
			HiddenNeurons[i].adjustweights();
		}

	}

	private void forward() {

		for (int i = 0; i < HiddenNeurons.length; i++) {
			HiddenNeurons[i].calculateinputs();
			HiddenNeurons[i].load();
		}
		
		System.out.println();
		OutputNeurons[0].calculateinputs();
		OutputNeurons[0].load();
	}
}
