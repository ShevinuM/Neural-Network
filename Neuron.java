import java.util.ArrayList;
import java.util.List;
import java.util.Random;

class Neuron {
    private static final Random random = new Random();
    private static final double eta = 0.15; // learning rate
    private static final double alpha = 0.5; // momentum
    private double outputVal;
    private List<Connection> outputWeights;
    private int myIndex;
    private double gradient;

    public Neuron(int numOutputs, int myIndex) {
        this.myIndex = myIndex;
        this.outputWeights = new ArrayList<>();
        for (int i = 0; i < numOutputs; i++) {
            Connection conn = new Connection();
            conn.weight = random.nextDouble();
            outputWeights.add(conn);
        }
    }

    public static double transferFunction(double x) {
        return Math.tanh(x);
    }

    public static double transferFunctionDerivative(double x) {
        return 1.0 - x * x;
    }

    public void setOutputVal(double outputVal) {
        this.outputVal = outputVal;
    }

    public double getOutputVal() {
        return outputVal;
    }

    public void feedForward(List<Neuron> prevLayer) {
        double sum = 0.0;
        for (int i = 0; i < prevLayer.size(); i++) {
            sum += prevLayer.get(i).getOutputVal() * prevLayer.get(i).outputWeights.get(myIndex).weight;
        }
        outputVal = Neuron.transferFunction(sum);
    }

    public void calcOutputGradients(double targetVal) {
        double delta = targetVal - outputVal;
        this.gradient = delta * Neuron.transferFunctionDerivative(outputVal);
    }

    public void calcHiddenGradients(List<Neuron> nextLayer) {
        double dow = sumDOW(nextLayer);
        this.gradient = dow * Neuron.transferFunctionDerivative(outputVal);
    }

    public double sumDOW(List<Neuron> nextLayer) {
        double sum = 0.0;
        for (int n = 0; n < nextLayer.size() - 1; n++) {
            sum += outputWeights.get(n).weight * nextLayer.get(n).gradient;
        }
        return sum;
    }

    public void updateInputWeights(List<Neuron> prevLayer) {
        for (int n = 0; n < prevLayer.size(); n++) {
            Neuron neuron = prevLayer.get(n);
            double oldDeltaWeight = neuron.outputWeights.get(myIndex).deltaWeight;
            double newDeltaWeight = eta * neuron.getOutputVal() * gradient + alpha * oldDeltaWeight;
            neuron.outputWeights.get(myIndex).deltaWeight = newDeltaWeight;
        }
    }
    // Other methods like feedForward, calcOutputGradients, etc.
    // These methods will need to be implemented with the same logic as in the C++ code
}
