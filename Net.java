import java.util.ArrayList;
import java.util.List;

public class Net {
    private List<List<Neuron>> layers;
    private double error;
    private double recentAverageError;
    private double recentAverageSmoothingFactor;

    public Net(List<Integer> topology) {
        layers = new ArrayList<>();
        int numLayers = topology.size();
        for (int layerNum = 0; layerNum < numLayers; layerNum++) {
            List<Neuron> layer = new ArrayList<>();
            int numOutputs = layerNum == topology.size() - 1 ? 0 : topology.get(layerNum + 1);
            for (int neuronNum = 0; neuronNum <= topology.get(layerNum); neuronNum++) {
                layer.add(new Neuron(numOutputs, neuronNum));
            }
        }
        layers.get(layers.size() - 1).get(layers.size() - 1).setOutputVal(1.0);
    }

    public void feedForward(List<Double> inputVals) {
        if (inputVals.size() != layers.get(0).size() - 1)
            throw new IllegalArgumentException("Input values must match the number of neurons in the input layer");

        for (int i = 0; i < inputVals.size(); i++) {
            layers.get(0).get(i).setOutputVal(inputVals.get(i));
        }

        for (int layerNum = 1; layerNum < layers.size(); layerNum++) {
            for (int neuronNum = 0; neuronNum < layers.get(layerNum).size() - 1; neuronNum++) {
                layers.get(layerNum).get(neuronNum).feedForward(layers.get(layerNum - 1));
            }
        }
    }

    public void backProp(List<Double> targetVals) {
        List<Neuron> outputLayer = layers.get(layers.size() - 1);
        this.error = 0.0;
        for (int i = 0; i < outputLayer.size() - 1; i++) {
            double delta = targetVals.get(i) - outputLayer.get(i).getOutputVal();
            this.error += delta * delta;
        }
        this.error /= outputLayer.size() - 1;
        this.error = Math.sqrt(error);

        this.recentAverageError = (this.recentAverageError * this.recentAverageSmoothingFactor + this.error) / (this.recentAverageSmoothingFactor + 1.0);

        for (int i = 0; i < outputLayer.size() - 1; i++) {
            outputLayer.get(i).calcOutputGradients(targetVals.get(i));
        }

        for (int layerNum = layers.size() - 2; layerNum > 0; layerNum--) {
            List<Neuron> hiddenLayer = layers.get(layerNum);
            List<Neuron> nextLayer = layers.get(layerNum + 1);

            for (int i = 0; i < hiddenLayer.size(); i++) {
                hiddenLayer.get(i).calcHiddenGradients(nextLayer);
            }
        }

        for (int layerNum = layers.size() - 1; layerNum > 0; layerNum--) {
            List<Neuron> layer = layers.get(layerNum);
            List<Neuron> prevLayer = layers.get(layerNum - 1);

            for (int i = 0; i < layer.size() - 1; i++) {
                layer.get(i).updateInputWeights(prevLayer);
            }
        }
    }

    public void getResults(List<Double> resultVals) {
        resultVals.clear();
        for (int i = 0; i < layers.get(layers.size() - 1).size() - 1; i++) {
            resultVals.add(layers.get(layers.size() - 1).get(i).getOutputVal());
        }
    }

    public double getRecentAverageError() {
        return recentAverageError;
    }
}
