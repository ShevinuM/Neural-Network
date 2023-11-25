import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        TrainingData trainData = new TrainingData("xnor_training_data.txt");

        // Example topology - adjust as needed
        List<Integer> topology = new ArrayList<>();
        trainData.getTopology(topology);

        Net myNet = new Net(topology);

        List<Double> inputVals = new ArrayList<>();
        List<Double> targetVals = new ArrayList<>();
        List<Double> resultVals = new ArrayList<>();
        int trainingPass = 0;

        while (!trainData.isEof()) {
            trainingPass++;
            System.out.println("\nPass " + trainingPass);

            // Get new input data and feed it forward
            if (trainData.getNextInputs(inputVals) != topology.get(0)) {
                break;
            }
            showVectorVals("Inputs:", inputVals);
            myNet.feedForward(inputVals);

            // Collect the net's actual output results
            myNet.getResults(resultVals);
            showVectorVals("Outputs:", resultVals);

            // Train the net what the outputs should have been
            trainData.getTargetOutputs(targetVals);
            showVectorVals("Targets:", targetVals);
            assert targetVals.size() == topology.get(topology.size() - 1);

            myNet.backProp(targetVals);

            // Report how well the training is working
            System.out.println("Net recent average error: " + myNet.getRecentAverageError());
        }

        System.out.println("\nDone");
    }

    private static void showVectorVals(String label, List<Double> v) {
        System.out.print(label + " ");
        for (Double val : v) {
            System.out.print(val + " ");
        }
        System.out.println();
    }
}
