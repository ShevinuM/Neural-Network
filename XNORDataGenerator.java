import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class XNORDataGenerator {

    private final int numberOfSamples;
    private final Random random;

    public XNORDataGenerator(int numberOfSamples) {
        this.numberOfSamples = numberOfSamples;
        this.random = new Random();
    }

    public void generateToFile(String filename) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            for (int i = 0; i < numberOfSamples; i++) {
                int input1 = random.nextInt(2); // Random 0 or 1
                int input2 = random.nextInt(2); // Random 0 or 1
                int output = (input1 == input2) ? 1 : 0; // XNOR operation

                writer.write("in: " + input1 + " " + input2 + "\n");
                writer.write("out: " + output + "\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        XNORDataGenerator generator = new XNORDataGenerator(1000);
        generator.generateToFile("xnor_training_data.txt");
    }
}