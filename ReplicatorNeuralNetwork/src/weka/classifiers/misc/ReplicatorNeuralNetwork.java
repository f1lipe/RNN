/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    IsolationForest.java
 *    Copyright (C) 2012-16 University of Waikato, Hamilton, New Zealand
 *
 */
package weka.classifiers.misc;

import java.io.Serializable;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;
import java.util.logging.Level;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;
import org.neuroph.util.data.norm.MaxMinNormalizer;

import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Debug.Log;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;

/**
 * <!-- globalinfo-start --> Implements the Replicator Neural Network method for anomaly
 * detection.<br>
 * <br>
 * Note that this classifier is designed for anomaly detection, it is not
 * designed for solving two-class or multi-class classification problems!<br>
 * <br>
 * The data is expected to have have a class attribute with one or two values,
 * which is ignored at training time. The distributionForInstance() method
 * returns (1 - anomaly score) as the first element in the distribution, the
 * second element (in the case of two classes) is the anomaly score.<br>
 * <br>
 * To evaluate performance of this method for a data set where anomalies are
 * known, simply code the anomalies using the class attribute: normal cases
 * should correspond to the first value of the class attribute, anomalies to the
 * second one.<br>
 * <br>
 *  * 
 * <!-- globalinfo-end -->
 * 
 * <!-- technical-bibtex-start --> BibTeX:

 * </pre>
 * 
 * <br>
 * <br>
 * <!-- technical-bibtex-end -->
 * 
 * <!-- options-start --> Valid options are:
 * <p>
 * 
 * <pre>
 *  -H &lt;num&gt;
 *  Hidden layer offset.
 *  (default -1)
 * </pre>
 * 
 * <pre>
 *  -I &lt;num&gt;
 *  Max epochs in training phase
 *  (default 1000)
 * </pre>
 * 
 * <pre>
 *  -D &lt;num&gt;
 *  Classificator error distance threshold
 *  (default 0.1)
 * </pre>
 * 
 * <pre>
 *  -E &lt;num&gt;
 *  Max error to stop training
 *  (default 0.001)
 * </pre>
 * 
 * <pre>
 *  -output-debug-info
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console
 * </pre>
 * 
 * <pre>
 *  -do-not-check-capabilities
 *  If set, classifier capabilities are not checked before classifier is built
 *  (use with caution).
 * </pre>
 * 
 * <pre>
 *  -num-decimal-places
 *  The number of decimal places for the output of numbers in the model (default 2).
 * </pre>
 * 
 * 
 * <!-- options-end -->
 * 
 * @author Filipe (filipe@rocketmail.com)
 * @version $Revision: 12345 $
 */
public class ReplicatorNeuralNetwork extends AbstractClassifier implements TechnicalInformationHandler, Serializable {

	// For serialization
	private static final long serialVersionUID = 5586634623177348798L;

	// Log messages
	private Log log = new Log();

	// Trained neural network
	protected NeuralNetwork<?> neuralNetwork;
	private int rowSize;

	// Parameters
	private int maxEpochs = 1000;

	private int hiddenLayerOffset = -1;

	private double distanceThreshold = 0.1d;

	private double maxError = 0.001;

	/**
	 * Returns a string describing this filter
	 */
	public String globalInfo() {

		return "Implements the Replicator Neural Network method for anomaly detection.\n\n"
				+ "Note that this classifier is designed for anomaly detection, it is not designed for solving "
				+ "two-class or multi-class classification problems!\n\n"
				+ "The data is expected to have have a class attribute with one or two values, "
				+ "which is ignored at training time. The distributionForInstance() "
				+ "method returns (1 - anomaly score) as the first element in the distribution, "
				+ "the second element (in the case of two classes) is the anomaly score.\n\nTo evaluate performance "
				+ "of this method for a dataset where anomalies are known, simply "
				+ "code the anomalies using the class attribute: normal cases should "
				+ "correspond to the first value of the class attribute, anomalies to " + "the second one."
				+ "\n\nFor more information, see:\n\n" + getTechnicalInformation().toString();
	}

	/**
	 * Returns an instance of a TechnicalInformation object, containing detailed
	 * information about the technical background of this class, e.g., paper
	 * reference or book this class is based on.
	 * 
	 * @return the technical information about this class
	 */
	@Override
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation result;

		result = new TechnicalInformation(Type.INPROCEEDINGS);
		result.setValue(Field.AUTHOR, "@f1lipe");
		result.setValue(Field.TITLE, "Replicator Neural Network");
		result.setValue(Field.YEAR, "2018");
		return result;
	}

	/**
	 * Returns the Capabilities of this filter.
	 */
	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();

		// attributes
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.DATE_ATTRIBUTES);

		// class
		// result.enable(Capability.UNARY_CLASS);
		result.enable(Capability.BINARY_CLASS);
		// result.enable(Capability.MISSING_CLASS_VALUES);

		// instances
		result.setMinimumNumberInstances(100);

		return result;
	}

	/**
	 * Returns brief description of the classifier.
	 */
	@Override
	public String toString() {

		if (neuralNetwork == null) {
			return "No model built yet.";
		} else {
			return "Neural network: " + rowSize + " | " + (rowSize + hiddenLayerOffset) + " | " + rowSize;
		}
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String subsampleSizeTipText() {
		return "The size of the subsample used to build each tree.";
	}

	/**
	 * Lists the command-line options for this classifier.
	 * 
	 * @return an enumeration over all possible options
	 */
	@Override
	public Enumeration<Option> listOptions() {

		Vector<Option> newVector = new Vector<Option>();

		newVector.addElement(new Option(
				"\tThe difference between the number of neurons in the input/output layers and the hidden layer (default "
						+ hiddenLayerOffset + ").",
				"H", 1, "-H <difference number>"));

		newVector.addElement(new Option("\tThe maximum number of epochs (default " + maxEpochs + ").", "I", 1,
				"-I <the maximum number of epochs>"));

		newVector.addElement(new Option(
				"\tThe threshold distance to classify an instance as annomaly (default " + distanceThreshold + ").",
				"D", 1, "-D <distance threshold>"));

		newVector.addElement(new Option("\tThe max error to stop training (default " + maxError + ").", "E", 1,
				"-E <maximum error>"));

		newVector.addAll(Collections.list(super.listOptions()));

		return newVector.elements();
	}

	/**
	 * Gets options from this classifier.
	 * 
	 * @return the options for the current setup
	 */
	@Override
	public String[] getOptions() {

		Vector<String> result = new Vector<String>();

		result.add("-H");
		result.add(String.valueOf(hiddenLayerOffset));

		result.add("-I");
		result.add(String.valueOf(maxEpochs));

		result.add("-D");
		result.add(String.valueOf(distanceThreshold));

		result.add("-E");
		result.add(String.valueOf(maxError));

		Collections.addAll(result, super.getOptions());

		return result.toArray(new String[result.size()]);
	}

	/**
	 * Parses a given list of options.
	 * <p>
	 * 
	 * <!-- options-start --> Valid options are:
	 * <p>
	 * 
	 * <pre>
	 *  -H &lt;num&gt;
	 *  Hidden layer offset to input and output layer.
	 *  (default -1)
	 * </pre>
	 * 
	 * <pre>
	 *  -I &lt;num&gt;
	 *  Max epochs in training phase
	 *  (default 1000)
	 * </pre>
	 * 
	 * <pre>
	 *  -D &lt;num&gt;
	 *  Classificator error distance threshold
	 *  (default 0.1)
	 * </pre>
	 * 
	 * <pre>
	 *  -E &lt;num&gt;
	 *  Max error to stop training
	 *  (default 0.001)
	 * </pre>
	 * 
	 * <pre>
	 *  -output-debug-info
	 *  If set, classifier is run in debug mode and
	 *  may output additional info to the console
	 * </pre>
	 * 
	 * <pre>
	 *  -do-not-check-capabilities
	 *  If set, classifier capabilities are not checked before classifier is built
	 *  (use with caution).
	 * </pre>
	 * 
	 * <pre>
	 *  -num-decimal-places
	 *  The number of decimal places for the output of numbers in the model (default 2).
	 * </pre>
	 * 
	 * <!-- options-end -->
	 * 
	 * @param options
	 *            the list of options as an array of strings
	 * @throws Exception
	 *             if an option is not supported
	 */
	@Override
	public void setOptions(String[] options) throws Exception {
		String optionString;

		optionString = Utils.getOption('H', options);
		if (optionString.length() != 0) {
			hiddenLayerOffset = Integer.parseInt(optionString);
		}

		optionString = Utils.getOption('I', options);
		if (optionString.length() != 0) {
			maxEpochs = Integer.parseInt(optionString);
		}

		optionString = Utils.getOption('D', options);
		if (optionString.length() != 0) {
			distanceThreshold = Double.parseDouble(optionString);
		}

		optionString = Utils.getOption('E', options);
		if (optionString.length() != 0) {
			maxError = Double.parseDouble(optionString);
		}

		super.setOptions(options);

		Utils.checkForRemainingOptions(options);
	}

	/**
	 * Builds the classificator.
	 */
	@Override
	public void buildClassifier(Instances data) throws Exception {
		log.log(Level.INFO, "Starting building classifier...");
		// Can classifier handle the data?
		getCapabilities().testWithFail(data);

		// Evaluate the network length.
		rowSize = data.numAttributes() - 1;
		// Set the hidden layer size.
		int hiddenLayerSize = rowSize - hiddenLayerOffset;

		// Generate the neural network
		NeuralNetwork<BackPropagation> neuralNetwork = new MultiLayerPerceptron(TransferFunctionType.SIGMOID, 
				rowSize, hiddenLayerSize, rowSize);

		DataSet trainingSet = new DataSet(rowSize, rowSize);

		for (Instance instance : data) {
			if (m_Debug) {
				log.log(Level.FINEST, "Train instance: " + instance);
			}
			double[] rowArray = new double[rowSize];
			for (int i = 0; i < rowSize; i++) {
				rowArray[i] = instance.value(i);
			}
			trainingSet.addRow(rowArray, rowArray);
		}

		MaxMinNormalizer normalizer = new MaxMinNormalizer();
		normalizer.normalize(trainingSet);
		BackPropagation backPropagation = new BackPropagation();

		backPropagation.setMaxIterations(maxEpochs);
		backPropagation.setMaxError(maxError);
		neuralNetwork.learn(trainingSet, backPropagation);

		log.log(Level.INFO, "Total error:" + backPropagation.getTotalNetworkError());

		if (backPropagation.getTotalNetworkError() == Double.NaN)
			throw new Exception(
					"O treinamento do modelo nao convergiu. Verifique o arquivo de log e os dados de entrada.");

		log.log(Level.INFO, "Number of ephocs:" + backPropagation.getCurrentIteration());
		log.log(Level.INFO, "Fim do treinamento: " + trainingSet.size() + " samples processed.");
		this.neuralNetwork = neuralNetwork;
		log.log(Level.INFO, "Iniciando testes...");
	}

	/**
	 * Returns distribution of scores.
	 * 
	 * @throws Exception
	 */
	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		if (instance.numClasses() != 2)
			throw new Exception("Wrong class number: " + instance.numClasses());

		double[] rowArray = new double[rowSize];
		for (int i = 0; i < rowSize; i++) {
			rowArray[i] = instance.value(i);
		}
		neuralNetwork.setInput(rowArray);
		neuralNetwork.calculate();
		double euclidianDistance = getEuclidianDistance(rowArray, neuralNetwork.getOutput());

		double[] scores = new double[instance.numClasses()];

		if (m_Debug) {			
			log.log(Level.FINE, "Test Instance: " + instance + " | " 
					+ "euclidianDistance: " + euclidianDistance );		
		}
					
		// scores[1] = 1.0 means anomaly.
		if (euclidianDistance > distanceThreshold) {			
			scores[1] = 1.0;
		} else {
			scores[1] = 0.0;
		}
		scores[0] = 1.0 - Math.abs(scores[1]);

		return scores;
	}

	/**
	 * Main method for this class.
	 */
	public static void main(String[] args) {
		runClassifier(new ReplicatorNeuralNetwork(), args);
	}

	/**
	 * Calculate the Euclidian distance between two vectors.
	 * 
	 * @param input
	 * @param output
	 * @return the euclidian distance
	 */
	public double getEuclidianDistance(double[] input, double[] output) {
		if (input.length != output.length)
			throw new RuntimeException("Different input and output lenghts");

		double powerSum = 0.0;
		int i = 0;
		String logMessage = "";
		for (double outputValue : output) {
			powerSum += Math.pow((outputValue - input[i]), 2);
			logMessage += outputValue + ":" + input[i] + " | ";
			i++;
		}
		if (m_Debug) {
			log.log(Level.FINE, "euclidianDistance: " + logMessage );	
		}
		return Math.sqrt(powerSum);
	}

}
