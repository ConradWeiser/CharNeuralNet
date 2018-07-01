import org.apache.commons.io.FileUtils;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.charset.Charset;
import java.util.Random;

public class StoryModeler {


    public static void main(String[] args) throws Exception {

        System.out.println("Test");

        int lstmLayerSize = 200;        // How many units are in each LSTM layer
        int miniBatchSize = 32;        //Size of batch to use while training (Previously 32)
        int exampleLength = 1500;       // Length of each training example sequence to use. (Increase this probably) (Previously 3000)
        int tbpttLength = 50;           //Truncated backpropigation through time (Do parameter updates every X characters) (Previous 50)
        int numEpochs = 30;              //Number of training epochs
        int generateSamplesEveryNMiniBatches = 10;  // How freqeuntly should this generate a sample from the network. (1000 characters / 50 tbptt length: 20 parameter per batch
        int nSamplesToGenerate = 4;        // Number of writing samples to generate after eacy training epoch
        int nCharactersToSample = 500;      //Length of each sample to generate
        String generalizationInitializeation = null;    //Optional character initialization. Random if null.

        long currTime = System.currentTimeMillis();
        System.out.println("Using rng-seed of: " + currTime);

        Random rng = new Random(currTime);

        // Get a DataSetIterator that handles our network
        CharacterIterator dataIter = getInputFromFile(miniBatchSize, exampleLength, currTime);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(currTime)
                .l2(0.002)
                .dropOut(0.5)
                .weightInit(WeightInit.XAVIER)
                .updater(new RmsProp(0.1))
                .list()
                .layer(0, new LSTM.Builder().nIn(dataIter.inputColumns()).nOut(lstmLayerSize)
                        .activation(Activation.TANH).build())
                .layer(1, new LSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
                        .activation(Activation.TANH).build())
                .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)
                        .nIn(lstmLayerSize).nOut(dataIter.totalOutcomes()).build())

                .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
                .pretrain(false)
                .backprop(true)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        //net.setListeners(new ScoreIterationListener(1));
        UIServer uiServer = UIServer.getInstance();

        // Configure the data in memory
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        net.setListeners(new StatsListener(statsStorage));

        // Print the number of parameters in the network and for each layer
        org.deeplearning4j.nn.api.Layer[] layers = net.getLayers();
        int totalNumParams = 0;

        for(int i = 0; i < layers.length; i++)
        {
            int nParams = layers[i].numParams();
            System.out.println("Number of parameters in layer " + i + ": " + nParams);
            totalNumParams += nParams;
        }

        System.out.println("Total number of network parameters: " + totalNumParams);

        // Do the training, and then generate and print samples from the network
        int miniBatchNumber = 0;
        for(int i = 0; i < numEpochs; i++)
        {
            while(dataIter.hasNext())
            {
                DataSet ds = dataIter.next();
                net.fit(ds);

                if(++miniBatchNumber % generateSamplesEveryNMiniBatches == 0)
                {
                    System.out.println("--------------");
                    System.out.println("Completed " + miniBatchNumber + " minibatches of size " + miniBatchSize + "x" + exampleLength + " characters");
                    System.out.println("Sampling characters from network given initialization \"" + (generalizationInitializeation == null ? "" : generalizationInitializeation) + "\"");

                    String[] samples =  sampleCharactersFromNetwork(generalizationInitializeation,net,dataIter,rng,nCharactersToSample,nSamplesToGenerate);

                    for(int j = 0; j < samples.length; j++)
                    {
                        System.out.println("---- Sample " + j + " ----");
                        System.out.println(samples[j] + "\n");
                    }
                }
            }

            dataIter.reset();
        }

        System.out.println("Complete");
    }


    public static CharacterIterator getInputFromFile(int miniBatchSize, int sequenceLength, long rngSeed) throws Exception
    {
        String tempDir = System.getProperty("java.io.tmpdir");
        String fileLocation = tempDir + "/LearningFile.txt";
        File f = new File(fileLocation);

        if(!f.exists())
        {
            System.err.println("Missing file at path: " + f.getAbsolutePath());
        }

        char[] validCharacters = CharacterIterator.getMinimalCharacterSet();
        return new CharacterIterator(fileLocation, Charset.forName("UTF-8"),
                miniBatchSize, sequenceLength, validCharacters, new Random(rngSeed));
    }

    private static String[] sampleCharactersFromNetwork(String initialization, MultiLayerNetwork net, CharacterIterator iter,
                                                        Random rng, int charactersToSample, int numSamples)
    {
        // Set up init. If none, use a random character
        if(initialization == null || initialization == "")
        {
            initialization = String.valueOf(iter.getRandomCharacter());
        }

        // Create input for init
        INDArray initializationInput = Nd4j.zeros(numSamples, iter.inputColumns(), initialization.length());
        char[] init = initialization.toCharArray();

        for(int i = 0; i < init.length; i++)
        {
            int idx = iter.convertCharacterToIndex(init[i]);
            for(int j = 0; j < numSamples; j++)
            {
                initializationInput.putScalar(new int[] {j, idx, i}, 1.0f);
            }
        }

        StringBuilder[] builder = new StringBuilder[numSamples];
        for(int i = 0; i < numSamples; i++)
        {
            builder[i] = new StringBuilder(initialization);
        }

        // Sample from the network one character at a time
        net.rnnClearPreviousState();
        INDArray output = net.rnnTimeStep(initializationInput);
        output = output.tensorAlongDimension(output.size(2) - 1, 1, 0);

        for(int i = 0; i < charactersToSample; i++)
        {
            INDArray nextInput = Nd4j.zeros(numSamples, iter.inputColumns());

            for(int s = 0; s < numSamples; s++)
            {
                double[] outputProbDistribution = new double[iter.totalOutcomes()];
                for(int j = 0; j < outputProbDistribution.length; j++)
                {
                    outputProbDistribution[j] = output.getDouble(s, j);
                }
                
                int sampledCharacterIdx = sampleFromDistribution(outputProbDistribution, rng);
                
                nextInput.putScalar(new int[] {s, sampledCharacterIdx}, 1.0f);
                builder[s].append(iter.convertIndexToCharacter(sampledCharacterIdx));
            }
            
            output = net.rnnTimeStep(nextInput);
        }
        
        String[] out = new String[numSamples];
        for(int i = 0; i < numSamples; i++)
        {
            out[i] = builder[i].toString();
        }
        
        return out;
    }

    private static int sampleFromDistribution(double[] distribution, Random rng) {
        double d = 0.0;
        double sum = 0.0;
        for( int t=0; t<10; t++ ) {
            d = rng.nextDouble();
            sum = 0.0;
            for( int i=0; i< distribution.length; i++ ){
                sum += distribution[i];
                if( d <= sum ) return i;
            }
            //If we haven't found the right index yet, maybe the sum is slightly
            //lower than 1 due to rounding error, so try again.
        }
        //Should be extremely unlikely to happen if distribution is a valid probability distribution
        throw new IllegalArgumentException("Distribution is invalid? d="+d+", sum="+sum);
    }
}
