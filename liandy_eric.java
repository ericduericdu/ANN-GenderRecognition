import java.util.*;
import java.io.*;

/**
 * A program that predicts the gender of a person in a photo using Artificial Neural Network.
 * This is the third assignment of the Artificial Intelligence class at UC Davis.
 * @author L. Hardikoesoemo, E. Du
 */

public class liandy_eric {
   public static ArrayList<Image> trainingImages;
   public static ArrayList<Image> testingImages;
   // public static ArrayList<Image> femaleImages;
   // public static ArrayList<Image> maleImages;

   public static final int LENGTH = 128;
   public static final int WIDTH = 120;
   public static final int DIMENSION = LENGTH * WIDTH;
   public static final int FOLDS = 5;
   public static final int EPOCHS = 500;
   public static final double LEARNING_RATE = 0.12;
   private static Random rand;

   public static class Image {
      public int actualGender = 99;   // 99 default, 1 Male, 0 Female. actualGender is used only for training.

      public double[] imagePixel;

      public Image(double[] pixel, int gender) {
         imagePixel = pixel;
         actualGender = gender;
      }

      // Used for testing data.
      public Image(double[] pixel) {
         imagePixel = pixel;
      }

      public String toString() {
         return Arrays.toString(imagePixel);
      }
   }

   public static class NeuralNetwork {
      private static double[][][] weights;
      private static final int LAYER = 3;
      private static final double INITIAL_WEIGHT = 0.5;
      private static final int HIDDEN_NODES_EACH_LAYER = 4;
      private static final int INPUT_NO = DIMENSION;
      private static double[][] nodeValues;
      private static double[][] errors;

      public NeuralNetwork() {
         rand = new Random();
         rand.setSeed(123);
         weights = new double[LAYER-1][][];
         for(int i = 0; i < LAYER-1; i++){    // LAYER-1 because the last layer doesn't have any outgong weight

            if(i == 0){ // First layer, has lots of outgoing weights from the inputs
               weights[i] = new double[HIDDEN_NODES_EACH_LAYER][];  // Create layer i with the # of specified nodes.
               for(int j = 0; j < HIDDEN_NODES_EACH_LAYER; j++){
                  weights[i][j] = new double[INPUT_NO];
                  for(int k = 0; k < INPUT_NO; k++){
                     weights[i][j][k] = rand.nextDouble()/5 + 0.1;   // add 0.1 to avoid the weight being set to 0.
                  }
               }
            } else if(i == LAYER-2) {  // Weights from the last hidden layer to the output
               weights[i] = new double[1][];   // [2] because we only have 2 outputs (Male & Female)
               for(int j = 0; j < 1; j++){
                  weights[i][j] = new double[HIDDEN_NODES_EACH_LAYER];
                  for(int k = 0; k < HIDDEN_NODES_EACH_LAYER; k++){
                     weights[i][j][k] = rand.nextDouble()/5 + 0.1;
                  }
               }
            } else {  // Weights from the last hidden layer to the output
               weights[i] = new double[HIDDEN_NODES_EACH_LAYER][];  // Create layer i with the # of specified nodes.
               for(int j = 0; j < HIDDEN_NODES_EACH_LAYER; j++){
                  weights[i][j] = new double[HIDDEN_NODES_EACH_LAYER];
                  for(int k = 0; k < HIDDEN_NODES_EACH_LAYER; k++){
                     weights[i][j][k] = rand.nextDouble()/5 + 0.1;
                  }
               }
            }
         }

         nodeValues = new double[LAYER][];
         for(int k = 0; k < LAYER; k++) {
            if(k == LAYER-1) {   // The output nodes
               nodeValues[k] = new double[1];   // 1 output node
            } else if(k == 0) {
               nodeValues[k] = new double[DIMENSION];
            } else {
               nodeValues[k] = new double[HIDDEN_NODES_EACH_LAYER];
            }
         }

         // The first layer of nodeValues contains the pixel value.

         errors = new double[LAYER][];
         for(int m = 0; m < LAYER; m++) {
            if(m == LAYER-1) {   // The output nodes
               errors[m] = new double[1];   // 1 output node

            } else if(m == 0) {
               errors[m] = new double[DIMENSION];
            } else {
               errors[m] = new double[HIDDEN_NODES_EACH_LAYER];

            }
         }
      }

      public void train(ArrayList<Image> images, int epochs){
         // long seed = System.nanoTime();
         // rand = new Random();
         // Collections.shuffle(images, new Random(seed));
         kFoldCV(images);
      }

      public static void kFoldCV(ArrayList<Image> images)
     {
        double mean;
        double sd;
        // int knum = (images.size()/FOLDS)*FOLDS;

       // double[] result = new double[knum];

        for(int e = 0; e < EPOCHS; e++)
        {
           for(int f = 0; f < FOLDS; f++)
           {
             //images BEFORE test data chunk
             for(int bt = 0; bt < f*(images.size()/FOLDS); bt++)
             {
                 feedforward(images.get(bt));
                 backpropagation(images.get(bt));
                 updateWeights(LEARNING_RATE);
             }

             //images AFTER test data chunk
             for(int at = f*(images.size()/FOLDS) + (images.size()/FOLDS); at < images.size(); at++)
             {
                 feedforward(images.get(at));
                 backpropagation(images.get(at));
                 updateWeights(LEARNING_RATE);
             }

             double acc = 0;
             //TEST DATA chunk
             for(int t = f*(images.size()/FOLDS), r = 0; t < f*(images.size()/FOLDS) + (images.size()/FOLDS); t++,r++)
             {
                 if(images.get(t).actualGender == 1 && feedforward(images.get(t)) >= .66)
                    acc++;
                 else if(images.get(t).actualGender == 0 && feedforward(images.get(t)) < .66)
                    acc++;
             }

            //  System.out.println("FOLD " + (f+1) + " Accuracy %: " + (acc/(images.size()/FOLDS)));
                 //result[t] = images.get(t).actualGender - feedforward(images.get(t));
                 // System.out.println(result[t]);
           }

         //   mean = getMean(result, knum);
         //   sd = getStandardDeviation(result, mean, knum);
         //   System.out.println(mean + " " + sd);
        }
     }

      public static double getMean(double[] result, int knum)
      {
         double mean = 0;

         for(int i = 0; i < result.length; i++)
            mean += result[i];

         return (mean/knum);
      }

      public static double getStandardDeviation(double[] result, double mean, int knum)
      {
         double sd = 0;

         for(int i = 0; i < result.length; i++)
            sd += Math.pow((result[i]-mean),2);

         sd /= knum;

         return Math.sqrt(sd);
      }

      private static void updateWeights(double learningRate) {
         for(int j = 0; j < weights[0].length; j++) {    // max 3

            double error = errors[1][j];
            for(int k = 0; k < weights[0][j].length; k++) {

               double input = nodeValues[0][k];
               weights[0][j][k] -= learningRate * error * input;
            }
         }

         for(int k = 0; k < weights[1].length; k++) {
            // weight = weight + learning_rate * error * input
            double error = errors[2][k];
            for(int l = 0; l < weights[1][k].length; l++) {
               double input = nodeValues[1][l];
               weights[1][k][l] -= learningRate * error * input;
            }
         }
      }

      public String predict(Image img) {
         double prediction = feedforward(img);
         System.out.println(prediction);
         if(prediction >= 0.50) return "MALE";
         else return "FEMALE";

      }

      public  String[] predictSet(ArrayList<Image> imageSet) {
         String[] result = new String[imageSet.size()];
         for(int i = 0; i < imageSet.size(); i++) {
            Image currImage = imageSet.get(i);
            result[i] = predict(currImage);
         }
         return result;
      }

      private static double feedforward(Image image) {
         // First Layer contains the input (pixels)
         for(int z = 0; z < (DIMENSION); z++) {
            nodeValues[0][z] = image.imagePixel[z];
         }

         // Zero out all of the hidden layer
         for(int o = 0; o < HIDDEN_NODES_EACH_LAYER; o++) {
            nodeValues[1][o] = 0;
         }

         // Between input and hidden layer.
         for(int i = 0; i < weights[0].length; i++) {  // Iterate through all hidden nodes
            for(int j = 0; j < weights[0][i].length; j++) {
               nodeValues[1][i] += weights[0][i][j]*nodeValues[0][j];
            }
            nodeValues[1][i] = sigmoid(nodeValues[1][i]);
         }

         // Between hidden layer and output layer.
         for(int m = 0; m < weights[1].length; m++) {
            for(int n = 0; n < weights[1][m].length; n++) {
               nodeValues[2][m] += weights[1][m][n]*nodeValues[1][n];
            }
            nodeValues[2][m] = sigmoid(nodeValues[2][m]);
         }

         return nodeValues[2][0];
      }

      private static void backpropagation(Image image) {
         // Get the error of the output node.
         double output = nodeValues[2][0];
         double expected = image.actualGender;
         errors[2][0] = (output - expected) * output * (1-output);

         double sumWeight1 = weights[1][0][0] * errors[2][0];
         errors[1][0] = nodeValues[2][0] * (1-nodeValues[2][0]) * sumWeight1;

         double sumWeight2 = weights[1][0][1] * errors[2][0];
         errors[1][1] = nodeValues[2][0] * (1-nodeValues[2][0]) * sumWeight2;

         double sumWeight3 = weights[1][0][2] * errors[2][0];
         errors[1][2] = nodeValues[2][0] * (1-nodeValues[2][0]) * sumWeight3;
      }

      private static double transferDerivative(double output) {
         return output * (1.0 - output);
      }

      private static double sigmoid(double x)
      {
          return (double) 1.0 / (1 + Math.exp(-x));
      }

      public static void printNoWeightsEachLayer() {
         for(int i = 0; i < LAYER-1; i++) {
            System.out.println();
            System.out.println("Set of weights #" + (i+1));
            for(double[] node : weights[i]){
               System.out.println(node.length);
               System.out.println(Arrays.toString(node));
            }
         }
      }

      public void printNodesValue() {
         System.out.println("######################");
         System.out.println("Node values:");
         for(int i = 0 ; i < LAYER; i++) {
            System.out.print(" Layer #" + (i+1) + ": ");
            for(int j = 0; j < nodeValues[i].length; j++) {
               System.out.print(nodeValues[i][j] + ",");
            }
            System.out.println();
         }
      }
   }

   public static void main(String[] args) {
      boolean training = false;
      boolean testing = false;
      if(contains(args, "-train")) training = true;
      if(contains(args, "-test")) testing = true;

      NeuralNetwork nn = new NeuralNetwork();
      if(training) {
         trainingImages  = new ArrayList<Image>();
         // femaleImages = new ArrayList<Image>();
         // maleImages = new ArrayList<Image>();
         String[] trainingFileNames = getTrainingFiles(args);
         for(String fileName : trainingFileNames) {
            // System.out.println(fileName);
            if(fileName.equals("Male") || fileName.equals("male")) readImages("./Male", true, 1);
            else if(fileName.equals("Female") || fileName.equals("female")) readImages("./Female", true, 0);
         }


         // Valid training data = all male and female training data are labeled 1 and 0 respectively.
         System.out.println("Valid training data: " + validTrainingFiles(trainingImages) );
         System.out.println("Please wait, the whole training/testing process will run no more than 3 minutes.");
         nn.train(trainingImages, EPOCHS);
      }

      if(testing) {
         testingImages  = new ArrayList<Image>();
         String testFileName = getTestingFile(args);
         readImages("./" + testFileName, false, 99);
         String[] testResult = nn.predictSet(testingImages);
         System.out.println("\nTesting using the Test files...");
         System.out.println("Test result: ");
         printTestResult(testResult);
      }

      // Write the final weights into a text file to be visualized using MATLAB later.
      for(int l = 1; l <= nn.HIDDEN_NODES_EACH_LAYER; l++) {
         BufferedWriter bw = null;
   		FileWriter fw = null;

   		try {
   			fw = new FileWriter("weight"+ l + ".txt");
   			bw = new BufferedWriter(fw);
            int i = 0;
   			for(int z = 0; z < WIDTH; z++) {
               for(int x = 0; x < LENGTH; x++) {
                  bw.write(nn.weights[0][l-1][i] + " ");
                  i++;
               }
               bw.write("\n");
            }

   			System.out.println("weight" + l + ".txt has been written successfully.");

   		} catch (IOException e) {

   			e.printStackTrace();

   		} finally {

   			try {

   				if (bw != null)
   					bw.close();

   				if (fw != null)
   					fw.close();

   			} catch (IOException ex) {

   				ex.printStackTrace();

   			}

   		}
      }

   }

   public static void printTestResult(String[] testResult) {
      for(int i = 0; i < testResult.length; i++) {
         System.out.println(testResult[i]);
      }
      System.out.println();
   }

   public static void readImages(String path, boolean isTrainingData, int gender) {
      File folder = new File(path);
      File[] listOfFiles = folder.listFiles();

      for (File file : folder.listFiles()) {
         if (file.isFile() && file.getName().endsWith(".txt")) {
            double[] imagePixel = new double[LENGTH*WIDTH];
            int i = 0;
             try {

                 Scanner sc = new Scanner(file);

                 while (sc.hasNext()) {
                     double num = Double.parseDouble(sc.next());
                     imagePixel[i] = num;
                     i++;
                 }
                 sc.close();
             }
             catch (FileNotFoundException e) {
                 e.printStackTrace();
             }

            if(isTrainingData){
               trainingImages.add(new Image(imagePixel,gender));
            } else {
               testingImages.add(new Image(imagePixel));
            }
            // if(!isTrainingData)System.out.println(file.getName());
         }
      }
   }

   // public static void getFemaleImages(){
   //    for(Image image : trainingImages) {
   //       if(image.actualGender == 0) {
   //          femaleImages.add(image);
   //       }
   //    }
   // }
   //
   // public static void getMaleImages(){
   //    for(Image image : trainingImages) {
   //       if(image.actualGender == 1) {
   //          maleImages.add(image);
   //       }
   //    }
   // }

   public static void printImagesArray(){
      for(Image image : trainingImages) {
         System.out.println(image.toString());

      }
   }

   public static boolean validTrainingFiles(ArrayList<Image> ar) {
      for(int i = 0; i < ar.size(); i++) {
         if(ar.get(i).actualGender == 99) {
            return false;
         }
      }
      return true;
   }

   public static boolean contains(String[] ar, String target) {
      for(int i = 0; i < ar.length; i++) {
         if(ar[i].equals(target)) return true;
      }
      return false;
   }

   public static String[] getTrainingFiles(String[] ar) {
      int i = 1;  // Assume first element is -train
      ArrayList<String> result = new ArrayList<String>();
      while(!ar[i].equals("-test") && i < ar.length) {
         result.add(ar[i]);
         i++;
      }
      return result.toArray(new String[0]);
   }

   public static String getTestingFile(String[] ar) {
      int i = 0;
      while(i < ar.length) {
         if(ar[i].equals("-test")) {
            return ar[i+1];
         }
         i++;
      }
      return null;
   }
}
