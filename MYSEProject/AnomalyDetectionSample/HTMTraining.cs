using NeoCortexApi;
using System.Diagnostics;
using System;
using System.Collections.Generic;

namespace AnomalyDetection
{
    public class HTMTraining
    {
        /// <summary>
        /// Executes the HTM model training experiment using CSV files from specified folders and returns the trained predictor.
        /// </summary>
        /// <param name="trainingFolderPath">The path to the folder containing the CSV files used for training.</param>
        /// <param name="predictionFolderPath">The path to the folder containing the CSV files used for prediction.</param>
        /// <param name="trainedPredictor">The trained model that will be used for prediction.</param>
        public void RunHTMTraining(string trainingFolderPath, string predictionFolderPath, out Predictor trainedPredictor)
        {
            Console.WriteLine("------------------------------");
            Console.WriteLine();
            Console.WriteLine("Begins the anomaly detection sample project!!");
            Console.WriteLine();
            Console.WriteLine("------------------------------");
            Console.WriteLine();
            Console.WriteLine("HTM Training initiated...................");

            // Using Stopwatch to measure the total training time
            Stopwatch sw = Stopwatch.StartNew();

            // Read numerical sequences from CSV files in the specified training folder
            CSVReader_Folder trainDataReader = new CSVReader_Folder(trainingFolderPath);
            var trainingSequences = trainDataReader.ReadFolder();

            // Read numerical sequences from CSV files in the specified prediction folder
            CSVReader_Folder PredictDataReader = new CSVReader_Folder(predictionFolderPath);
            var predictionSequences = PredictDataReader.ReadFolder();

            // Combine sequences from both training and prediction folders
            List<List<double>> combinedSequences = new List<List<double>>(trainingSequences);
            combinedSequences.AddRange(predictionSequences);

            // Convert sequences to HTM input format
            CSVToHTM sequenceConverter = new CSVToHTM();
            var htmInput = sequenceConverter.ConvertToHTMInput(combinedSequences);

            // Start multi-sequence learning experiment to generate predictor model
            MultiSequenceLearning learningAlgorithm = new MultiSequenceLearning();
            trainedPredictor = learningAlgorithm.Run(htmInput);

            // HTM model training completed

            sw.Stop();

            Console.WriteLine();
            Console.WriteLine("------------------------------");
            Console.WriteLine();
            Console.WriteLine("HTM Training completed! Total training time: " + sw.Elapsed.TotalSeconds + " seconds.");
            Console.WriteLine();
            Console.WriteLine("------------------------------");
        }
    }
}
