// This file was auto-generated by ML.NET Model Builder. 

using System;
using SemanticML.Model;
using System.Collections.Generic;

namespace SemanticML.ConsoleApp
{
    class Program
    {
        static void Main(string[] args)
        {
           



            // Create single instance of sample data from first line of dataset for model input
            ModelInput sampleData = new ModelInput()
            {
                Feedbacktext = @"this is good",
            };

            // Make a single prediction on the sample data and print results
            var predictionResult = ConsumeModel.Predict(sampleData);

            Console.WriteLine("Using model to make single prediction -- Comparing actual Value with predicted Value from sample data...\n\n");
            Console.WriteLine($"Feedbacktext: {sampleData.Feedbacktext}");
            Console.WriteLine($"\n\nPredicted Value value {predictionResult.Prediction} \nPredicted Value scores: [{String.Join(",", predictionResult.Score)}]\n\n");
            Console.WriteLine("=============== End of process, hit any key to finish ===============");
            Console.ReadKey();
        }
    }
}
