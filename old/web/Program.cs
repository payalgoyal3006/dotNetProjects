using Microsoft.ML.Data;
using System;
using Microsoft.ML;
using System.Security.Cryptography.X509Certificates;
using web.MLModel;
using System.IO;

namespace sentiment
{
    class Program
    {
        static readonly string _Traindatapath = Path.Combine(Environment.CurrentDirectory, "Data", "Training.csv");
       
        static readonly string _modelpath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        static void Main(string[] args)
        {

            MLContext mlContext = new MLContext();

            IDataView trainData = mlContext.Data.LoadFromTextFile<ModelInput>("TrainingData.csv");

            var dataprocessingpipeline = mlContext.Transforms.Text.FeaturizeText("feedbacktext", "Features");

            var trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features");

            var trainingpipeline = dataprocessingpipeline.Append(trainer);

            var model = trainingpipeline.Fit(trainData);

            IDataView testdata = mlContext.Data.LoadFromTextFile<ModelInput>("testData.csv");

            IDataView predictions = model.Transform(testdata);

            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "value");
            Console.WriteLine($"Model Quality" + metrics.Accuracy);

            var input = new ModelInput
            {
                feedbacktext = "this is so awesome"
            };

            var result = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model).Predict(input);

            Console.WriteLine($"Predicted fare: " + $"{result.prediction}");

            mlContext.Model.Save(model, trainData.Schema, "MLModel.zip");


        }
    }
}
