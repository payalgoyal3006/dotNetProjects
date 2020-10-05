using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;


namespace demo
{
    class FeedBackTrainingData
    {
        [LoadColumn(0), ColumnName("label")]
        public bool IsGood { get; set; }

        [LoadColumn(1)]
        public string FeedBackText { get; set; }

    }
    class FeedBackPrediction
    {
        [ColumnName("predictedLabel")]
        public bool IsGood { get; set; }
    }
    class Program
    {
        static List<FeedBackTrainingData> trainingdata = new List<FeedBackTrainingData>();

        static List<FeedBackTrainingData> testdata = new List<FeedBackTrainingData>();
        static void LoadTestData()
        {
            testdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "good",
                IsGood = true
            });
            testdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "bad",
                IsGood = false
            });
            testdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "nice",
                IsGood = true
            });
            testdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "yuks",
                IsGood = false
            });
            testdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "awesome",
                IsGood = true
            });
            testdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "horrible",
                IsGood = true
            });
        }

        static void LoadTrainingData()
        {
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "this is good",
                IsGood = true
            });

            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "this is bad",
                IsGood = false
            });

            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "not so good ok ok ",
                IsGood = true
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "this is horrible",
                IsGood = false
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "this is nice",
                IsGood = true
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "nice and good",
                IsGood = true
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "very bad",
                IsGood = false
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "this is yuks",
                IsGood = false
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "awesome",
                IsGood = true
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "sweet and nice",
                IsGood = true
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "bad like hell",
                IsGood = false
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "bad and yuks",
                IsGood = false
            });
        }
        static void Main(string[] args)
        {
            LoadTrainingData();

            var mlContext = new MLContext();

            IDataView dataView = mlContext.Data.LoadFromEnumerable<FeedBackTrainingData>(trainingdata);

            var pipeline = mlContext.Transforms.Text.FeaturizeText("FeedBackText", "Features")
                .Append(mlContext.BinaryClassification.Trainers
                .FastTree(numLeaves: 50, numTrees: 50, minDatapointsInLeaves: 1));

            var model = pipeline.FIT(dataView);

            LoadTestData();
            IDataView dataView1 = mlContext.Data.LoadFromEnumerable<FeedBackTrainingData>(testdata);
            var predictions = model.Transform(dataView1);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
            Console.WriteLine(metrics.Accuracy);
            Console.Read();
            Console.WriteLine("Enter a feedback string");
            string feedbackstring = Console.Read().ToString();
            var predictionFunction = model.MakePredictionFunction<FeedBackTrainingData, FeedBackPrediction>(mlContext);

            var feedbackinput = new FeedBackTrainingData();

            feedbackinput.FeedBackText = feedbackstring;

            var feedbackpredicted = predictionFunction.predict(feedbackinput);
            Console.WriteLine("Predicted :- " + feedbackpredicted.IsGood);
            Console.ReadLine();

        }
    }
}

