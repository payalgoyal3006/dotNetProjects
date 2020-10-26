using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.ML;


namespace sentiment_analysis
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


    public class Program
    {
        static List<FeedBackTrainingData> trainingdata = new List<FeedBackTrainingData>();

                static List<FeedBackTrainingData> testdata = new List<FeedBackTrainingData>();
        

     
            static void Main(string[] args)
            {
                

                var mlContext = new MLContext();

                IDataView dataView = mlContext.Data.LoadFromEnumerable<FeedBackTrainingData>(trainingdata);

                var pipeline = mlContext.Transforms.Text.FeaturizeText("FeedBackText", "Features")
                    .Append(mlContext.BinaryClassification.Trainers
                    .FastTree(numLeaves: 50, numTrees: 50, minDatapointsInLeaves: 1));

                var model = pipeline.FIT(dataView);

                
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

            




            CreateHostBuilder(args).Build().Run();
        }

        public static IHostBuilder CreateHostBuilder(string[] args) =>
            Host.CreateDefaultBuilder(args)
                .ConfigureWebHostDefaults(webBuilder =>
                {
                    webBuilder.UseStartup<Startup>();
                });
    }
}
