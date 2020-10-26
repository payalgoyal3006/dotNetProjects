using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace stock_prediction
{
    class Program
    {
        public class ModelInput
        {
            [LoadColumn(0)]
            public DateTime Date;
            [LoadColumn(1)]
            public int quantity_received;
            [LoadColumn(2)]
            public string quantity_received_Delhi;
            [LoadColumn(3)]
            public string quantity_received_Mumbai;
        }
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            IDataView trainingData = mlContext.Data.LoadFromTextFile<ModelInput>("stocktrain1.csv",separatorChar:',');

            var pipeline = mlContext.Transforms.;


        }
    }
}
