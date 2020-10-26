using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.Data;
using Microsoft.ML;

namespace web.MLModel
{
    public class ModelOutput
    {
        [ColumnName("Score")]
        public bool prediction;

        MLContext mlContext = new MLContext();
        string modelPath = AppDomain.CurrentDomain.BaseDirectory + "MLModel.zip";
        var mlModel = mlContext.Model.Load(_modelPath, out var modelInputSchema);
        var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);
        ModelOutput result = predEngine.Predict(input);
    }
}
