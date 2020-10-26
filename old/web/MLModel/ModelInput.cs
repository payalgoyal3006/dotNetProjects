using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.Data;
namespace web.MLModel
{
    public class ModelInput
    {
        [LoadColumn(0)]
        public string feedbacktext;
        [LoadColumn(1), ColumnName("Label")]
        public bool value;
    }
}
