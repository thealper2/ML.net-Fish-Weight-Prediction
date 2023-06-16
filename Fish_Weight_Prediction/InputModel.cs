using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLnetBeginner.Fish_Weight_Prediction
{
    internal class InputModel
    {
        [LoadColumn(0)]
        public string Species { get; set; }

        [LoadColumn(1)]
        public Single Weight { get; set; }

        [LoadColumn(2)]
        public float Length1 { get; set; }

        [LoadColumn(3)]
        public float Length2 { get; set; }

        [LoadColumn(4)]
        public float Length3 { get; set; }

        [LoadColumn(5)]
        public float Height { get; set; }

        [LoadColumn(6)]
        public float Width { get; set; }
    }
}
