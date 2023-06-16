using Microsoft.Identity.Client;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLnetBeginner.Fish_Weight_Prediction
{
    internal class Demo
    {
        public static void Execute()
        {
            var context = new MLContext();

            var path = "C:\\Users\\akrc2\\Downloads\\Fish.csv";

            IDataView data = context.Data.LoadFromTextFile<InputModel>(path: path, separatorChar: ',', hasHeader: true);

            var dataSplit = context.Data.TrainTestSplit(data, testFraction: 0.1);

            var preprocessPipeline = context.Transforms
                .SelectColumns(nameof(InputModel.Species), nameof(InputModel.Weight), nameof(InputModel.Length1), nameof(InputModel.Length2), nameof(InputModel.Length3), nameof(InputModel.Height), nameof(InputModel.Width))
                .Append(context.Transforms.CopyColumns("Label", nameof(InputModel.Weight)))
                .Append(context.Transforms.Categorical.OneHotEncoding("Encoded_Species", nameof(InputModel.Species)))
                .Append(context.Transforms.Concatenate("Features", "Encoded_Species", nameof(InputModel.Length1), nameof(InputModel.Length2), nameof(InputModel.Length3), nameof(InputModel.Height), nameof(InputModel.Width)))
                .Append(context.Transforms.NormalizeMinMax("Features"));

            var trainPipeline = preprocessPipeline
                .Append(context.Regression.Trainers.LbfgsPoissonRegression());

            var model = trainPipeline.Fit(dataSplit.TrainSet);

            var testData = model.Transform(dataSplit.TestSet);
            var predictions = context.Data.CreateEnumerable<InputModel>(testData, reuseRowObject: false);

            Console.WriteLine("Fish Weight Prediction");
            Console.WriteLine("----------------------");
            Console.WriteLine($"SPECIES\tLENGTH1\tLENGTH2\tLENGTH3\tHEIGHT\tWIDTH\tWEIGHT");
            foreach (var prediction in predictions)
            {
                Console.WriteLine($"{prediction.Species}\t{prediction.Length1}\t{prediction.Length2}\t{prediction.Length3}\t{prediction.Height}\t{prediction.Width}\t{prediction.Weight}");
            }

            var metrics = context.Regression.Evaluate(testData);
            Console.WriteLine("Model Performance");
            Console.WriteLine("-----------------");
            Console.WriteLine($"R2 Score: {metrics.RSquared}");
            Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError}");

            var predictionEngine = context.Model.CreatePredictionEngine<InputModel, ResultModel>(model);
            var input = new InputModel { Species = "Bream", Length1 = 23.2F, Length2 = 25.4F, Length3 = 30.0F, Height = 14.17F, Width = 5.27F };
            var result = predictionEngine.Predict(input);
            Console.WriteLine($"Prediction: {result.Prediction}");

            var save_path = "C:\\Users\\akrc2\\Downloads\\FishWeightPrediction.zip";
            context.Model.Save(model, data.Schema, save_path);
        }
    }
}
