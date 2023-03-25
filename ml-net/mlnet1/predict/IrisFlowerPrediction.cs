using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Reflection;

namespace mlnet1.predict
{
    public class IrisFlowerPrediction
    {
        public static void TrainModel()
        {
            // STEP 2: Create a ML.NET environment
            var context = new MLContext();

            // If working in Visual Studio, make sure the 'Copy to Output Directory'
            // property of iris-data.txt is set to 'Copy always'
            var data = context.Data.LoadFromTextFile<IrisData>(path: "predict/iris-data.txt", separatorChar: ',');
            
            // Splitting source data to a testing data according to test fraction
            var testDataSplit = context.Data.TrainTestSplit(data, testFraction: 0.1);

            // STEP 3: Transform your data and add a learner pipeline (aka classifier)
            // Assign numeric values to text in the "Label" column, because only
            // numbers can be processed during model training.
            // Add a learning algorithm to the pipeline. e.g.(What type of iris is this?)
            // Convert the Label back into original text (after converting to number in step 3)
            var pipeline = context.Transforms.Conversion.MapValueToKey("Label")
                .Append(context.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
                .AppendCacheCheckpoint(context)
                .Append(context.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features"))
                .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // STEP 4: Train your model based on the data set
            Console.WriteLine("Training model...");
            var model = pipeline.Fit(testDataSplit.TrainSet);

            Console.WriteLine("Checking model predictions with test set...");
            var testSetPredictions = model.Transform(testDataSplit.TestSet);
            var predictionMetrics = context.MulticlassClassification.Evaluate(testSetPredictions);
            Console.WriteLine($"MacroAccuracy:{predictionMetrics.MacroAccuracy}");

            // STEP 5: Use your model to make a prediction
            // Creating prediction engine
            Console.WriteLine("Creating prediction engine...");
            using var predictionEngine = context.Model.CreatePredictionEngine<IrisData, IrisPrediction>(model);

            //5.1,3.5,1.4,0.2,Iris-setosa (not in the training model)
            Console.WriteLine("Testing prediction...");
            var prediction = predictionEngine.Predict(new IrisData()
            {
                SepalLength = 5.1f,
                SepalWidth = 3.5f,
                PetalLength = 1.4f,
                PetalWidth = 0.2f,
            });

            Console.WriteLine($"Predicted flower type is: {prediction.PredictedLabels}");
            Console.WriteLine();

            Console.WriteLine("Generating model file...");
            context.Model.Save(model, data.Schema, "model.zip");
            Console.WriteLine("Model saved on the filesystem");

            // erasing model cache
            _model = null;
        }

        private static ITransformer _model;
        public static void LoadModelAndPredict()
        {
            Console.WriteLine("Loadingmodel file...");
            MLContext mlContext = new();

            _model ??= mlContext.Model.Load("model.zip", out DataViewSchema inputSchema);

            // Creating prediction engine
            Console.WriteLine("Creating prediction engine...");
            using var predictionEngine = mlContext.Model.CreatePredictionEngine<IrisData, IrisPrediction>(_model);

            //5.1,3.5,1.4,0.2,Iris-setosa (not in the training model)
            Console.WriteLine("Testing prediction...");
            var prediction = predictionEngine.Predict(new IrisData()
            {
                SepalLength = 5.1f,
                SepalWidth = 3.5f,
                PetalLength = 1.4f,
                PetalWidth = 0.2f,
            });

            Console.WriteLine($"Predicted flower type is: {prediction.PredictedLabels}");
            Console.WriteLine();
        }

        // STEP 1: Define your data structures
        // IrisData is used to provide training data, and as
        // input for prediction operations
        // - First 4 properties are inputs/features used to predict the label
        // - Label is what you are predicting, and is only set when training
        class IrisData
        {
            [LoadColumn(0)]
            public float SepalLength;

            [LoadColumn(1)]
            public float SepalWidth;

            [LoadColumn(2)]
            public float PetalLength;

            [LoadColumn(3)]
            public float PetalWidth;

            [LoadColumn(4)]
            public string Label;
        }

        // IrisPrediction is the result returned from prediction operations
        class IrisPrediction
        {
            [ColumnName("PredictedLabel")]
            public string PredictedLabels;
        }
    }
}
