using Microsoft.ML;
using Microsoft.ML.Data;
using System;

namespace mlnet1.predict
{
    public class IrisFlowerPrediction
    {
        public static void Execute()
        {
            Console.WriteLine(typeof(IrisFlowerPrediction).Name);

            // STEP 2: Create a ML.NET environment
            MLContext mlContext = new MLContext();

            // If working in Visual Studio, make sure the 'Copy to Output Directory'
            // property of iris-data.txt is set to 'Copy always'
            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<IrisData>(path: "predict/iris-data.txt", hasHeader: false, separatorChar: ',');

            // STEP 3: Transform your data and add a learner pipeline (aka classifier)
            // Assign numeric values to text in the "Label" column, because only
            // numbers can be processed during model training.
            // Add a learning algorithm to the pipeline. e.g.(What type of iris is this?)
            // Convert the Label back into original text (after converting to number in step 3)
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // STEP 4: Train your model based on the data set
            var model = pipeline.Fit(trainingDataView);
            
            // STEP 5: Use your model to make a prediction
            // You can change these numbers to test different predictions

            //5.1,3.5,1.4,0.2,Iris-setosa (not in the training model)
            var prediction = mlContext.Model.CreatePredictionEngine<IrisData, IrisPrediction>(model).Predict(
                new IrisData()
                {
                    SepalLength = 5.1f,
                    SepalWidth = 3.5f,
                    PetalLength = 1.4f,
                    PetalWidth = 0.2f,
                });


            Console.WriteLine($"Predicted flower type is: {prediction.PredictedLabels}");
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
