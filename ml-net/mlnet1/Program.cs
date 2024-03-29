﻿using mlnet1.predict;
using System;
using System.Collections.Generic;
using System.Text;

namespace mlnet1
{
    public class Program
    {
        static void Main(string[] args)
        {
            string opt;
            do
            {
                Console.WriteLine("Execute..");
                Console.WriteLine($"1 = {nameof(IrisFlowerPrediction.TrainModel)}");
                Console.WriteLine($"2 = {nameof(IrisFlowerPrediction.LoadModelAndPredict)}");
                opt = Console.ReadLine();

                switch (opt)
                {
                    case "1": IrisFlowerPrediction.TrainModel(); break;
                    case "2": IrisFlowerPrediction.LoadModelAndPredict(); break;
                }
            } while (opt != "");
        }
    }
}
