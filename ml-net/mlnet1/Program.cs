using mlnet1.predict;
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
                opt = Console.ReadLine();

                switch (opt)
                {
                    case "1": IrisFlowerPrediction.Execute(); break;
                }
            } while (opt != "");
        }
    }
}
