using System;

namespace AnomalyDetection
{

    class Program
    {
        static void Main(string[] args)
        {
            // Start project that demonstrates how to perform detecting anomalies using MultiSequenceLearning.
            HTMAnomalyTesting tester = new HTMAnomalyTesting();
            tester.RunDetecting();

        }

    }
}