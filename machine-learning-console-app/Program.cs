using System;
using System.Text;
using System.Diagnostics;

class Program
{
    private static string appFolderPath = PathGetDirectoryNameTimes(4, AppDomain.CurrentDomain.BaseDirectory);

    /// <summary>
    /// Used to apply Path.GetDirectoryName() to a path string a specified amount of times.
    /// </summary>
    /// <param name="times">The times Path.GetDirectoryName() should be applied.</param>
    /// <param name="s">The path string.</param>
    /// <returns>A new path string.</returns>
    private static string PathGetDirectoryNameTimes(int times, string s)
    {
        for (int i = 0; i < times; i++)
            s = Path.GetDirectoryName(s);
        return s;
    }

    private static Program? _instance;

    static void Main(string[] args)
    {
        _instance = new Program(args);
    }

    public Program(string[] args)
    {
        // Program parameters
        int crossvalPredictFolds = 5; // Amount of CrossvalPredict folds, set to 0 to train and test on the full dataset.
        int predictableShuffle = 5; // Predictably shuffles the dataset - see NaiveBayesModel.PredictableShuffle().
        var availableDatasets = new string[] { "iris", "banknote_authentication" }; // Names of available datasets.
        int datasetID = 0;  // ID of the dataset to use from availableDatasets array.

        var dataset = ReadDataset(availableDatasets[datasetID]);
        //dataset.PrintDataset(); // Uncomment line to print dataset to console after parsing.

        var naiveBayesModel = new NaiveBayes(dataset.IdToWord);
        var x = dataset.Inputs;
        var y = dataset.Labels;

        // Applies Predictable Shuffle to dataset.
        if (predictableShuffle > 1)
        {
            x = naiveBayesModel.PredictableShuffle(x, predictableShuffle);
            y = naiveBayesModel.PredictableShuffle(y, predictableShuffle);
        }

        int[] predictions;
        if (crossvalPredictFolds > 1)
            predictions = naiveBayesModel.CrossvalPredict(x, y, 5);
        else
        {
            naiveBayesModel.Fit(x, y);
            //naiveBayesModel.PrintModels(dataset.IdToWord, dataset.AttributeNames);
            //naiveBayesModel.PrintTestCalculateGaussianPDF();
            predictions = naiveBayesModel.Predict(x);
            naiveBayesModel.PrintPredictions(y, predictions);
            naiveBayesModel.PrintAccuracyScore(predictions, y);
            var confusionMatrix = naiveBayesModel.ConfusionMatrix(predictions, y);
            naiveBayesModel.PrintConfusionMatrix(confusionMatrix);
        }
    }

    /// <summary>
    /// Class responsible for training and testing Naive Bayes algorithm machine learning models.
    /// </summary>
    public class NaiveBayes
    {
        /// <summary>
        /// Class respresting a category/label.
        /// </summary>
        private class Category
        {
            private List<float[]> _inputs;
            private float[] _means;
            private float[] _stds;
            public List<float[]> Inputs { get => _inputs; set => _inputs = value; }
            public float[] Means { get => _means; set => _means = value; }
            public float[] Stds { get => _stds; set => _stds = value; }

            public Category()
            {
                _inputs = new List<float[]>();
            }
        }

        private Dictionary<int, Category> _categories;
        private string[] _categoryNames;
        public string[] CategoryNames => _categoryNames;

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="idToWord">IDtoWord dictionary from a dataset, only used to retrieve category names.</param>
        public NaiveBayes(Dictionary<int, string> idToWord)
        {
            _categoryNames = new string[idToWord.Count];
            for (int i = 0; i < idToWord.Count; i++)
                _categoryNames[i] = idToWord[i];
        }

        public int[] CrossvalPredict(float[][] x, int[] y, int folds)
        {
            void PrintTrainingAndTestingArrays(int bucketStartIndex, float[][] xTesting, int[] yTesting, float[][] xTraining, int[] yTraining)
            {
                Console.WriteLine($"\tTesting:");
                for (int j = 0; j < xTesting.Length; j++)
                    PrintFloatArray(bucketStartIndex + j, xTesting[j], yTesting[j]);
                Console.WriteLine($"\tTraining:");
                for (int j = 0; j < xTraining.Length; j++)
                    PrintFloatArray(bucketStartIndex + j, xTraining[j], yTraining[j]);
            }

            // Determine bucket size and start indices.
            var bucketStartIndexes = new int[folds];
            int bucketSize = y.Length / folds;
            bucketStartIndexes[0] = 0;
            for (int i = 1; i < bucketStartIndexes.Length; i++)
                bucketStartIndexes[i] = bucketStartIndexes[i - 1] + bucketSize;

            List<float> accuracyScores = new List<float>();

            // Initiate cross-validation.
            for (int i = 0; i < bucketStartIndexes.Length; i++)
            {
                // Create arrays for training and testing data.
                int bucketStartIndex = bucketStartIndexes[i];
                int bucketEndIndex = i != bucketStartIndexes.Length - 1 ? bucketStartIndexes[i + 1] : y.Length;
                int bucketLength = bucketEndIndex - bucketStartIndex;
                var xTesting = new float[bucketLength][];
                var yTesting = new int[bucketLength];
                Array.Copy(x, bucketStartIndex, xTesting, 0, bucketLength);
                Array.Copy(y, bucketStartIndex, yTesting, 0, bucketLength);
                var xTrainingList = x.ToList<float[]>();
                xTrainingList.RemoveRange(bucketStartIndex, bucketLength);
                var xTraining = xTrainingList.ToArray();
                var yTrainingList = y.ToList<int>();
                yTrainingList.RemoveRange(bucketStartIndex, bucketLength);
                var yTraining = yTrainingList.ToArray();
                // Train and test data.
                Fit(xTraining, yTraining);
                var predictions = Predict(xTesting);
                // Print Fold results.
                Console.WriteLine($"Fold {i} ({bucketStartIndex}, {bucketEndIndex}, {bucketLength}):");
                //PrintTrainingAndTestingArrays(bucketStartIndex, xTesting, yTesting, xTraining, yTraining);
                PrintPredictions(yTesting, predictions, bucketStartIndex);
                PrintAccuracyScore(predictions, yTesting);
                var confusionMatrix = ConfusionMatrix(predictions, yTesting);
                PrintConfusionMatrix(confusionMatrix);
                accuracyScores.Add(AccuracyScore(predictions, yTesting));
            }
            // Print final results after all folds have finished.
            string accuracyScoreStr = "All Accuracy Scores: ";
            foreach (var score in accuracyScores)
                accuracyScoreStr += score.ToString("0.00") + "\t";
            Console.WriteLine(accuracyScoreStr);
            Console.WriteLine($"Total Accuracy Score: {accuracyScores.Sum() / folds}");

            return Predict(x);
        }

        public void Fit(float[][] x, int[] y)
        {
            var categories = new Dictionary<int, Category>();
            for (int i = 0; i < x.Length; i++)
            {
                var inputs = x[i];
                var label = y[i];
                if (!categories.ContainsKey(label))
                    categories.Add(label, new Category());

                categories[label].Inputs.Add(inputs);
            }

            double StandardDeviation(List<float> values)
            {
                double avg = values.Average();
                return Math.Sqrt(values.Average(v => Math.Pow(v - avg, 2)));
            }

            foreach (var category in categories.Values)
            {
                category.Means = new float[category.Inputs[0].Length];
                category.Stds = new float[category.Inputs[0].Length];
                for (int j = 0; j < category.Inputs[0].Length; j++)
                {
                    List<float> values = new List<float>();
                    for (int k = 0; k < category.Inputs.Count; k++)
                        values.Add(category.Inputs[k][j]);

                    category.Means[j] = values.Average();
                    category.Stds[j] = (float)StandardDeviation(values);
                }
            }

            _categories = categories;
        }

        private float CalculateGaussianPDF(float x, float mean, float std)
            => (float)((1 / (Math.Sqrt(2 * Math.PI) * std)) * Math.Pow(Math.E, (-(Math.Pow((x - mean), 2)) / (2 * Math.Pow(std, 2)))));

        public int[] Predict(float[][] x)
        {
            var predictions = new int[x.Length];
            for (int i = 0; i < predictions.Length; i++)
            {
                var input = x[i];
                var categoryPDFs = new Dictionary<int, float>();
                foreach (KeyValuePair<int, Category> keyValPair in _categories)
                {
                    var category = keyValPair.Value;
                    var attributePDFs = new float[input.Length];
                    for (int j = 0; j < input.Length; j++)
                    {
                        float gaussianPDF = CalculateGaussianPDF(input[j], category.Means[j], category.Stds[j]);
                        attributePDFs[j] = gaussianPDF;
                    }
                    float p = 1f;
                    foreach (float pdf in attributePDFs)
                        p *= pdf;
                    categoryPDFs.Add(keyValPair.Key, p);
                }
                float sumOfPs = 0;
                foreach (KeyValuePair<int, float> keyValPair in categoryPDFs)
                    sumOfPs += keyValPair.Value;
                var categoryPsNormalized = new Dictionary<int, float>();
                foreach (KeyValuePair<int, float> keyValPair in categoryPDFs)
                    categoryPsNormalized.Add(keyValPair.Key, keyValPair.Value / sumOfPs);

                var categoryWithHighestProbability = categoryPsNormalized.Aggregate(
                    (x, y) => x.Value > y.Value ? x : y).Key;
                predictions[i] = categoryWithHighestProbability;
            }

            return predictions;
        }

        public float AccuracyScore(int[] predictions, int[] y)
        {
            int correct = 0;
            for (int i = 0; i < predictions.Length; i++)
            {
                if (predictions[i] == y[i])
                    correct++;
            }

            return (float)correct / predictions.Length;
        }

        public int[][] ConfusionMatrix(int[] predictions, int[] y)
        {
            int max = Math.Max(predictions.Max(), y.Max()) + 1;
            var confusionMatrix = new int[max][];
            for (int i = 0; i < confusionMatrix.Length; i++)
            {
                confusionMatrix[i] = new int[max];
                for (int j = 0; j < confusionMatrix[i].Length; j++)
                    confusionMatrix[i][j] = 0;
            }
            for (int i = 0; i < predictions.Length; i++)
            {
                confusionMatrix[y[i]][predictions[i]]++;
            }
            return confusionMatrix;
        }

        public void PrintModels(Dictionary<int, string> idToWord, string[] attributeNames = null)
        {
            foreach (KeyValuePair<int, Category> keyValPair in _categories)
            {
                var label = idToWord.ContainsKey(keyValPair.Key) ? idToWord[keyValPair.Key] : keyValPair.Key.ToString();
                var category = keyValPair.Value;
                Console.WriteLine($"Printing Category '{label}':");
                var attributeNamesStr = "\t";
                var meansStr = "\t";
                var stdsStr = "\t";
                for (int i = 0; i < category.Means.Length; i++)
                {
                    if (attributeNames != null)
                        attributeNamesStr += attributeNames[i] + "\t";
                    meansStr += category.Means[i].ToString("0.00") + "\t";
                    stdsStr += category.Stds[i].ToString("0.00") + "\t";
                }
                if (attributeNames != null)
                    Console.WriteLine(attributeNamesStr);
                Console.WriteLine(meansStr);
                Console.WriteLine(stdsStr);
            }
        }

        public void PrintPredictions(int[] x0, int[] x1, int startIndex = 0)
        {
            Console.WriteLine($"Printing predictions:");
            int correct = 0;
            for (int i = 0; i < x0.Length; i++)
            {
                if (x0[i] == x1[i])
                    correct++;

                Console.WriteLine($"\t{startIndex + i} : {_categoryNames[x0[i]]} / {_categoryNames[x1[i]]}");
            }
            Console.WriteLine($"Correct predictions : {correct} / {x0.Length}");
        }

        public void PrintTestCalculateGaussianPDF()
        {
            Console.WriteLine("Test GaussianPDF Function 0: " + CalculateGaussianPDF(1.6f, 1.45f, 0.14f));
            Console.WriteLine("Test GaussianPDF Function 1: " + CalculateGaussianPDF(0.8f, 0.25f, 0.08f));
            Console.WriteLine("Test GaussianPDF Function 2: " + CalculateGaussianPDF(1.6f, 4.40f, 0.52f));
            Console.WriteLine("Test GaussianPDF Function 3: " + CalculateGaussianPDF(0.8f, 1.40f, 0.24f));
        }
        public void PrintAccuracyScore(int[] predictions, int[] y)
            => Console.WriteLine($"Accuracy: {AccuracyScore(predictions, y)}");

        public void PrintConfusionMatrix(int[][] confusionMatrix, Dictionary<int, string> idToWord = null)
        {
            Console.WriteLine("Confusion Matrix:");
            string firstRow = "\t";
            for (int x = 0; x < confusionMatrix.Length; x++)
                firstRow += x + "\t";
            Console.WriteLine(firstRow);
            Console.WriteLine("-----------------------------");
            for (int y = 0; y < confusionMatrix.Length; y++)
            {
                string s = "\t" + y + "|";
                for (int x = 0; x < confusionMatrix.Length; x++)
                {
                    s += confusionMatrix[y][x] + "\t";
                }
                Console.WriteLine(s);
            }
        }

        public void PrintFloatArray(int id, float[] arr, int label = -1)
        {
            string s = $"\t\t{id}: ";
            foreach (float f in arr)
                s += f.ToString("0.00") + "\t";
            if (label != -1)
                s += label;
            Console.WriteLine(s);
        }

        public T[] PredictableShuffle<T>(T[] arr, int times)
        {
            var lists = new List<T>[times];
            for (int i = 0; i < lists.Length; i++)
                lists[i] = new List<T>();
            for (int i = 0; i < arr.Length; i++)
                lists[i % times].Add(arr[i]);
            var newList = new List<T>();
            foreach (List<T> list in lists)
                newList.AddRange(list);
            return newList.ToArray();
        }
    }

    public class ParsedDataset
    {
        private float[][] _inputs;
        private int[] _labels;
        private Dictionary<string, int> _wordToId;
        private Dictionary<int, string> _idToWord;
        private string[] _attributeNames;
        public float[][] Inputs { get => _inputs; set => _inputs = value; }
        public int[] Labels { get => _labels; set => _labels = value; }
        public Dictionary<string, int> WordToId { get => _wordToId; set => _wordToId = value; }
        public Dictionary<int, string> IdToWord { get => _idToWord; set => _idToWord = value; }
        public string[] AttributeNames { get => _attributeNames; set => _attributeNames = value; }

        public ParsedDataset()
        {
            _wordToId = new Dictionary<string, int>();
            _idToWord = new Dictionary<int, string>();
        }

        public int GetIdForWord(string word)
        {
            if (_wordToId.ContainsKey(word))
                // Word found in Dictionary.
                return _wordToId[word];
            else
            {
                // Add missing word to Dictionary.
                int id = _wordToId.Count;
                _wordToId.Add(word, id);
                _idToWord.Add(id, word);
                return id;
            }
        }
        public void PrintIdToWord()
        {
            Console.WriteLine($"Printing contents of IdToWord:");
            foreach (var keyvalpair in IdToWord)
                Console.WriteLine($"\t {keyvalpair.Key} : {keyvalpair.Value}");
        }

        public void PrintWordToId()
        {
            Console.WriteLine($"Printing contents of WordToId:");
            foreach (var keyvalpair in WordToId)
                Console.WriteLine($"\t {keyvalpair.Key} : {keyvalpair.Value}");
        }

        public void PrintDataset()
        {
            Console.WriteLine($"Printing contents of parsed dataset:");
            for (int i = 0; i < _inputs.Length; i++)
            {
                var s = "";
                for (int j = 0; j < _inputs[i].Length; j++)
                    s += _inputs[i][j] + "\t";

                Console.WriteLine($"\t{s}: {_labels[i]}");
            }
        }
    }

    private ParsedDataset ReadDataset(string nameOfDataset)
    {
        void ReadCSV(string path, Action<string[], string[]> onReadLineAction)
        {
            using (var reader = new StreamReader(appFolderPath + path))
            {
                bool isFirstLine = true;
                string[] firstLineValues = null;
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var values = line.Split(',');

                    if (!isFirstLine)
                        onReadLineAction(firstLineValues, values);
                    else
                    {
                        firstLineValues = values;
                        isFirstLine = false;
                    }
                }
            }
        }

        var parsedDataset = new ParsedDataset();
        var inputs = new List<float[]>();
        var labels = new List<int>();

        ReadCSV($@"\datasets\{nameOfDataset}.csv", (firstLineValues, values) =>
        {
            if (values.Length < 2)
                return;

            if (parsedDataset.AttributeNames == null)
                parsedDataset.AttributeNames = firstLineValues;

            var floats = new float[values.Length - 1];

            for (int i = 0; i < values.Length; i++)
            {
                if (i != values.Length - 1)
                    floats[i] = float.Parse(values[i].Replace('.', ','));
                else
                    labels.Add(parsedDataset.GetIdForWord(values[i]));
            }

            inputs.Add(floats);
        });

        parsedDataset.Inputs = inputs.ToArray();
        parsedDataset.Labels = labels.ToArray();

        return parsedDataset;
    }
}
