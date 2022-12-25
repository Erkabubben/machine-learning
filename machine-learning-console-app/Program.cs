using System;
using System.Text;
using System.Diagnostics;

class Program
{
    private static string appFolderPath = PathGetDirectoryNameTimes(4, AppDomain.CurrentDomain.BaseDirectory);

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
        var irisDataset = ReadDataset("iris");
        irisDataset.PrintDataset();

        var irisNaiveBayesModel = new NaiveBayes();
        irisNaiveBayesModel.Fit(irisDataset.Inputs, irisDataset.Labels);
        irisNaiveBayesModel.PrintModels(irisDataset.IdToWord, irisDataset.AttributeNames);
        irisNaiveBayesModel.TestCalculateGaussianPDF();
    }

    public class NaiveBayes
    {
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
        {
            double pdf = (1 / (Math.Sqrt(2 * Math.PI) * std)) * Math.Pow(Math.E, (-(Math.Pow((x - mean), 2)) / (2 * Math.Pow(std, 2))));
            //(1 / (sqrt(2 * PI) * std)) * e ^ (-((x - mean)^2)/ (2 * std ^ 2)))
            return (float)pdf;
        }

        public void TestCalculateGaussianPDF()
        {
            Console.WriteLine("Test GaussianPDF Function 0: " + CalculateGaussianPDF(1.6f, 1.45f, 0.14f));
            Console.WriteLine("Test GaussianPDF Function 1: " + CalculateGaussianPDF(0.8f, 0.25f, 0.08f));
            Console.WriteLine("Test GaussianPDF Function 2: " + CalculateGaussianPDF(1.6f, 4.40f, 0.52f));
            Console.WriteLine("Test GaussianPDF Function 3: " + CalculateGaussianPDF(0.8f, 1.40f, 0.24f));
        }

        public int[] Predict0(float[][] x)
        {
            var predictions = new int[x.Length];
            for (int i = 0; i < predictions.Length; i++)
            {
                var input = x[i];
                //var categoryPs = new Dictionary<int, float>();
                for (int j = 0; j < input.Length; j++)
                {
                    var attributePDFs = new Dictionary<int, float>();
                    float pdfs = 0f;
                    foreach (KeyValuePair<int, Category> keyValPair in _categories)
                    {
                        var category = keyValPair.Value;
                        float gaussianPDF = CalculateGaussianPDF(input[j], category.Means[j], category.Stds[j]);
                        attributePDFs.Add(j, gaussianPDF);
                    }
                    float p = 0;
                    foreach (KeyValuePair<int, float> keyValPair in attributePDFs)
                        p *= keyValPair.Value;
                }
            }

            return null;
        }

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
                    float p = 0;
                    foreach (float pdf in attributePDFs)
                        p *= pdf;
                    categoryPDFs.Add(keyValPair.Key, p);
                }
                float sumOfPs = 0;
                var categoryPsNormalized = new Dictionary<int, float>();
                foreach (KeyValuePair<int, float> keyValPair in categoryPDFs)
                    sumOfPs += keyValPair.Value;
                foreach (KeyValuePair<int, float> keyValPair in categoryPDFs)
                    categoryPsNormalized.Add(keyValPair.Key, keyValPair.Value / sumOfPs);

                predictions[i] = categoryPsNormalized.Aggregate((x, y) => x.Value > y.Value ? x : y).Key;
            }

            return predictions;
        }

        public float AccuracyScore(int[] predictions)
        {

            return 0f;
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
