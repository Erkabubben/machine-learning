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
        ReadDatasets("iris");
    }

    public class ParsedDataset
    {
        private float[,] _inputs;
        private int[] _labels;
        private Dictionary<string, int> _wordToId;
        private Dictionary<int, string> _idToWord;
        public Dictionary<string, int> WordToId { get => _wordToId; set => _wordToId = value; }
        public Dictionary<int, string> IdToWord { get => _idToWord; set => _idToWord = value; }

        public ParsedDataset()
        {
            _wordToId = new Dictionary<string, int>();
            _idToWord = new Dictionary<int, string>();
        }

        public void PrintIdToWord()
        {
            Console.WriteLine($"Printing contents of IdToWord:");
            foreach (var keyvalpair in IdToWord)
                Console.WriteLine($"\t {keyvalpair.Key} : {keyvalpair.Value}");
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

        public void PrintWordToId()
        {
            Console.WriteLine($"Printing contents of WordToId:");
            foreach (var keyvalpair in WordToId)
                Console.WriteLine($"\t {keyvalpair.Key} : {keyvalpair.Value}");
        }
    }

    void ReadDatasets(string nameOfDataset)
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

        var inputs = new List<float[]>();
        var labels = new List<int>();

        ReadCSV($@"\datasets\{nameOfDataset}.csv", (firstLineValues, values) =>
        {
            if (values.Length < 2)
                return;

            var floats = new float[values.Length - 1];

            for (int i = 0; i < values.Length; i++)
            {
                string? value = values[i];
                if (i != values.Length - 1)
                    floats[i] = float.Parse(value);
                else
                    labels.Add(0);
            }

            inputs.Add(floats);
        });
    }
}
