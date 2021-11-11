# A4 - Machine Learning

## Description

* In this assignment, you shall implement the Naïve Bayes machine learning algorithm and use it on some datasets.
* You can use any programming language you like.
* ~~You can work alone or in a group of two students.~~
* You shall present your application and code at an oral examination.
* You are not required to build a REST web service for this assignment.

## Submission instructions

See the [Deadlines and Submissions](https://coursepress.lnu.se/courses/web-intelligence/study-guide/deadlines-and-submissions) page.

## Requirements

<table>
  <tr>
    <th>Grade</th>
    <th>Requirements</th>
  </tr>
  <tr>
    <td>E</td>
    <td>
      <ul>
        <li>Implement the Naïve Bayes algorithm, using the code structure below (you are allowed to add more classes and methods if needed).</li>
        <li>Train the model on the <em>Iris</em> and <em>Banknote authentication datasets</em> (see <a href="https://coursepress.lnu.se/courses/web-intelligence/assignments/datasets">Datasets</a> page).</li>
        <li>Calculate classification accuracies for both datasets (use all data for both training and testing).</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td>C-D</td>
    <td>
      <ul>
        <li>Implement code for generating confusion matrices, using the code structure below.</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td>A-B</td>
    <td>
      <ul>
        <li>Implement code for n-fold cross-validation, using the code structure below.</li>
        <li>It shall be possible to use 3, 5 or 10 folds (it is okay if your implementation supports other folds).</li>
        <li>Calculate accuracy score for 5-fold cross-validation on both datasets.</li>
      </ul>
    </td>
  </tr>
</table>

><b>Note!</b> The purpose of this assignment is that you shall learn how to implement Naïve Bayes, encoding label strings to integers, calculating accuracy scores, performing cross-validation, and generating a confusion matrix. These functionalities are often available in machine learning libraries such as Weka or Scikit-learn, which you are not allowed to use. You are allowed to use library functions for loading and shuffling the data and all necessary mathematical operations, data structures, etc.

### Code structure requirements

| NaiveBayes class |     |
|------------------|:---:|
| <span style="font-family:monospace"><font color="red">void</font> fit ( <font color="blue">X:float[][]</font>, <font color="blue">y:int[]</font> )</span> | *Trains the model on input examples X and labels y.* |
| <span style="font-family:monospace"><font color="red">int[]</font> predict ( <font color="blue">X:float[][]</font> )</span> | *Classifies examples X and returns a list of predictions.* |

<br />

| Other methods |     |
|---------------|:---:|
| <span style="font-family:monospace"><font color="red">float</font> accuracy_score ( <font color="blue">preds:int[]</font>, <font color="blue">y:int[]</font> )</span> | *Calculates accuracy score for a list of predictions.* |
| <span style="font-family:monospace"><font color="red">int[][]</font> confusion_matrix ( <font color="blue">preds:int[]</font>, <font color="blue">y:int[]</font> )</span> | *Generates a confusion matrix and returns it as an integer matrix.* |
| <span style="font-family:monospace"><font color="red">int[]</font> crossval_predict ( <font color="blue">X:float[][]</font>, <font color="blue">y:int[]</font>, <font color="blue">folds:int</font>  )</span> | *Runs n-fold cross-validation and returns a list of predictions.* |

>Input data (a float matrix with input variables as columns and examples as rows) is usually denoted with <code>X</code>. The categories/labels (a list of integers) is usually denoted as <code>y</code>. Predictions (a list of integers) shall be compared with the actual labels (<code>y</code>) when calculating the accuracy score (percentage correct predictions) and generating the confusion matrix.

## Test cases

You can verify your results with the results in [Web ML Experimenter](http://aiguy.000webhostapp.com/webml/experimenter.html). The *Iris* dataset is built-in in Web ML (click the *Try Iris dataset* button), and the *Banknote authentication* can be uploaded from the CSV file. Note that the cross-validation results can differ slightly due to how the data is split into folds, but the accuracy you get should be almost similar to the accuracy in Web ML.
