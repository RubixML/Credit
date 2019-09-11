# Credit Card Default Predictor
An example Rubix ML project that is able to predict the probability of a customer defaulting on their credit card bill the next month using a [Logistic Regression](https://docs.rubixml.com/en/latest/classifiers/logistic-regression.html) estimator and a 30,000 sample dataset. We'll also vizualize the dataset using a manifold learning technique called [t-SNE](https://docs.rubixml.com/en/latest/embedders/t-sne.html).

- **Difficulty:** Medium
- **Training time:** Minutes
- **Memory needed:** < 1G

## Installation

Clone the repository locally using [Git](https://git-scm.com/):
```sh
$ git clone https://github.com/RubixML/Credit
```

Install dependencies using [Composer](https://getcomposer.org/):
```sh
$ composer install
```

## Requirements
- [PHP](https://php.net) 7.1.3 or above

## Tutorial

### Introduction
The dataset provided to us contains 30,000 labeled samples from customers of a Taiwanese credit card issuer. Our objective is to train an estimator that predicts the probability of a customer defaulting on their credit card bill the next month. Since this is a *binary* classification problem (*will* default or *won't* default) we can use a [Logistic Regression](https://docs.rubixml.com/en/latest/classifiers/logistic-regression.html) classifier which implements the Probabilistic interface. Logistic Regression is a supervised learner that trains using an algorithm called *Gradient Descent* under the hood.

> **Note:** The source code for this example can be found in the [train.php](https://github.com/RubixML/Credit/blob/master/train.php) file in project root.

### Extracting the Data
In Rubix ML, data are passed in specialized containers called [Dataset objects](https://docs.rubixml.com/en/latest/datasets/api.html). We'll start out by extracting the data provided in the `dataset.csv` file and then instantiating a [Labeled](https://docs.rubixml.com/en/latest/datasets/labeled.html) dataset object from it.

```php
use League\Csv\Reader;

$reader = Reader::createFromPath('dataset.csv')
    ->setDelimiter(',')->setEnclosure('"')->setHeaderOffset(0);

$samples = $reader->getRecords([
    'credit_limit', 'gender', 'education', 'marital_status', 'age',
    'timeliness_1', 'timeliness_2', 'timeliness_3', 'timeliness_4',
    'timeliness_5', 'timeliness_6', 'balance_1', 'balance_2', 'balance_3',
    'balance_4', 'balance_5', 'balance_6', 'payment_1', 'payment_2',
    'payment_3', 'payment_4', 'payment_5', 'payment_6', 'avg_balance',
    'avg_payment',
]);

$labels = $reader->fetchColumn('default');
```

First, we'll import the PHP League's [CSV reader](https://csv.thephpleague.com/) into our project to help extract the data from the CSV file. Both the `getRecords()` and `fetchColumn()` methods return iterators which we'll use to load the samples and labels respectively. Afterward, we can instantiate the labeled dataset object using the `fromIterator()` static factory method.

```php
use Rubix\ML\Datasets\Labeled;

$dataset = Labeled::fromIterator($samples, $labels);
```

### Dataset Preparation
Since the CSV Reader imports everything as a string type we'll need to convert the numeric types to their integer and floating point number representations before proceeding. Fortunately, the [Numeric String Converter](https://docs.rubixml.com/en/latest/transformers/numeric-string-converter.html) accomplishes this for us by applying a transformation to the dataset all in one go.

After applying Numeric String Converter, the categorical features such as gender, education, and marital status - as well as the continuous features such as age and credit limit - are represented in their appropriate format. However, the Logistic Regression estimator is not compatible with categorical features directly so we'll need to [One Hot Encode](https://docs.rubixml.com/en/latest/transformers/one-hot-encoder.html) them to convert them into continuous ones.

In addition, it is a good practice to center and scale the data as it helps speed up the convergence of the Gradient Descent algorithm used to train the learner. To do so, we'll transform the dataset with the [Z Scale Standardizer](https://docs.rubixml.com/en/latest/transformers/z-scale-standardizer.html) prior to feeding it to Logistic Regression.

To apply the transformations, call the `apply()` method on the dataset object with each transformer in the proper order.

> **Note:** `apply()` returns self for convenient method chaining.

```php
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Transformers\ZScaleStandardizer;

$dataset->apply(new NumericStringConverter())
    ->apply(new OneHotEncoder())
    ->apply(new ZScaleStandardizer());
```

We'll need to set some of the data aside so that it can be used for testing. The reason we set the data aside rather than training the learner on all the data is because we want to be able to test the learner on samples it has never seen before. The `stratifiedSplit()` method on the Dataset object fairly splits the data into two subsets by a user-specified ratio. For this example, we'll use 80% of the data for training and hold out 20% for testing that we'll do later.

```php
[$training, $testing] = $dataset->stratifiedSplit(0.8);
```

### Instantiating the Learner
You'll notice that [Logistic Regression](https://docs.rubixml.com/en/latest/classifiers/logistic-regression.html) has a few parameters in its constructor. Those are are the *hyper-parameters* of the learner and they control the behavior of the algorithm during training and inference. For this example, we'll specify the first two hyper-parameters, the *batch size* and the Gradient Descent *optimizer* with *learning rate*.

As previously mentioned, Logistic Regression trains using Gradient Descent. Specifically, it uses Mini-batch Gradient Descent which is a form of GD that feeds small batches of the randomized dataset into the learner which are then used to estimate the gradient of the loss function at each epoch. The size of the batch is determined by the *batch size* parameter. A small batch size typically trains faster but produces a rougher gradient. For our example, we'll choose 200 but feel free to play with this setting.

The next hyper-parameter is the Optimizer which controls the update step of the Gredient Descent algorithm. The [Step Decay](https://docs.rubixml.com/en/latest/neural-network/optimizers/step-decay.html) optimizer gradually decreases the learning rate by a given factor every n steps from its global setting. This allows training to be fast at first and then slow as it get closer to the minima of the gradient. We'll keep the default settings of 1e-3 decay every 100 steps with a starting rate of 0.01.

To instantiate the learner, pass these parameters to the constructor.

```php
use Rubix\ML\Classifiers\LogisticRegression;
use Rubix\ML\NeuralNet\Optimizers\StepDecay;

$estimator = new LogisticRegression(200, new StepDecay(0.01, 100));
```

### Setting a Logger
Since Logistic Regression implements the [Verbose](https://docs.rubixml.com/en/latest/verbose.html) interface, we can hand it a [PSR-3](https://www.php-fig.org/psr/psr-3/) compatible logger and it will log helpful information for us. For the purposes of this example we will use the Screen logger that comes built-in with Rubix ML, but there are many great loggers to choose from such as [Monolog](https://github.com/Seldaek/monolog) and [Analog](https://github.com/jbroadway/analog) to name a couple.

```php
use Rubix\ML\Other\Loggers\Screen;

$estimator->setLogger(new Screen('credit'));
```

### Training
Now we are ready to train the learner. Simply pass the *training* set that we created earlier to the `train()` method on the learner instance.

```php
$estimator->train($dataset);
```

### Training Loss
The `steps()` method on Logistic Regression outputs the value of the cost function at each epoch from the last training session. You can plot those values by dumping the them to a CSV file and importing them into your favorite plotting software such as [Plotly](https://plot.ly/) or [Tableu](https://public.tableau.com/en-us/s/).

```php
$losses = $estimator->steps();
```

 The loss should be decreasing at each epoch and changes in the loss value should get smaller the closer the learner is to converging on the minimum of the cost function. As you can see, the Logistic Regression learns quickly at first and then gradually *lands* as it fine tunes the weights of the model for the best setting.

![Cross Entropy Loss](https://raw.githubusercontent.com/RubixML/Credit/master/docs/images/training-loss.svg?sanitize=true)

### Cross Validation
Once the learner has been trained, the next step is to determine if the final model can generalize well. For this process we'll need the testing data that we set aside earlier. We'll generate two reports that compare the predictions outputted by the estimator with the ground truth labels.

The [Multiclass Breakdown](https://docs.rubixml.com/en/latest/cross-validation/reports/multiclass-breakdown.html) report gives us detailed statistics (Accuracy, F1 Score, MCC) about the model's performance at the class level. A [Confusion Matrix](https://docs.rubixml.com/en/latest/cross-validation/reports/confusion-matrix.html) is a table that compares the number of predictions for a class with the actual ground truth. We can wrap both of these reports in an [Aggregate Report](https://docs.rubixml.com/en/latest/cross-validation/reports/aggregate-report.html) to generate them both at the same time.

```php
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;

$report = new AggregateReport([
    new MulticlassBreakdown(),
    new ConfusionMatrix(),
]);
```

To generate the predictions for the report, call the `predict()` method on the estimator with the testing set.

```php
$predictions = $estimator->predict($testing);
```

Then generate the report with the predictions and the labels.

```php
$results = $report->generate($predictions, $testing->labels());
```

The output should look something like this.

```json
{
    "overall": {
        "accuracy": 0.8275287452091318,
        "precision": 0.7854524313333155,
        "recall": 0.6558631809497781,
        "specificity": 0.6558631809497781,
        "negative_predictive_value": 0.7854524313333155,
        "false_discovery_rate": 0.2145475686666844,
        "miss_rate": 0.34413681905022186,
        "fall_out": 0.34413681905022186,
        "false_omission_rate": 0.2145475686666844,
        "f1_score": 0.6843061817340108,
        "mcc": 0.42186027998596265,
        "informedness": 0.31172636189955627,
        "markedness": 0.5709048626666311,
        "true_positives": 4966,
        "true_negatives": 4966,
        "false_positives": 1035,
        "false_negatives": 1035,
        "cardinality": 6001
    }
}
```

### Vizualizing the Dataset
The dataset given to us has 25 features and after one hot encoding it becomes 93. Visualizing this type of high-dimensional data is only possible by reducing the number of dimensions to something that makes sense to plot on a chart (1 - 3 dimensions). Such dimensionality reduction is called *Manifold Learning*. Here we will use a popular manifold learning algorithm called [t-SNE](https://docs.rubixml.com/en/latest/embedders/t-sne.html) to help us visualize the data by embedding it into just two dimensions.

### Extracting the Data
As usual, we start by importing the dataset from its CSV file.

> **Note:** The source code for this example can be found in the [explore.php](https://github.com/RubixML/Housing/blob/master/explore.php) file in project root.

```php
use League\Csv\Reader;

$reader = Reader::createFromPath('/dataset.csv')
    ->setDelimiter(',')->setEnclosure('"')->setHeaderOffset(0);

$samples = $reader->getRecords([
    'credit_limit', 'gender', 'education', 'marital_status', 'age',
    'timeliness_1', 'timeliness_2', 'timeliness_3', 'timeliness_4',
    'timeliness_5', 'timeliness_6', 'balance_1', 'balance_2', 'balance_3',
    'balance_4', 'balance_5', 'balance_6', 'payment_1', 'payment_2',
    'payment_3', 'payment_4', 'payment_5', 'payment_6', 'avg_balance',
    'avg_payment',
]);

$labels = $reader->fetchColumn('default');
```

This time, we are only going to use 1000 samples from the dataset. The `head()` method on the dataset object will return the first *n* samples and labels from the dataset in a new dataset object. We'll randomize the dataset beforehand for good measure.

```php
use Rubix\ML\Datasets\Labeled;

$dataset = Labeled::fromIterator($samples, $labels)->randomize()->head(1000);
```

### Dataset Preparation
Since T-SNE is only compatible with continuous features, we can perform the necessary transformations on the dataset by passing the transformers to the `apply()` method on the dataset object like before.

```php
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Transformers\ZScaleStandardizer;

$dataset->apply(new NumericStringConverter())
    ->apply(new OneHotEncoder())
    ->apply(new ZScaleStandardizer());
```

> **Note:** Centering and standardizing the data with [Z Scale Standardizer](https://docs.rubixml.com/en/latest/transformers/z-scale-standardizer.html) or another standardizer is not always necessary, however, it just so happens that both Logistic Regression and t-SNE benefit when the data are centered and standardized.

### Instantiating the Embedder
Now we'll instantiate a [t-SNE](https://docs.rubixml.com/en/latest/embedders/t-sne.html) embedder. T-SNE stands for t-Distributed Stochastic Neighbor Embedding and is a powerful dimensionality reduction technqiue suited for vizualization of high-dimensional datasets. The first hyper-parameter is the number of dimensions of the target embedding. Since we want to be able to plot the embedding as a 2-d scatterplot we'll set this parameter to 2. The next hyper-parameter is the learning rate which we'll set to 20.0. The last hyper-parameter we'll set is called *perplexity* and can the thought of as the number of nearest neighbors to consider when computing the variance of the distribution of a sample. The value 20 works pretty well for this problem. Refer to the documentation for a full description of the hyper-parameters.

```php
use Rubix\ML\Manifold\TSNE;

$embedder = new TSNE(2, 20.0, 20);
```

### Embedding
Lastly, pass the dataset to the `embed()` method on the [Embedder](https://docs.rubixml.com/en/latest/embedders/api.html) instance to return an array of samples in two dimensions.

```php
$embedding = $embedder->embed($dataset);
```

Here is an example of what a typical embedding looks like when plotted in 2 dimensions. As you can see the samples form two distinct blobs that correspond to the group likely to *default* and the group likely to pay *on time*. If you wanted to, you could even plot the labels such that each point is colored accordingly to its class label.

![t-SNE Embedding](https://raw.githubusercontent.com/RubixML/Credit/master/docs/images/embedding.svg?sanitize=true)

> **Note**: Due to the stochastic nature of the t-SNE algorithm, each embedding will look a little different from the last. The important information is contained in the overall *structure* of the data.

### Wrap Up
- Data is passed in Rubix ML using specialized containers called [Dataset](https://docs.rubixml.com/en/latest/datasets/api.html) objects.
- We can use [Transformers](https://docs.rubixml.com/en/latest/transformers/api.html) to get the data into the correct shape and format for the learner to understand.
- [Logistic Regression](https://docs.rubixml.com/en/latest/classifiers/logistic-regression.html) is a probabilistic classifier that uses a supervised learning algorithm called Gradient Descent under the hood.
- Cross Validation allows us to test the generalization performance of the trained estimator.
- We can embed high-dimensional datasets into easily visualizable low-dimensional represenations using a process called [Manifold Learning](https://docs.rubixml.com/en/latest/embedders/api.html).

### Next Steps
The Logistic Regression estimator we just trained is able to achieve the same results as in the original paper, however, there are other models to choose from that may perform better. Consider the same problem using an ensemble method such as [AdaBoost](https://docs.rubixml.com/en/latest/classifiers/adaboost.html) or [Random Forest](https://docs.rubixml.com/en/latest/classifiers/random-forest.html) for your next step.

## Slide Deck
You can refer to the [slide deck](https://docs.google.com/presentation/d/1ZteG0Rf3siS_o-8x2r2AWw95ntcCggmmEHUfwQiuCnk/edit?usp=sharing) that accompanies this example project if you need extra help or a more in depth look at the math behind Logistic Regression, Gradient Descent, and the Cross Entropy cost function.

## Original Dataset
Contact: I-Cheng Yeh
Emails: (1) icyeh '@' chu.edu.tw (2) 140910 '@' mail.tku.edu.tw  
Institutions: (1) Department of Information Management, Chung Hua University, Taiwan. (2) Department of Civil Engineering, Tamkang University, Taiwan. other contact information: 886-2-26215656 ext. 3181

### References
[1] Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480.
