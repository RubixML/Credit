# Credit Card Default Predictor

A Rubix ML example project that predicts the probability of a customer defaulting on their credit card bill next month using a Logistic Regression estimator and data transform pipeline. This project demonstrates binary classification, one hot encoding, standardization, and model persistence.

## Installation

Clone the repository locally:
```sh
$ git clone https://github.com/RubixML/Credit
```

Install dependencies:
```sh
$ composer install
```

## Requirements
- [PHP](https://php.net) 7.1.3 or above

### Slides
You can refer to the [slide deck](https://docs.google.com/presentation/d/1ZteG0Rf3siS_o-8x2r2AWw95ntcCggmmEHUfwQiuCnk/edit?usp=sharing) that accompanies this example project if you need extra help or wanted a more in depth look at the math behind Logistic Regression and Gradient Descent.

## Tutorial
The dataset provided to us contains 30,000 labeled samples from customers of a Taiwanese credit card issuer. Our objective is to train an estimator that predicts the probability of a customer defaulting on their credit card bill next month. Since this is a binary classification problem (*will* or *won't* default) we can use Rubix's [Logistic Regression](https://github.com/RubixML/RubixML#logistic-regression) classifier which implements the Probabilistic interface. Logistic Regression is a supervised learner that uses an algorithm called Gradient Descent under the hood. Since Logistic Regression is only compatible with continuous features (*ints* and/or *floats*) we will need [One Hot Encoder](https://github.com/RubixML/RubixML#one-hot-encoder) to convert all the categorical features such as gender, education, and marital status to continuous ones. We'll also demonstrate standardizing using the [Z Scale Standardizer](https://github.com/RubixML/RubixML#z-scale-standardizer) and model persistence using the [Persistent Model](https://github.com/RubixML/RubixML#persistent-model) wrapper.

### Training
Training is the process of feeding data to the learner so that it can build a model of the problem its trying to solve. In Rubix, data is carried in containers called *Datasets*. Let's start by extracting the dataset from the provided `dataset.csv` file and instantiating a *Labeled* dataset object from it.

> **Note**: The full code for this section can be found in `train.php`.

```php
use Rubix\ML\Datasets\Labeled;
use League\Csv\Reader;

$reader = Reader::createFromPath(__DIR__ . '/dataset.csv')
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

$dataset = Labeled::fromIterator($samples, $labels);
```

Here we use the PHP League's [CSV reader](https://csv.thephpleague.com/) to extract the data into two iterators - one for the samples and one for the labels. Next, we instantiate a *labeled* dataset object using the `fromIterator()` factory method. You can perform all sorts of operations on a Dataset once it's instantiated. For a full list check out the [API Reference](https://github.com/RubixML/RubixML#dataset-objects).

We now turn our attention to instantiating and setting the hyper-parameters of the learner. Hyper-parameters are parameters whose value is set during instantiation. Different settings of the hyper-parameters can lead to different solutions. Since learners are composable like Lego® bricks in Rubix, it's easy to rapidly iterate over different models and configurations until you find the best.

```php
use Rubix\ML\Pipeline;
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Classifiers\LogisticRegression;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;

$estimator = new PersistentModel(new Pipeline([
    new NumericStringConverter(),
    new OneHotEncoder(),
    new ZScaleStandardizer(),
], new LogisticRegression(128, new Adam(0.001), 1e-4, 300, 1e-4, new CrossEntropy())),
    new Filesystem(MODEL_FILE)
);
```

Meta-Estimators are estimators that wrap or manipulate other estimators. Pipeline is a meta-Estimator that takes care of applying various transformations to the dataset before it is handed off to the underlying estimator. For our problem, we will need 3 separate transformers. The [Numeric String Converter](https://github.com/RubixML/RubixML#numeric-string-converter) takes care of converting numeric strings (ex. '17', '2.03241') to their integer and floating point counterparts. The only reason why this is necessary is because the CSV reader only recognizes string types. Next we apply a [One Hot](https://en.wikipedia.org/wiki/One-hot) encoding to the categorical features of each sample. Finally, we use the [Z Scale Standardizer](https://github.com/RubixML/RubixML#z-scale-standardizer) to center and scale the samples such that they have 0 mean and unit variance. The last transformation has been shown to help our learning algorithm converge faster.

Next we define the parameters of the Logistic Regression estimator in the following order - batch size, Gradient Descent optimizer, regularization amount, max # of training epochs, minimum change in the parameters to continue training, and the cost function. The default parameters chosen for this project work fairly well and achieve results as good or slightly better than the results in the original paper.

Lastly, we wrap the entire Pipeline in a [Persistent Model](https://github.com/RubixML/RubixML#persistent-model) meta-Estimator so that we can save and load it from storage when we need it in another process.

Since the Logistic Regression estimator implements the Verbose interface, we can hand it any PSR-3 compatible logger and it will spit back helpful logging information. For the purposes of this example we will use the Screen logger that comes with Rubix, but there are many other great loggers that you can use such as [Monolog](https://github.com/Seldaek/monolog) or [Analog](https://github.com/jbroadway/analog).

```php
use Rubix\ML\Other\Loggers\Screen;

$estimator->setLogger(new Screen('credit'));
```

Now we are all set to train the estimator. But first, let's split the dataset into a *training* (80%) and *testing* set (20%) so that we can use the testing set to generate some cross validation reports later. We choose to randomize and stratify the dataset so that each subset contains roughly the same number of positive (*will* default) to negative (*won't* default) samples.

```php
list($training, $testing) = $dataset->randomize()->stratifiedSplit(0.80);

$estimator->train($training);
```

To generate the validation report which consists of a [Confusion Matrix](https://github.com/RubixML/RubixML#confusion-matrix) and [Multiclass Breakdown](https://github.com/RubixML/RubixML#multiclass-breakdown) wrapped in a [report aggregator](https://github.com/RubixML/RubixML#aggregate-report), simply pass it the predictions from the testing set along with their ground truth labels.

```php
$report = new AggregateReport([
    new MulticlassBreakdown(),
    new ConfusionMatrix(),
]);

$predictions = $estimator->predict($testing);

$results = $report->generate($predictions, $testing->labels());
```

Lastly, we direct the estimator to prompt us to save the model to storage.

```php
$estimator->prompt();
```

Since model wrappers allow you to call methods on its children, we can call the `steps()` method on Logistic Regression from the outer Persistent Model wrapper to output the value of the cost function at each epoch from the last training session.

```php
var_dump($estimator->steps());
```

You can later plot the cost function using your favorite plotting software and will get something like this. As you can see, the learner learns quickly at first and then gradually *lands* as it fine tunes the parameters for the best possible accuracy.

![Cross Entropy Loss](https://github.com/RubixML/Credit/blob/master/docs/images/cross-entropy-loss.png)

Now that the training script is set up, we can run the program using the [PHP CLI](http://php.net/manual/en/features.commandline.php) (Command Line Interface) in a terminal window.

To run the training script from the project root:
```sh
$ php train.php
```

That's it, if the results of the training session are good (the researchers in the original paper were able to achieve 82% accuracy using their version of Logistic Regression) then save the model and we'll use it to make predictions on some unknown samples in the next section.

### Predicting
Along with the training data, we provide 5 unknown (*unlabeled*) samples that can be used to demonstrate how to make predictions using the estimator we just trained and saved. First, we'll need to load the data from `unkown.csv` into a dataset object just like before but this time we use an *Unlabeled* dataset.

```php
use Rubix\ML\Datasets\Unlabeled;
use League\Csv\Reader;

$reader = Reader::createFromPath(__DIR__ . '/unknown.csv')
    ->setDelimiter(',')->setEnclosure('"')->setHeaderOffset(0);

$samples = $reader->getRecords([
    'credit_limit', 'gender', 'education', 'marital_status', 'age',
    'timeliness_1', 'timeliness_2', 'timeliness_3', 'timeliness_4',
    'timeliness_5', 'timeliness_6', 'balance_1', 'balance_2', 'balance_3',
    'balance_4', 'balance_5', 'balance_6', 'payment_1', 'payment_2',
    'payment_3', 'payment_4', 'payment_5', 'payment_6', 'avg_balance',
    'avg_payment',
]);

$dataset = Unlabeled::fromIterator($samples);
```

Use the `load()` factory method on Persistent Model to reconstitute the model from storage.

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;

$estimator = PersistentModel::load(new Filesystem(MODEL_FILE));
```

Finally, we output an array of class probabilities corresponding to the unknown samples and save them to a JSON file. We could also predict just the class outcomes if we wanted to, but we want to be able to measure varying degrees of risk (high, medium, low, etc.) so class probabilities make more sense.

```php
$probabilities = $estimator->proba($dataset);

// $predictions = $estimator->predict($dataset);

file_put_contents(PROBS_FILE, json_encode($probabilities, JSON_PRETTY_PRINT));
```

Now take a look at the predictions and observe that out of the 5 samples, one of them should have a higher probability of defaulting than the others.

### Cross Validation
Cross Validation tests the generalization performance of a model. There are a few forms of cross validation to work with in Rubix, but for this example we will just use Monte Carlo simulations. The [Monte Carlo](https://github.com/RubixML/RubixML#monte-carlo) validator works by repeatedly sampling training and testing sets from the master dataset and averaging the validation score of each trained model. We use the [F1 Score](https://github.com/RubixML/RubixML#f1-score) as the metric of validation because it takes into consideration both the precision and recall of the estimator.

```php
use Rubix\ML\CrossValidation\MonteCarlo;
use Rubix\ML\CrossValidation\Metrics\F1Score;

$validator = new MonteCarlo(10, 0.2, true);

$estimator = PersistentModel::load(new Filesystem(MODEL_FILE));

$score = $validator->test($estimator, $dataset, new F1Score());
```

A good F1 score for this dataset using Logistic Regression is in the higher 0.60s. If you're comfortable so far continue on to the next section where we'll explore a more advanced machine learning concept called manifold learning. We'll use it to visualize the credit card dataset.

### Exploring the Dataset
The dataset given to use has 26 dimensions and after one hot encoding it becomes 50+ dimensions. Visualizing this type of high-dimensional data is only possible by reducing the number of dimensions to something we can plot on a chart (1 - 3 dimensions). Such dimensionality reduction is called *Manifold Learning*. Here we will use a popular manifold learning algorithm called [t-SNE](https://github.com/RubixML/RubixML#t-sne) to help us visualize the data.

As always we start by importing the dataset from CSV, but this time we are only going to use 500 random samples to generate the low dimensional representation.

```php
use Rubix\ML\Datasets\Labeled;
use League\Csv\Reader;

$reader = Reader::createFromPath(__DIR__ . '/dataset.csv')
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

$dataset = Labeled::fromIterator($samples, $labels)->randomize()->head(500);
```

We instantiate the estimator using the same transformer pipeline as before and pass in a logger instance so we can monitor the progress of the embedding in real time. Refer to the [t-SNE documentation](https://github.com/RubixML/RubixML#t-sne) in the API reference for an explanation of the hyper-parameters.

```php
use Rubix\ML\Pipeline;
use Rubix\ML\Manifold\TSNE;
use Rubix\ML\Other\Loggers\Screen;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Transformers\NumericStringConverter;

$estimator = new Pipeline([
    new NumericStringConverter(),
    new OneHotEncoder(),
    new ZScaleStandardizer(),
], new TSNE(2, 30, 12., 100.0, 500, 1e-8, 5, new Euclidean()));

$estimator->setLogger(new Screen('credit'));
```

Then we train the estimator and use it to generate the low dimensional embedding. Finally, we save the embedding to a CSV file where it can be imported into your favorite plotting software such as [Tableu](https://www.tableau.com/) or Excel.

```php
$estimator->train(clone $dataset); // Clone dataset since we use it again later to predict

$predictions = $estimator->predict($dataset);

$writer = Writer::createFromPath(OUTPUT_FILE, 'w+');
$writer->insertOne(['x', 'y']);
$writer->insertAll($predictions);
```

> **Note**: Since we are using a transformer pipeline that modifies the dataset, we first clone the dataset to keep an original (untransformed) copy in memory to pass to `predict()`.

Here is an example of what a typical embedding would look like when plotted. As you can see the samples form two distinct blobs that correspond to the group likely to default and the group likely to pay on time. If you wanted to, you could even plot the labels such that each point is colored accordingly to its class label.

![Example t-SNE Embedding](https://github.com/RubixML/Credit/blob/master/docs/images/t-sne-embedding.png)

> **Note**: Due to the stochastic nature of t-SNE, each embedding will look a little different from the last. The important information is contained in the overall *structure* of the data.

### Wrap Up
- [Logistic Regression](https://github.com/RubixML/RubixML#logistic-regression) is a type of classifier that uses a supervised learning algorithm called Gradient Descent.
- Data is passed around in [Dataset objects](https://github.com/RubixML/RubixML#dataset-objects)
- We can use a data transform [Pipeline](https://github.com/RubixML/RubixML#pipeline) to get the data into the correct shape and format for the underlying estimator to understand
- [Cross Validation](https://github.com/RubixML/RubixML#cross-validation) allows us to test the generalization performance of the trained estimator
- We can convert high-dimensional datasets to easily visualizable low-dimensional represenations using a process called [Manifold Learning](https://github.com/RubixML/RubixML#embedders)

## Original Dataset
Contact: I-Cheng Yeh
Emails: (1) icyeh '@' chu.edu.tw (2) 140910 '@' mail.tku.edu.tw  
Institutions: (1) Department of Information Management, Chung Hua University, Taiwan. (2) Department of Civil Engineering, Tamkang University, Taiwan. other contact information: 886-2-26215656 ext. 3181

### References
[1] Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480.
