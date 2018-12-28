<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Pipeline;
use Rubix\ML\PersistentModel;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Loggers\Screen;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\NeuralNet\Optimizers\Momentum;
use Rubix\ML\Classifiers\LogisticRegression;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;
use League\Csv\Reader;
use League\Csv\Writer;

const MODEL_FILE = 'credit.model';
const PROGRESS_FILE = 'progress.csv';
const REPORT_FILE = 'report.json';

echo '╔═══════════════════════════════════════════════════════════════╗' . PHP_EOL;
echo '║                                                               ║' . PHP_EOL;
echo '║ Credit Card Default Predictor using Logistic Regression       ║' . PHP_EOL;
echo '║                                                               ║' . PHP_EOL;
echo '╚═══════════════════════════════════════════════════════════════╝' . PHP_EOL;
echo PHP_EOL;

echo 'Loading data into memory ...' . PHP_EOL;

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

list($training, $testing) = $dataset->randomize()->stratifiedSplit(0.80);

$estimator = new PersistentModel(new Pipeline([
    new NumericStringConverter(),
    new OneHotEncoder(),
    new ZScaleStandardizer(),
], new LogisticRegression(100, new Momentum(0.001), 1e-4)),
    new Filesystem(MODEL_FILE)
);

$estimator->setLogger(new Screen('credit'));

$estimator->train($training);

$writer = Writer::createFromPath(PROGRESS_FILE, 'w+');
$writer->insertOne(['loss']);
$writer->insertAll(array_map(null, $estimator->steps(), []));

echo 'Progress saved to ' . PROGRESS_FILE . PHP_EOL;

$report = new AggregateReport([
    new MulticlassBreakdown(),
    new ConfusionMatrix(),
]);

$predictions = $estimator->predict($testing);

$results = $report->generate($predictions, $testing->labels());

file_put_contents(REPORT_FILE, json_encode($results, JSON_PRETTY_PRINT));

echo 'Report saved to ' . REPORT_FILE . PHP_EOL;

$estimator->prompt();
