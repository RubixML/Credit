<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Pipeline;
use Rubix\ML\PersistentModel;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Reports\AggregateReport;
use Rubix\ML\Reports\ConfusionMatrix;
use Rubix\ML\Reports\PredictionSpeed;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Reports\MulticlassBreakdown;
use Rubix\ML\Classifiers\LogisticRegression;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;
use League\Csv\Reader;
use League\Csv\Writer;

const MODEL_FILE = 'credit.model';
const PROGRESS_FILE = 'progress.csv';
const REPORT_FILE = 'report.json';

echo '╔═══════════════════════════════════════════════════════════════╗' . "\n";
echo '║                                                               ║' . "\n";
echo '║ Credit Card Default Predictor using Logistic Regression       ║' . "\n";
echo '║                                                               ║' . "\n";
echo '╚═══════════════════════════════════════════════════════════════╝' . "\n";
echo "\n";

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

$estimator = new PersistentModel(new Pipeline(new LogisticRegression(100,
    new Adam(0.001), 1e-4, 300, 1e-4, new CrossEntropy()), [
        new NumericStringConverter(),
        new OneHotEncoder(),
        new ZScaleStandardizer(),
    ]),
    new Filesystem(MODEL_FILE)
);

$report = new AggregateReport([
    new MulticlassBreakdown(),
    new ConfusionMatrix(),
    new PredictionSpeed(),
]);

list($training, $testing) = $dataset->randomize()->stratifiedSplit(0.80);


echo 'Training started ...';

$start = microtime(true);

$estimator->train($training);

echo ' done  in ' . (string) (microtime(true) - $start) . ' seconds.' . "\n";

$writer = Writer::createFromPath(PROGRESS_FILE, 'w+');
$writer->insertOne(['loss']);
$writer->insertAll(array_map(null, $estimator->steps(), []));

echo 'Propgress saved to ' . PROGRESS_FILE . '.' . "\n";

echo "\n";


echo 'Generating report ...';

$start = microtime(true);

file_put_contents(REPORT_FILE, json_encode($report->generate($estimator,
    $testing), JSON_PRETTY_PRINT));

echo ' done  in ' . (string) (microtime(true) - $start) . ' seconds.' . "\n";

echo 'Report saved to ' . REPORT_FILE . '.' . "\n";

echo "\n";


$save = readline('Save this model? (y|[n]): ');

if (strtolower($save) === 'y') {
    $estimator->save();

    echo 'Model saved to ' . MODEL_FILE . '.' . "\n";
}
