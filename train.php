<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Pipeline;
use Rubix\ML\PersistentModel;
use Rubix\ML\Datasets\Labeled;
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

echo '╔═══════════════════════════════════════════════════════════════╗' . "\n";
echo '║                                                               ║' . "\n";
echo '║ Credit Card Default Predictor using Logistic Regression       ║' . "\n";
echo '║                                                               ║' . "\n";
echo '╚═══════════════════════════════════════════════════════════════╝' . "\n";
echo "\n";

$reader = Reader::createFromPath(__DIR__ . '/dataset.csv')
    ->setDelimiter(',')->setEnclosure('"')->setHeaderOffset(0);

$samples = iterator_to_array($reader->getRecords([
    'credit_limit', 'gender', 'education', 'marital_status', 'age',
    'timeliness_1', 'timeliness_2', 'timeliness_3', 'timeliness_4',
    'timeliness_5', 'timeliness_6', 'balance_1', 'balance_2', 'balance_3',
    'balance_4', 'balance_5', 'balance_6', 'payment_1', 'payment_2',
    'payment_3', 'payment_4', 'payment_5', 'payment_6', 'avg_balance',
    'avg_payment',
]));

$labels = iterator_to_array($reader->fetchColumn('default'));

$dataset = new Labeled($samples, $labels);

$estimator = new PersistentModel(new Pipeline(new LogisticRegression(100, 100,
    new Adam(0.001), 1e-4, new CrossEntropy(), 1e-5), [
        new NumericStringConverter(),
        new OneHotEncoder(),
        new ZScaleStandardizer(),
    ])
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

$end = microtime(true);

echo ' completed  in ' . (string) ($end - $start) . ' seconds.' . "\n";

$table = array_map(null, $estimator->steps(), []);

$writer = Writer::createFromPath('loss.csv', 'w+');
$writer->insertOne(['loss']);
$writer->insertAll($table);

echo "\n";

echo 'Reports:' . "\n";

var_dump($report->generate($estimator, $testing));

echo "\n";

echo 'Example test predictions:' . "\n";

var_dump($estimator->proba($testing->randomize()->head(3)));

echo "\n";

$save = readline('Save this model? (y|N): ');

if (strtolower($save) === 'y') {
    $estimator->save('credit.model');

    echo 'Saved.' . "\n";
}
