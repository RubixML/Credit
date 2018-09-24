<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\PersistentModel;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Reports\AggregateReport;
use Rubix\ML\Reports\ConfusionMatrix;
use Rubix\ML\Reports\PredictionSpeed;
use Rubix\ML\Reports\MulticlassBreakdown;
use League\Csv\Reader;

const PREDICTIONS_FILE = 'predictions.json';
const PROBS_FILE = 'probabilities.json';

echo '╔═══════════════════════════════════════════════════════════════╗' . "\n";
echo '║                                                               ║' . "\n";
echo '║ Credit Card Default Predictor using Logistic Regression       ║' . "\n";
echo '║                                                               ║' . "\n";
echo '╚═══════════════════════════════════════════════════════════════╝' . "\n";
echo "\n";

$reader = Reader::createFromPath(__DIR__ . '/unknown.csv')
    ->setDelimiter(',')->setEnclosure('"')->setHeaderOffset(0);

$samples = iterator_to_array($reader->getRecords([
    'credit_limit', 'gender', 'education', 'marital_status', 'age',
    'timeliness_1', 'timeliness_2', 'timeliness_3', 'timeliness_4',
    'timeliness_5', 'timeliness_6', 'balance_1', 'balance_2', 'balance_3',
    'balance_4', 'balance_5', 'balance_6', 'payment_1', 'payment_2',
    'payment_3', 'payment_4', 'payment_5', 'payment_6', 'avg_balance',
    'avg_payment',
]));

$dataset = new Unlabeled($samples);

$estimator = PersistentModel::restore('credit.model');

echo 'Computing predictions ... ';

$start = microtime(true);

$predictions = $estimator->predict(clone $dataset);
$probabilities = $estimator->proba(clone $dataset);

echo 'done in ' . (string) (microtime(true) - $start) . ' seconds.' . "\n";

file_put_contents(PREDICTIONS_FILE, json_encode($predictions, JSON_PRETTY_PRINT));
file_put_contents(PROBS_FILE, json_encode($probabilities, JSON_PRETTY_PRINT));

echo 'Predictions saved to ' . PREDICTIONS_FILE . '.' . "\n";
echo 'Probabilities saved to ' . PROBS_FILE . '.' . "\n";
