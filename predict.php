<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\PersistentModel;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Persisters\Filesystem;
use League\Csv\Reader;

const MODEL_FILE = 'credit.model';
const PREDICTIONS_FILE = 'predictions.json';
const PROBS_FILE = 'probabilities.json';

echo '╔═══════════════════════════════════════════════════════════════╗' . PHP_EOL;
echo '║                                                               ║' . PHP_EOL;
echo '║ Credit Card Default Predictor using Logistic Regression       ║' . PHP_EOL;
echo '║                                                               ║' . PHP_EOL;
echo '╚═══════════════════════════════════════════════════════════════╝' . PHP_EOL;
echo PHP_EOL;

echo 'Loading data into memory ...' . PHP_EOL;

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

$estimator = PersistentModel::load(new Filesystem(MODEL_FILE));

$probabilities = $estimator->proba($dataset);

file_put_contents(PROBS_FILE, json_encode($probabilities, JSON_PRETTY_PRINT));

echo 'Probabilities saved to ' . PROBS_FILE . PHP_EOL;
