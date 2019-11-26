<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Embedders\TSNE;
use Rubix\ML\Other\Loggers\Screen;
use League\Csv\Reader;
use League\Csv\Writer;

ini_set('memory_limit', '-1');

echo 'Loading data into memory ...' . PHP_EOL;

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

$dataset = Unlabeled::fromIterator($samples);

$dataset->apply(new NumericStringConverter());

$stats = $dataset->describe();

file_put_contents('stats.json', json_encode($stats, JSON_PRETTY_PRINT));

echo 'Stats saved to stats.json' . PHP_EOL;

$dataset = $dataset->randomize()->head(1000);

$dataset->apply(new OneHotEncoder())
    ->apply(new ZScaleStandardizer());

$embedder = new TSNE(2, 20.0, 20);

$embedder->setLogger(new Screen('credit'));

echo 'Embedding ...' . PHP_EOL;

$embedding = $embedder->embed($dataset);

$writer = Writer::createFromPath('embedding.csv', 'w+');
$writer->insertOne(['x', 'y']);
$writer->insertAll($embedding);

echo 'Embedding saved to embedding.csv' . PHP_EOL;