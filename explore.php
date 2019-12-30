<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Extractors\CSV;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Embedders\TSNE;
use Rubix\ML\Other\Loggers\Screen;
use League\Csv\Writer;

ini_set('memory_limit', '-1');

echo 'Loading data into memory ...' . PHP_EOL;

$dataset = Labeled::fromIterator(new CSV('dataset.csv', true))
    ->apply(new NumericStringConverter());

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