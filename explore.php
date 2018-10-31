<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Manifold\TSNE;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Kernels\Distance\Minkowski;
use Rubix\ML\Transformers\NumericStringConverter;
use League\Csv\Reader;
use League\Csv\Writer;

const OUTPUT_FILE = 'tsne.csv';

echo '╔═══════════════════════════════════════════════════════════════╗' . "\n";
echo '║                                                               ║' . "\n";
echo '║ Credit Card Dataset Visualizer using t-SNE                    ║' . "\n";
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

$dataset = Labeled::fromIterator($samples, $labels)->randomize()->head(200);

$converter = new NumericStringConverter();
$encoder = new OneHotEncoder();

$dataset->apply($converter);

$encoder->fit($dataset);

$dataset->apply($encoder);

$embedder = new TSNE(2, 30, 12., 1000, 1.0, 0.3, 1e-6, new Minkowski(3.));

echo 'Embedding started ...';

$start = microtime(true);

$samples = $embedder->embed($dataset);

echo ' done  in ' . (string) (microtime(true) - $start) . ' seconds.' . "\n";

$dataset = Labeled::quick($samples, $dataset->labels());

$writer = Writer::createFromPath(OUTPUT_FILE, 'w+');
$writer->insertOne(['x', 'y', 'label']);
$writer->insertAll($dataset->zip());

echo 'Embedding saved to ' . OUTPUT_FILE . '.' . "\n";