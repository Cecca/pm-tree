# PM-tree

An implementation of the [PM-tree](http://siret.ms.mff.cuni.cz/skopal/pub/kNNPMtree.pdf) data structure for Nearest Neighbors search.

**WARNING**: this is very early stage, and not ready for usage.

## Timings

Some preliminary performance numbers, on one million 25-dimensional GloVe vectors, with range queries with radius 2.0 and 4.0 and different parameterizations of the index.

```
group                               time (millis)
-----                               -------------
range query/B = 32 P = 4 r = 2.0      36.8±6.21msc
range query/B = 64 P = 4 r = 2.0      37.9±5.31msc
range query/B = 32 P = 8 r = 2.0      25.9±1.10msc
range query/B = 64 P = 8 r = 2.0      27.3±0.70msc

range query/B = 32 P = 4 r = 4.0     176.2±1.72msc
range query/B = 64 P = 4 r = 4.0    214.5±18.34msc
range query/B = 32 P = 8 r = 4.0     173.6±2.74msc
range query/B = 64 P = 8 r = 4.0     191.6±2.21msc
```
