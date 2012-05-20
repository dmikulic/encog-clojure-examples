# encog-clojure-examples

Hello world examples for Encog 3 in Clojure

Feedforward Neural Network (Perceptron)
  * Backpropagation
  * Resilient Propagation

## Usage

<code>lein repl</code>

```clj
user=> (load "encog_clojure_examples/rpropnnd")
user=> (in-ns 'encog_clojure_examples.rpropnnd)
```

<code>rpropnnd</code> contains functions to create a neural network with predefined properties (activation function, training)


```clj
user=> (def XOR_INPUT [[0.0 0.0] [1.0 0.0] [0.0 1.0] [1.0 1.0]])
user=> (def XOR_IDEAL [[0.0] [1.0] [1.0] [0.0]])

user=> (def training-set (make-training-set XOR_INPUT XOR_IDEAL))
user=> (def network (make-network 2 10 20 10 1))

user=> (train network training-set)
Epoch # 1  Error: 0.365003463818111
Epoch # 2  Error: 0.3432473243463201
Epoch # 3  Error: 0.25168279151848605
Epoch # 4  Error: 0.2821308623821046
Epoch # 5  Error: 0.25265135059601135
Epoch # 6  Error: 0.2365839166284867
Epoch # 7  Error: 0.24056300558822408
Epoch # 8  Error: 0.2238483037368802
Epoch # 9  Error: 0.21469238494639312
Epoch # 10  Error: 0.20547367105629846
Epoch # 11  Error: 0.19104940839020834
Epoch # 12  Error: 0.1789146430396987
Epoch # 13  Error: 0.1570771456919285
Epoch # 14  Error: 0.1345197078305611
Epoch # 15  Error: 0.10975405826293298
Epoch # 16  Error: 0.08706495681604509
Epoch # 17  Error: 0.06467544724710089
Epoch # 18  Error: 0.04574833236926629
Epoch # 19  Error: 0.029237035270946333
Epoch # 20  Error: 0.015202982302549939
Epoch # 21  Error: 0.006746934061242
nil

user=> (doseq [x XOR_INPUT] (print-result network x))
[0.0 0.0]  -->  (0.028933765611090848)
[1.0 0.0]  -->  (0.9257828848999123)
[0.0 1.0]  -->  (0.9894454891912495)
[1.0 1.0]  -->  (0.043607393890605715)
nil
```

<code>make-network (num of neurons on layer 1) (num of neurons on layer 2) ...</code>  


## License

Copyright (C) 2011 Darko Mikulic

Distributed under the Eclipse Public License, the same as Clojure.
