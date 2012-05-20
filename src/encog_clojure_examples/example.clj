(ns encog_clojure_examples.example
   (:use [encog_clojure_examples.rpropnnd]))

(def XOR_INPUT [[0.0 0.0] [1.0 0.0] [0.0 1.0] [1.0 1.0]])
(def XOR_IDEAL [[0.0] [1.0] [1.0] [0.0]])

(def training-set (make-training-set XOR_INPUT XOR_IDEAL))
(def network (make-network 2 10 20 10 1))

(train network training-set)

(doseq [x XOR_INPUT] (print-result network x))
