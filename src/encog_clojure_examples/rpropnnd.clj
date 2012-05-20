(ns encog_clojure_examples.rpropnnd
  (:import (org.encog Encog)
           (org.encog.engine.network.activation ActivationSigmoid)
           (org.encog.ml.data MLData MLDataPair MLDataSet)
           (org.encog.ml.data.basic BasicMLData BasicMLDataSet)
           (org.encog.neural.networks BasicNetwork)
           (org.encog.neural.networks.layers BasicLayer)
           (org.encog.neural.networks.training.propagation.resilient ResilientPropagation)))

(defn double-2d-array [inpvec] (into-array (map double-array inpvec)))

(defn make-training-set [input ideal]
  (BasicMLDataSet.
   (double-2d-array input)
   (double-2d-array ideal)))

(defn mknet [fst lst coll]
  (let [net (BasicNetwork.)]
    (.addLayer net (BasicLayer. nil true fst))
    (doseq [tcn coll]
      (.addLayer net (BasicLayer. (ActivationSigmoid.) true tcn)))
    (.addLayer net (BasicLayer. (ActivationSigmoid.) false lst))
    (.. net getStructure finalizeStructure)
    (.reset net)
    net))

(defn make-network [& npl]
  (if (> (count npl) 1)
    (let [inp-layer (first npl) out-layer (last npl) hidden-layers (drop-last (rest npl))]
      (mknet inp-layer out-layer hidden-layers))))

(defn train [network training-set]
  (let [tdata (doto
                (ResilientPropagation. network training-set)
                (.iteration))]
    (loop [epoch 1]
      (when (>= (.getError tdata) 0.01)
        (.iteration tdata)
        (println "Epoch #" epoch " Error:" (.getError tdata))
        (recur (inc epoch))))))

(defn get-result [network input]
  (let [result (.compute network (BasicMLData. (double-array input)))]
    (map #(.getData result %) (range 0 (.size result)))))

(defn print-result [network input] (println input " --> " (get-result network input)))
