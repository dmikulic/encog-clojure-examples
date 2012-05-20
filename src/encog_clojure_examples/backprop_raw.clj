(import
    '(org.encog Encog)
    '(org.encog.engine.network.activation ActivationSigmoid)
    '(org.encog.ml.data MLData MLDataPair MLDataSet)
    '(org.encog.ml.data.basic BasicMLData BasicMLDataSet)
    '(org.encog.neural.networks BasicNetwork)
    '(org.encog.neural.networks.layers BasicLayer)
    '(org.encog.neural.networks.training.propagation.back Backpropagation))

(def XOR_INPUT [[0.0 0.0] [1.0 0.0] [0.0 1.0] [1.0 1.0]])

(def XOR_IDEAL [[0.0] [1.0] [1.0] [0.0]])

(defn double-2d-array [vec] (into-array (map double-array vec)))

(def training-set  (BasicMLDataSet. (double-2d-array XOR_INPUT) (double-2d-array XOR_IDEAL)))

(def network
  (doto (BasicNetwork.)
    (.addLayer (BasicLayer. nil false 2))
    (.addLayer (BasicLayer. (ActivationSigmoid.) true 3))
    (.addLayer (BasicLayer. (ActivationSigmoid.) true 1))
    (.. getStructure finalizeStructure)
    (.reset)))


(def train (Backpropagation. network training-set 0.7 0.8))
(.iteration train)

(loop [epoch 1]
  (when (>= (.getError train) 0.01)
      (.iteration train)
      (println "Epoch #" epoch " Error:" (.getError train))
      (recur (inc epoch))))

(println "Neural Network Results:")

(defn get-result [input] (.getData (.compute network (BasicMLData. input)) 0))

(defn print-result [input] (println (vec input) " --> " (get-result input)))

(doseq [x (double-2d-array XOR_INPUT)] (print-result x))
