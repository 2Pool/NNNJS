const math = require('mathjs');

// TODO regularization to prevent overfitting

class Network {
  constructor(numInputs, numOutputs) {
    const numHidden = Math.ceil((numInputs + numOutputs) / 2);

    this.hiddenLayer = new Layer(numInputs, numHidden);
    this.outputLayer = new Layer(numHidden, numOutputs);
  }

  /**
   * Takes inputs and outputs of type Array
   * Returns Void
   */
  trainOne({inputs, outputs}) {
    const hiddenOutputs = this.hiddenLayer.predict(inputs);
    const actualOutputs = this.outputLayer.predict(hiddenOutputs);

    const actualErrors = math.subtract(actualOutputs, outputs);
    const hiddenErrors = this.outputLayer.feedback({errors: actualErrors});
    this.hiddenLayer.feedback({errors: hiddenErrors});
  }

  /**
   * Takes an array with size numInputs
   * Returns an array with size numOutputs
   */
  predict(inputs) {
    const hiddenOutputs = this.hiddenLayer.predict(inputs);
    const actualOutputs = this.outputLayer.predict(hiddenOutputs);

    return actualOutputs.toArray();
  }
}

class Layer {
  constructor(numInputs, numOutputs) {
    this.numInputs = numInputs;
    this.neurons = [...new Array(numOutputs)].map(() => new Neuron(numInputs));
  }

  /**
   * Takes a numOutputs x 1 error matrix
   * Returns a numInputs x 1 error matrix
   */
  feedback({errors}) {
    return this.neurons.reduce((nextErrors, neuron, i) => {
      const deltasWithBias = neuron.feedback({error: errors.get([i])}); // (numInputs + 1) x 1
      const deltas = deltasWithBias.resize([deltasWithBias.size()[0] - 1]); // numInputs x 1

      return math.add(nextErrors, deltas); // numInputs x 1
    }, math.zeros(this.numInputs) /* numInputs x 1 */);
  }

  /**
   * Takes a numInputs x 1 matrix
   * Returns a numOutputs x 1 matrix
   */
  predict(inputs) {
    const mInputs = math.matrix(inputs);
    return math.matrix(this.neurons.map(neuron => neuron.predict(mInputs)));
  }
}

// EVERYTHING IS A MATRIX
class Neuron {
  // this.weights = (n + 1) x 1
  // this.inputs = (n + 1) x 1
  // this.output = scalar

  constructor(n) {
    this.weights = math.range(0, n + 1).map(() => Math.random() + 0.001);
  }

  sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
  }

  sigmoidGradient(z) {
    return this.sigmoid(z) * (1 - this.sigmoid(z));
  }

  /**
   * Takes a scalar error
   * Returns a (n + 1) x 1 error matrix
   */
  feedback({error, numExamples = 1, learningRate = 0.1}) {
    // error provided by NEXT layer, is an error per output
    // provided by LAYER, not just neuron
    // error is -my error- and is a scalar
    //
    // ALGO
    // impact = error * sigmoidGradient(this.output); == scalar

    const update = math.multiply(this.inputs, error);
    const scaledUpdate = math.multiply(update, 1 / numExamples * learningRate);

    this.updateWeights(scaledUpdate);

    const feedback = math.dotMultiply(this.weights, update);
    const gradient = math.dotMultiply(this.inputs, math.subtract(1, this.inputs));
    const derivative = math.dotMultiply(feedback, gradient);

    return derivative;
  }

  /**
   * Takes a n x 1 matrix (vector!!)
   * Returns a scalar
   */
  predict(inputs) {
    // inputs is n x 1

    // SAVE INPUTS to this.inputs
    const inputsWithBias = inputs.clone().resize([inputs.size()[0] + 1], 1);

    this.inputs = inputsWithBias; // (n + 1) x 1

    // compute output
    const mult = math.multiply(this.inputs, this.weights);

    const z = math.multiply(this.inputs, this.weights);
    this.output = this.sigmoid(z);

    return this.output;
  }

  updateWeights(deltas) {
    this.weights = math.subtract(this.weights, deltas);
  }
}

module.exports = {Neuron, Network};
