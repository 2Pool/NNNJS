const Network = require('./index').Network,
  test = require('ava');

// Compute XOR
test('Gets better at XOR', t => {
  const network = new Network(2, 1);

  const pred1 = network.predict([0, 1]);

  network.trainOne({ inputs: [0, 1], outputs: [1] });

  const pred2 = network.predict([0, 1]);

  for (let i = 0; i < 1000; i++) {
    network.trainOne({ inputs: [0, 1], outputs: [1] });
  }

  const pred3 = network.predict([0, 1]);

  console.log('Pred1: ' + pred1);
  console.log('Pred2: ' + pred2);
  console.log('Pred3: ' + pred3);

  t.pass();
});

test('Gets better at XOR for real?', t => {
  const network = new Network(2, 1);

  for (let i = 0; i < 1000; i++) {
    network.trainOne({ inputs: [0, 0], outputs: [0] });
    network.trainOne({ inputs: [1, 0], outputs: [1] });
    network.trainOne({ inputs: [0, 1], outputs: [1] });
    network.trainOne({ inputs: [1, 1], outputs: [0] });
  }

  console.log('[0, 0]: ', network.predict([0, 0]));
  console.log('[1, 0]: ', network.predict([1, 0]));
  console.log('[0, 1]: ', network.predict([0, 1]));
  console.log('[1, 1]: ', network.predict([1, 1]));

  t.pass();
});
