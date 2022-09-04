async function init() {
  model = await tf.loadLayersModel(
    'https://raw.githubusercontent.com/derrydwi/model-test/main/model.json'
  );

  const stream = await navigator.mediaDevices.getUserMedia({
    video: true,
  });

  video = document.createElement('video');
  video.height = 100;
  video.width = 100;
  video.autoplay = true;
  video.srcObject = stream;

  setInterval(() => {
    predict();
  }, 1000);
}

const predict = async () => {
  const tensor = tf.browser
    .fromPixels(video)
    .resizeNearestNeighbor([48, 48])
    .mean(2)
    .toFloat()
    .expandDims(0)
    .expandDims(-1);
  const result = await model.predict(tensor).arraySync()[0];
  const label = [
    'angry',
    'disgust',
    'fear',
    'happy',
    'sad',
    'surprise',
    'neutral',
  ];
  const parsedResult = Object.fromEntries(
    label.map((name, index) => [name, result[index]])
  );

  console.log({ proba: parsedResult, predict: getExpression(parsedResult) });
};

const getExpression = (expressions) => {
  const maxValue = Math.max(
    ...Object.values(expressions).filter((value) => value <= 1)
  );
  const expressionsKeys = Object.keys(expressions);
  const mostLikely = expressionsKeys.find(
    (expression) => expressions[expression] === maxValue
  );
  return mostLikely ? mostLikely : 'neutral';
};

init();
