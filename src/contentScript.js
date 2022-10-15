'use strict';

import { loadLayersModel, browser } from '@tensorflow/tfjs';
import { load } from '@tensorflow-models/blazeface';

let video;
let recognitionModel;
let blazefaceModel;
let canvas = document.body.appendChild(document.createElement('canvas'));
let ctx = canvas.getContext('2d');

const init = async () => {
  await Promise.all([
    navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
      video = document.createElement('video');
      video.height = 180;
      video.width = 180;
      canvas.height = 180;
      canvas.width = 180;
      video.autoplay = true;
      video.srcObject = stream;
      document.body.appendChild(video);
    }),
    loadLayersModel(
      // 'https://raw.githubusercontent.com/derrydwi/model-test/main/model.json'
      // 'https://raw.githubusercontent.com/StromR/FER_model_tfjs/main/CK%2B_Model/CK%2B_HOG/model.json'
      // 'https://raw.githubusercontent.com/StromR/FER_model_tfjs/main/CK%2B_Model/CK%2B_WFE/model.json'
      // 'https://raw.githubusercontent.com/StromR/FER_model_tfjs/main/FER13_Model/FER_HOG/model.json'
      'https://raw.githubusercontent.com/StromR/FER_model_tfjs/main/LBP_model/model.json'
    ).then((result) => {
      recognitionModel = result;
    }),
    load().then((result) => {
      blazefaceModel = result;
    }),
  ]);

  setInterval(() => {
    detect();
  }, 2000);
};

const detect = async () => {
  const returnTensors = false;
  const predictions = await blazefaceModel.estimateFaces(video, returnTensors);

  ctx.drawImage(video, 0, 0, 180, 180);

  predictions.forEach(async (element) => {
    const croppedImage = ctx.getImageData(
      element.topLeft[0],
      element.topLeft[1] - 10,
      element.bottomRight[0] - element.topLeft[0],
      element.bottomRight[1] - element.topLeft[1]
    );
    ctx.putImageData(croppedImage, 0, 0);
    const tensor = browser
      .fromPixels(croppedImage)
      .resizeNearestNeighbor([48, 48])
      .mean(2)
      .toFloat()
      .expandDims(0)
      .expandDims(-1);
    const result = await recognitionModel.predict(tensor).arraySync()[0];
    const label = [
      'surprise',
      'fear',
      'neutral',
      'angry',
      'disgust',
      'happy',
      'sad',
    ];
    // const label = [
    //   'sadness',
    //   'disgust',
    //   'happy',
    //   'contemp',
    //   'surprise',
    //   'fear',
    //   'anger',
    // ];
    // const label = [
    //   'angry',
    //   'disgust',
    //   'fear',
    //   'happy',
    //   'sad',
    //   'surprise',
    //   'neutral',
    // ];
    const parsedResult = Object.fromEntries(
      label.map((name, index) => [name, result[index]])
    );

    console.log('FER::', {
      proba: parsedResult,
      predict: getExpression(parsedResult),
    });
  });
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
