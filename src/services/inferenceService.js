const tf = require('@tensorflow/tfjs-node');
const InputError = require('../exceptions/InputError');

async function predictClassification(model, image) {
    try {
        const tensor = tf.node
            .decodeJpeg(image)
            .resizeNearestNeighbor([224, 224])
            .expandDims()
            .toFloat();

        const classes = ['Cancer', 'Non-cancer'];

        const prediction = model.predict(tensor);
        const score = await prediction.data();
        const confidenceScore = Math.max(...score) * 100;

        const maxIndex = score.indexOf(Math.max(...score));
        const label = confidenceScore >= 50 ? classes[maxIndex] : 'Non-cancer';
 
        let suggestion;
 
        if (label === 'Cancer') {
          suggestion = "Segera periksa ke dokter!"
        }
      
        if (label === 'Non-cancer') {
          suggestion = "Anda sehat"
        }
 
  return { confidenceScore, label, suggestion }
    } catch (error) {
        throw new InputError(`Terjadi kesalahan dalam melakukan prediksi`)
    }
}

module.exports = predictClassification;