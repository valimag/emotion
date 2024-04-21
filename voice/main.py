import sounddevice as sd
import numpy as np
import onnxruntime as ort
import librosa
import scipy.io.wavfile as wav

# Определение классов
class_labels = [
    'женщина_злая', 'женщина_спокойная', 'женщина_испуганная', 'женщина_счастливая', 'женщина_грустная',
    'мужчина_злой', 'мужчина_спокойный', 'мужчина_испуганный', 'мужчина_счастливый', 'мужчина_грустный'
]

# Загрузка модели ONNX
def record_audio(duration=10, sample_rate=22050, filename='output.wav'):
    print("Запись началась. Говорите...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Ожидание окончания записи
    print("Запись окончена.")
    wav.write(filename, sample_rate, recording)  # Сохранение записи в файл

# Функция предварительной обработки аудио из файла
def preprocess_audio(audio_path):
    X, sample_rate = librosa.load(audio_path, res_type='kaiser_fast', duration=2.5, sr=22050*2, offset=0.5)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
    mfccs = np.expand_dims(np.expand_dims(mfccs, axis=0), axis=2)
    return mfccs

# Загрузка модели ONNX
sess = ort.InferenceSession('models/emotion_voice.onnx')

# Запись аудио в файл
record_audio()

# Предварительная обработка и инференс
input_data = preprocess_audio('output.wav')
inputs = {sess.get_inputs()[0].name: input_data}
outputs = sess.run(None, inputs)

output_probs = outputs[0]
predicted_class = np.argmax(output_probs)
predicted_label = class_labels[predicted_class]
print(f"Предсказанная эмоция: {predicted_label}")