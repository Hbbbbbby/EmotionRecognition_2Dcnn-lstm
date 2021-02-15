import librosa
import pathlib
import numpy as np
from sklearn.model_selection import train_test_split

EmoDB_file_path = '/home/hby/Documents/DataSet/EmoDB'


def get_log_mel_spectrogram(path, n_fft, hop_length, n_mels):
    """
    Extract log mel spectrogram
        1) The length of the raw audio used is 8s long,
        2) and then get the MelSpectrogram,
        2) finally perform logarithmic operation to MelSpectrogram.
    """
    y, sr = librosa.load(path, sr=16000, duration=8)

    file_length = np.size(y)
    if file_length != 128000:
        y = np.concatenate((y, np.zeros(128000-file_length)), axis=0)

    mel_spectrogram = librosa.feature.melspectrogram(y, sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    log_mel_spectrogram = librosa.amplitude_to_db(mel_spectrogram)
    # print(np.shape(log_mel_spectrogram))
    log_mel_spectrogram = log_mel_spectrogram.reshape((-1,))
    return log_mel_spectrogram


def classify_files(path):
    """
    Classify emotion files and count them.
        Position 6 of emotion file name represent emotion which according to the label as follow:
        ( Emotion label letter used german word.)
        ----------------------------
           letter   |   emotion(En)
        ------------+---------------
              W     |   anger
              L     |   boredom
              E     |   disgust
              A     |   anxiety/fear
              F     |   happiness
              T     |   sadness
        ------------+---------------

    Dataset preprocessing.
        Dataset data are divided into

    Return:
        dataset_dict:
            a dict structure with 'total' used to count all file number, and a sub-dict named 'file_dict' which including three keys,
    """
    dataset_dict = {
        'total': 0,
        'file_dict': {
            'W': {'represent': 0, 'count': 0, 'all_data': []},
            'L': {'represent': 1, 'count': 0, 'all_data': []},
            'E': {'represent': 2, 'count': 0, 'all_data': []},
            'A': {'represent': 3, 'count': 0, 'all_data': []},
            'F': {'represent': 4, 'count': 0, 'all_data': []},
            'T': {'represent': 5, 'count': 0, 'all_data': []},
            'N': {'represent': 6, 'count': 0, 'all_data': []}
        }
    }

    wav_path = pathlib.Path(path+'/wav')
    emotion_file_list = [str(file_name) for file_name in wav_path.glob('*.wav')]

    p = len(str(wav_path))

    emotion_label_list = dataset_dict['file_dict'].keys()

    for emotion_label in emotion_label_list:
        """
        count all emotion files
        """
        emotion_classify_file_list = [letter for letter in emotion_file_list if letter[p + 6] == emotion_label]

        files_count = len(emotion_classify_file_list)

        dataset_dict['file_dict'][emotion_label]['count'] = files_count
        dataset_dict['total'] = dataset_dict['total'] + files_count

        emotion_data = [get_log_mel_spectrogram(path, n_fft=2048, hop_length=512, n_mels=128)
                        for path in emotion_classify_file_list]
        dataset_dict['file_dict'][emotion_label]['all_data'] = emotion_data

    return dataset_dict


def load_data(path):
    """

    Return:
        data
    """
    train_data_x = []
    train_data_y = []
    validation_data_x = []
    validation_data_y = []
    test_data_x = []
    test_data_y = []

    dataset_dict = classify_files(path)

    emotion_label_list = dataset_dict['file_dict'].keys()

    for emotion_label in emotion_label_list:
        x = dataset_dict['file_dict'][emotion_label]['all_data']
        count = dataset_dict['file_dict'][emotion_label]['count']
        y = np.full(count, dataset_dict['file_dict'][emotion_label]['represent'])

        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8)

        # print(np.shape(x_train), np.shape(y_train), np.shape(x_test), np.shape(y_test))

        train_data_x = np.append(train_data_x, x_train)
        train_data_y = np.append(train_data_y, y_train)

        validation_data_x = np.append(validation_data_x, x_val)
        validation_data_y = np.append(validation_data_y, y_val)

        test_data_x = np.append(test_data_x, x_test)
        test_data_y = np.append(test_data_y, y_test)

        # print(np.shape(train_data_x), np.shape(train_data_y))

    train_data_x = np.array(train_data_x).reshape(-1, 128, 251, 1)
    train_data_y = np.array(train_data_y)
    validation_data_x = np.array(validation_data_x).reshape(-1, 128, 251, 1)
    validation_data_y = np.array(validation_data_y)

    test_data_x = np.array(test_data_x).reshape(-1, 128, 251, 1)
    test_data_y = np.array(test_data_y)

    # print(train_data_x.shape)
    # print(train_data_y.shape)

    # return train_data_x, train_data_y, test_data_x, test_data_y

    return train_data_x, train_data_y, validation_data_x, validation_data_y, test_data_x, test_data_y


# load_data(EmoDB_file_path)
