import numpy as np

import tensorflow as tf
import tf_keras

ECHOFINE_VIEWS = {"0": "a2ch-full", "1": "a2ch-la", "2": "a2ch-lv", "3": "a3ch-full", "4": "a3ch-la", "5": "a3ch-lv", "6": "a3ch-outflow", 
                  "7": "a4ch-full", "8": "a4ch-ias", "9": "a4ch-la", "10": "a4ch-lv", "11": "a4ch-ra", "12": "a4ch-rv", "13": "a5ch-full", 
                  "14": "a5ch-outflow", "15": "apex", "16": "doppler-ao-descending", "17": "doppler-av", "18": "doppler-mv", "19": "doppler-pv", 
                  "20": "doppler-tissue-lateral", "21": "doppler-tissue-rv", "22": "doppler-tissue-septal", "23": "doppler-tv", "24": "mmode-a4ch-rv", 
                  "25": "mmode-ivc", "26": "mmode-plax-av", "27": "mmode-plax-lv", "28": "mmode-plax-mitral", "29": "plax-full-la", "30": "plax-full-lv", 
                  "31": "plax-full-mv", "32": "plax-full-out", "33": "plax-full-rv-ao", "34": "plax-tv", "35": "plax-valves-av", "36": "plax-valves-mv", 
                  "37": "psax-all", "38": "psax-av", "39": "psax-lv-apex", "40": "psax-lv-base", "41": "psax-lv-mid", "42": "psax-pv", "43": "psax-tv", 
                  "44": "subcostal-heart", "45": "subcostal-ivc", "46": "suprasternal"}


def setup_echofine_view_classifier(weights, device_id: int):
    """Initialize the Keras EchoForge view classifier on `device`."""
    with tf.device(f"GPU:{device_id}"):
        model = tf_keras.models.load_model(weights)
    return model


def classify_view_echofine(echofine_model, video: np.ndarray, batch_size: int = 32) -> np.ndarray:
    """Run EchoForge on `video` using this worker's Keras model."""
    B, H, W, C = video.shape
    assert (H, W) == (224, 224), f"Expected image resolution (224, 224), got {(H, W)}"
    assert C == 3, f"Expected num_channels=3, got {C}"
    extra_batch = 0 if B % batch_size == 0 else 1
    probits = []
    for k in range(B // batch_size + extra_batch):
        image_batch = video[k * batch_size:min((k + 1) * batch_size, B)]
        # Direct call instead of .predict(): no per-call progress bar or tf.data overhead
        probits.append(np.asarray(echofine_model(image_batch, training=False)))
    return np.concatenate(probits, axis=0)
