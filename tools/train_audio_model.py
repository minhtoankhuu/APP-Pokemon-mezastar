import argparse
import json
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf

SAMPLE_RATE = 16000
AUG_MARKER = '_aug'


def is_augmented(path: Path) -> bool:
    return AUG_MARKER in path.stem


def load_audio(path: Path, samples: int, augment_crop: bool) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(audio) < samples:
        audio = np.pad(audio, (0, samples - len(audio)))
    elif len(audio) > samples:
        start = random.randint(0, len(audio) - samples) if augment_crop else (len(audio) - samples) // 2
        audio = audio[start:start + samples]
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio.astype(np.float32)


def split_label_files(files):
    """Tach file goc (khong co _aug) va file augment trong 1 thu muc nhan vat."""
    originals = sorted(f for f in files if not is_augmented(f))
    augmented = sorted(f for f in files if is_augmented(f))
    return originals, augmented


def collect_dataset(data_dir: Path, duration: float):
    """Thu thap dataset, chia train/val theo NGUON GOC de tranh ro ri.

    Neu chia ngau nhien tren toan bo file (goc + augment), cac bien the
    augment cua CUNG mot file goc de bi nam ca trong train va val, khien
    val_accuracy o muc cao gia tao (model chi can nho ban ghi goc, khong
    can hoc dac trung that). O day, file goc duoc uu tien danh cho tap
    validation, phan con lai (augment + goc du) dung de train.
    """
    samples = int(SAMPLE_RATE * duration)
    labels = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    if len(labels) < 2:
        raise SystemExit('Can it nhat 2 nhan vat de train model.')

    train_paths, train_y = [], []
    val_paths, val_y = [], []
    counts = {}

    for idx, label in enumerate(labels):
        folder = data_dir / label
        audio_files = []
        for ext in ('*.wav', '*.mp3', '*.m4a', '*.ogg', '*.flac'):
            audio_files.extend(folder.glob(ext))
        counts[label] = len(audio_files)
        if not audio_files:
            print(f'Bo qua {label}: khong co audio.')
            continue

        originals, augmented = split_label_files(audio_files)

        if len(originals) >= 2:
            n_val = max(1, round(len(originals) * 0.2))
            val_files = originals[:n_val]
            train_files = originals[n_val:] + augmented
        elif len(originals) == 1:
            val_files = originals
            train_files = augmented
        else:
            # Khong co file goc (truong hop hiem), danh 1 file augment lam val.
            val_files = augmented[:1]
            train_files = augmented[1:]

        if not train_files:
            # Qua it du lieu de tach rieng val, dung chung cho ca train.
            train_files = val_files

        for file in train_files:
            train_paths.append((file, idx))
        for file in val_files:
            val_paths.append((file, idx))

    if not train_paths:
        raise SystemExit('Khong tim thay audio hop le.')

    def load_all(pairs, augment_crop):
        x, y = [], []
        for file, idx in pairs:
            try:
                x.append(load_audio(file, samples, augment_crop))
                y.append(idx)
            except Exception as exc:
                print(f'Khong doc duoc {file}: {exc}')
        return np.stack(x), np.array(y, dtype=np.int64)

    x_train, y_train = load_all(train_paths, augment_crop=True)
    x_val, y_val = load_all(val_paths, augment_crop=False)

    return x_train, y_train, x_val, y_val, labels, counts


def build_model(input_samples: int, num_classes: int) -> tf.keras.Model:
    l2 = tf.keras.regularizers.l2(1e-4)
    inputs = tf.keras.Input(shape=(input_samples, 1), name='audio')
    x = tf.keras.layers.GaussianNoise(0.01)(inputs)
    x = tf.keras.layers.Conv1D(16, 9, strides=2, padding='same', activation='relu', kernel_regularizer=l2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(32, 9, strides=2, padding='same', activation='relu', kernel_regularizer=l2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(64, 9, strides=2, padding='same', activation='relu', kernel_regularizer=l2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(96, 9, strides=2, padding='same', activation='relu', kernel_regularizer=l2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(128, 9, strides=2, padding='same', activation='relu', kernel_regularizer=l2)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='character')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def per_class_val_report(model, x_val, y_val, labels):
    if len(x_val) == 0:
        return {}
    preds = np.argmax(model.predict(x_val, verbose=0), axis=1)
    report = {}
    for idx, label in enumerate(labels):
        mask = y_val == idx
        total = int(mask.sum())
        if total == 0:
            continue
        correct = int((preds[mask] == idx).sum())
        report[label] = {'correct': correct, 'total': total}
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/raw')
    parser.add_argument('--out', default='models')
    parser.add_argument('--duration', type=float, default=2.0)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding='utf-8')
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_samples = int(SAMPLE_RATE * args.duration)
    x_train, y_train, x_val, y_val, labels, counts = collect_dataset(data_dir, args.duration)
    x_train = x_train[..., np.newaxis]
    x_val = x_val[..., np.newaxis] if len(x_val) else x_val

    print(f'Train: {len(x_train)} mau, Val (chi tu file goc, khong augment): {len(x_val)} mau')

    model = build_model(input_samples, len(labels))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=5, factor=0.5),
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val) if len(x_val) else None,
        epochs=args.epochs,
        batch_size=16,
        callbacks=callbacks,
    )

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    (out_dir / 'pokemon_voice.tflite').write_bytes(tflite_model)
    (out_dir / 'labels.txt').write_text('\n'.join(labels), encoding='utf-8')

    per_class = per_class_val_report(model, x_val, y_val, labels)

    report = {
        'sample_rate': SAMPLE_RATE,
        'duration_seconds': args.duration,
        'input_samples': input_samples,
        'labels': labels,
        'audio_counts': counts,
        'train_samples': int(len(x_train)),
        'val_samples': int(len(x_val)),
        'last_train_accuracy': float(history.history['accuracy'][-1]),
        'last_val_accuracy': float(history.history['val_accuracy'][-1]) if 'val_accuracy' in history.history else None,
        'per_class_val': per_class,
        'note': (
            'Validation duoc lay tu file audio GOC (khong qua augment) cho tung nhan vat, '
            'train tren cac file augment con lai, de tranh ro ri du lieu giua train/val. '
            'Voi cac nhan vat chi co dung 1 file audio that, tap validation van chi phan anh '
            'kha nang nhan dien 1 ban ghi duy nhat, chua the danh gia day du kha nang tong quat hoa '
            'sang giong noi/moi truong ghi am khac. Nen bo sung them audio that cho tung nhan vat.'
        ),
    }
    (out_dir / 'training_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Da tao model:', out_dir / 'pokemon_voice.tflite')


if __name__ == '__main__':
    main()
