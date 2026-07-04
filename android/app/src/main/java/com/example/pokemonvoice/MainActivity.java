package com.example.pokemonvoice;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.BitmapFactory;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends Activity {
    private static final int SAMPLE_RATE = 16000;
    private static final int MAX_RECORD_SECONDS = 5;
    private static final int MAX_SAMPLES = SAMPLE_RATE * MAX_RECORD_SECONDS;
    // Do dai co dinh dua vao model, phai khop voi --duration luc train (tools/train_audio_model.py).
    private static final float MODEL_INPUT_SECONDS = 3.0f;
    private static final int INPUT_SAMPLES = (int) (SAMPLE_RATE * MODEL_INPUT_SECONDS);
    private static final int REQUEST_RECORD_AUDIO = 10;

    private TextView resultText;
    private ImageView imageView;
    private Button listenButton;
    private Interpreter interpreter;
    private final List<String> labels = new ArrayList<>();

    private volatile boolean isRecording = false;
    private volatile boolean stopRequested = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        buildUi();
        try {
            interpreter = new Interpreter(loadModelFile());
            loadLabels();
            resultText.setText("Sẵn sàng nghe tiếng nhân vật");
        } catch (Exception e) {
            resultText.setText("Chưa có model.tflite hoặc labels.txt trong assets");
        }
    }

    private void buildUi() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setGravity(Gravity.CENTER_HORIZONTAL);
        root.setPadding(32, 48, 32, 32);

        resultText = new TextView(this);
        resultText.setTextSize(20);
        resultText.setGravity(Gravity.CENTER);
        resultText.setText("Đang tải model...");

        imageView = new ImageView(this);
        imageView.setAdjustViewBounds(true);
        imageView.setMaxHeight(700);
        imageView.setPadding(0, 32, 0, 32);

        listenButton = new Button(this);
        listenButton.setText("Bắt đầu nghe");
        listenButton.setOnClickListener(v -> onListenButtonClicked());

        root.addView(resultText, new LinearLayout.LayoutParams(-1, -2));
        root.addView(imageView, new LinearLayout.LayoutParams(-1, 0, 1));
        root.addView(listenButton, new LinearLayout.LayoutParams(-1, -2));
        setContentView(root);
    }

    private void onListenButtonClicked() {
        if (isRecording) {
            stopRequested = true;
            return;
        }
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, REQUEST_RECORD_AUDIO);
            return;
        }
        if (interpreter == null || labels.isEmpty()) {
            resultText.setText("Bạn cần train và chép model vào assets trước.");
            return;
        }
        isRecording = true;
        stopRequested = false;
        listenButton.setText("Dừng lại");
        resultText.setText("Đang nghe... bấm \"Dừng lại\" khi xong (tối đa " + MAX_RECORD_SECONDS + "s)");
        new Thread(() -> {
            float[] audio = recordAudio();
            Prediction prediction = predict(audio);
            runOnUiThread(() -> showPrediction(prediction));
        }).start();
    }

    private float[] recordAudio() {
        int minBuffer = AudioRecord.getMinBufferSize(
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT
        );
        AudioRecord recorder = new AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                Math.max(minBuffer, MAX_SAMPLES * 2)
        );
        short[] full = new short[MAX_SAMPLES];
        short[] chunk = new short[1600];
        int offset = 0;
        recorder.startRecording();
        while (offset < MAX_SAMPLES && !stopRequested) {
            int toRead = Math.min(chunk.length, MAX_SAMPLES - offset);
            int read = recorder.read(chunk, 0, toRead);
            if (read > 0) {
                System.arraycopy(chunk, 0, full, offset, read);
                offset += read;
            }
        }
        recorder.stop();
        recorder.release();

        return extractLoudestWindow(full, offset);
    }

    /**
     * Nguoi dung co the ghi toi da MAX_RECORD_SECONDS, nhung model duoc train co dinh
     * o MODEL_INPUT_SECONDS. Neu doan ghi duoc dai hon input model, tim cua so
     * MODEL_INPUT_SECONDS co nang luong am thanh (RMS) lon nhat de tranh dua ca doan
     * im lang vao model; neu ngan hon thi dem 0 cho du, giong luc train.
     */
    private float[] extractLoudestWindow(short[] full, int recordedSamples) {
        int start = 0;
        if (recordedSamples > INPUT_SAMPLES) {
            long windowEnergy = 0;
            for (int i = 0; i < INPUT_SAMPLES; i++) windowEnergy += (long) full[i] * full[i];

            long bestEnergy = windowEnergy;
            int bestStart = 0;
            for (int s = 1; s <= recordedSamples - INPUT_SAMPLES; s++) {
                int outIdx = s - 1;
                int inIdx = s - 1 + INPUT_SAMPLES;
                windowEnergy -= (long) full[outIdx] * full[outIdx];
                windowEnergy += (long) full[inIdx] * full[inIdx];
                if (windowEnergy > bestEnergy) {
                    bestEnergy = windowEnergy;
                    bestStart = s;
                }
            }
            start = bestStart;
        }

        int usableSamples = Math.min(INPUT_SAMPLES, recordedSamples - start);
        float peak = 1f;
        for (int i = 0; i < usableSamples; i++) {
            peak = Math.max(peak, Math.abs((float) full[start + i]));
        }

        float[] floats = new float[INPUT_SAMPLES];
        for (int i = 0; i < usableSamples; i++) floats[i] = full[start + i] / peak;
        // Phan con lai (neu doan ghi ngan hon INPUT_SAMPLES) giu nguyen 0f, giong luc train.
        return floats;
    }

    private Prediction predict(float[] audio) {
        float[][][] input = new float[1][INPUT_SAMPLES][1];
        for (int i = 0; i < INPUT_SAMPLES; i++) input[0][i][0] = audio[i];
        float[][] output = new float[1][labels.size()];
        interpreter.run(input, output);

        int best = 0;
        for (int i = 1; i < output[0].length; i++) {
            if (output[0][i] > output[0][best]) best = i;
        }
        return new Prediction(labels.get(best), output[0][best]);
    }

    private void showPrediction(Prediction prediction) {
        isRecording = false;
        listenButton.setText("Bắt đầu nghe");
        resultText.setText(prediction.label + " - " + Math.round(prediction.confidence * 100) + "%");
        try {
            InputStream stream = openImageAsset(prediction.label);
            if (stream != null) imageView.setImageBitmap(BitmapFactory.decodeStream(stream));
        } catch (Exception ignored) {
            imageView.setImageDrawable(null);
        }
    }

    private InputStream openImageAsset(String label) throws Exception {
        String[] extensions = {"png", "jpg", "jpeg", "webp"};
        for (String ext : extensions) {
            try {
                return getAssets().open("images/" + label + "." + ext);
            } catch (Exception ignored) {
            }
        }
        return null;
    }

    private MappedByteBuffer loadModelFile() throws Exception {
        AssetFileDescriptor fileDescriptor = getAssets().openFd("model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private void loadLabels() throws Exception {
        BufferedReader reader = new BufferedReader(new InputStreamReader(getAssets().open("labels.txt")));
        String line;
        while ((line = reader.readLine()) != null) {
            if (!line.trim().isEmpty()) labels.add(line.trim());
        }
        reader.close();
    }

    private static class Prediction {
        final String label;
        final float confidence;
        Prediction(String label, float confidence) {
            this.label = label;
            this.confidence = confidence;
        }
    }
}
