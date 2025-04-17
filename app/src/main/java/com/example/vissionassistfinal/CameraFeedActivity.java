package com.example.vissionassistfinal;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.util.Size;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;
import com.google.common.util.concurrent.ListenableFuture;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.concurrent.ExecutionException;

public class CameraFeedActivity extends AppCompatActivity {

    private static final int CAMERA_PERMISSION_REQUEST_CODE = 1001;
    // 1 frame every 3 seconds => 3000 ms
    private static final long FRAME_INTERVAL_MS = 3000;

    private PreviewView previewView;
    private Button btnToggleCapture;
    private boolean isCapturing = false;
    private long lastCaptureTime = 0L;

    private ProcessCameraProvider cameraProvider;
    private ImageAnalysis imageAnalysis;
    private YuvToRgbConverter yuvToRgbConverter;

    // Buffer folder path (created in app's external Pictures folder)
    private File bufferFolder;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera_feed);

        // Initialize Chaquopy if not already started
        if (!Python.isStarted()) {
            Python.start(new AndroidPlatform(this));
        }

        previewView = findViewById(R.id.previewView);
        btnToggleCapture = findViewById(R.id.btnToggleCapture);
        yuvToRgbConverter = new YuvToRgbConverter();

        btnToggleCapture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                toggleCapture();
            }
        });

        // Check camera permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA},
                    CAMERA_PERMISSION_REQUEST_CODE);
        } else {
            startCamera();
        }
    }

    private void toggleCapture() {
        if (!isCapturing) {
            // Starting capture: Create buffer folder and set flag
            createBufferFolder();
            isCapturing = true;
            btnToggleCapture.setText("Stop Capture");
        } else {
            // Stopping capture: Reset flag and delete buffer folder
            isCapturing = false;
            btnToggleCapture.setText("Start Capture");
            deleteBufferFolder();
        }
    }

    private void createBufferFolder() {
        File picturesDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        if (picturesDir != null) {
            bufferFolder = new File(picturesDir, "buffer");
            Log.d("buffer path", bufferFolder.toString());
            if (!bufferFolder.exists()) {
                bufferFolder.mkdirs();
            }
        }
    }

    private void deleteBufferFolder() {
        if (bufferFolder != null && bufferFolder.exists()) {
            File[] files = bufferFolder.listFiles();
            if (files != null) {
                for (File file : files) {
                    file.delete();
                }
            }
            bufferFolder.delete();
        }
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() -> {
            try {
                cameraProvider = cameraProviderFuture.get();
                bindPreviewAndAnalysis();
            } catch (ExecutionException | InterruptedException e) {
                e.printStackTrace();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void bindPreviewAndAnalysis() {
        cameraProvider.unbindAll();

        // Preview use case for the live camera feed
        Preview preview = new Preview.Builder().build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        // ImageAnalysis use case to capture frames
        imageAnalysis = new ImageAnalysis.Builder()
                .setTargetResolution(new Size(1280, 720))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();

        imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(this), new ImageAnalysis.Analyzer() {
            @Override
            public void analyze(@NonNull ImageProxy image) {
                if (isCapturing && bufferFolder != null) {
                    long currentTime = System.currentTimeMillis();
                    if ((currentTime - lastCaptureTime) >= FRAME_INTERVAL_MS) {
                        lastCaptureTime = currentTime;
                        Bitmap bitmapFrame = yuvToRgbConverter.toBitmap(image);
                        File savedFrame = saveFrameToBufferFile(bitmapFrame);
                        if (savedFrame != null) {
                            // Call Python with the buffer folder path (not individual file)
                            callPythonScript(bufferFolder.getAbsolutePath());
                        }
                    }
                }
                image.close();
            }
        });

        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();

        cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);
    }

    // Save each frame as a new file in the buffer folder (without compressing as JPEG if desired)
    private File saveFrameToBufferFile(Bitmap bitmap) {
        try {
            if (bufferFolder == null || !bufferFolder.exists()) {
                createBufferFolder();
            }
            // Use PNG format to save a lossless image (change extension accordingly)
            String fileName = "frame_" + System.currentTimeMillis() + ".png";
            File outputFile = new File(bufferFolder, fileName);
            FileOutputStream fos = new FileOutputStream(outputFile);
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, fos);
            fos.flush();
            fos.close();
            Log.d("Adding Frames ", "adding frame to buffer: " + fileName);
            return outputFile;
        } catch (Exception e) {
            e.printStackTrace();
            Log.d(" failed Adding Frames ", "adding frame to buffer ");
            return null;
        }
    }

    // Updated call: Pass the buffer folder path to the Python script.
    private void callPythonScript(String bufferFolderPath) {
        try {
            Python py = Python.getInstance();
            // Get the base writable directory from the app's internal storage
            File baseDir = getFilesDir();

            // Create a folder named "ultralytics" within the base directory
            File ultralyticsFolder = new File(baseDir, "ultralytics");
            if (!ultralyticsFolder.exists()) {
                if (!ultralyticsFolder.mkdirs()) {
                    throw new IOException("Failed to create ultralytics directory");
                }
            }
            // Get the absolute path of the ultralytics folder
            String ultralyticsPath = ultralyticsFolder.getAbsolutePath();
            Log.d("TEST",ultralyticsPath);
            // First, ensure the environment is set up using the ultralytics folder as home
            py.getModule("Final.main_ui_frames").callAttr(
                    "setup_environment", ultralyticsPath);

            // Then, call the main function and pass the ultralytics folder as the --home argument
            py.getModule("Final.main_ui_frames").callAttr(
                    "initialize_and_run",
                    "--folder", bufferFolderPath,
                    "--fps", "2",
                    "--home", ultralyticsPath
            );

            runOnUiThread(() -> Toast.makeText(
                    CameraFeedActivity.this,
                    "Called Python with folder: " + bufferFolderPath,
                    Toast.LENGTH_SHORT).show());
        } catch (Exception e) {
            e.printStackTrace();
            runOnUiThread(() -> Toast.makeText(
                    CameraFeedActivity.this,
                    "Error calling Python: " + e.getMessage(),
                    Toast.LENGTH_LONG).show());
        }
    }





    // Handle camera permission results
    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE) {
            if (grantResults.length > 0 &&
                    grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startCamera();
            } else {
                Toast.makeText(this, "Camera permission denied.",
                        Toast.LENGTH_LONG).show();
            }
        }
    }
}
