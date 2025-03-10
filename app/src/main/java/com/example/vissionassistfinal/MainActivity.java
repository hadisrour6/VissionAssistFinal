package com.example.vissionassistfinal;

import android.content.ContentResolver;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;
import java.io.File;

public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_CODE_PICK_VIDEO = 1001; // Request code for selecting video
    private String selectedVideoPath = null;  // Store selected video path

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Initialize Chaquopy (Only if not already initialized)
        if (!Python.isStarted()) {
            Python.start(new AndroidPlatform(this));
        }

        // Find buttons
        Button uploadButton = findViewById(R.id.btn_upload);
        Button processButton = findViewById(R.id.btn_process);

        // Upload button click listener
        uploadButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Video.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(intent, REQUEST_CODE_PICK_VIDEO);
            }
        });

        // Process button click listener
        processButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (selectedVideoPath != null) {
                    processVideo(selectedVideoPath);
                } else {
                    Toast.makeText(MainActivity.this, "Please select a video first", Toast.LENGTH_SHORT).show();
                }
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == REQUEST_CODE_PICK_VIDEO && resultCode == RESULT_OK && data != null) {
            Uri selectedVideoUri = data.getData();
            selectedVideoPath = getRealPathFromURI(selectedVideoUri);

            if (selectedVideoPath != null && new File(selectedVideoPath).exists()) {
                Toast.makeText(this, "Video selected: " + selectedVideoPath, Toast.LENGTH_SHORT).show();
                Log.d("VideoPath", "Selected video path: " + selectedVideoPath);
            } else {
                Toast.makeText(this, "Invalid video file", Toast.LENGTH_SHORT).show();
                Log.e("VideoError", "File does not exist at path: " + selectedVideoPath);
            }
        }
    }


    private void processVideo(String videoPath) {
        try {
            // Call Python script using Chaquopy
            Python py = Python.getInstance();
            String processedVideoPath = py.getModule("video_processor").callAttr("run_video_processing", videoPath).toString();

            Toast.makeText(this, "Processing completed! Playing video...", Toast.LENGTH_SHORT).show();
            playProcessedVideo(processedVideoPath);
        } catch (Exception e) {
            Toast.makeText(this, "Error processing video: " + e.getMessage(), Toast.LENGTH_LONG).show();
            Log.e("VideoProcessing", "Error: ", e);
        }
    }

    private void playProcessedVideo(String videoPath) {
        Intent intent = new Intent(Intent.ACTION_VIEW);
        intent.setDataAndType(Uri.parse(videoPath), "video/mp4");
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        startActivity(intent);
    }

    // ✅ Converts Uri to absolute file path
    private String getRealPathFromURI(Uri uri) {
        String[] projection = {MediaStore.Video.Media.DATA};
        Cursor cursor = getContentResolver().query(uri, projection, null, null, null);
        if (cursor != null) {
            int columnIndex = cursor.getColumnIndexOrThrow(MediaStore.Video.Media.DATA);
            cursor.moveToFirst();
            String path = cursor.getString(columnIndex);
            cursor.close();
            return path;
        }
        return null;
    }



}
