package com.example.vissionassistfinal;

import static androidx.core.content.ContextCompat.startActivity;

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
        Button howToUseButton = findViewById(R.id.howToUseButton);
        Button StartButton = findViewById(R.id.startButton);
        Button ReadyVideoButton = findViewById(R.id.ReadyVideos);



        howToUseButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Launch HowToUseActivity
                Intent intent = new Intent(MainActivity.this, HowToUseActivity.class);
                startActivity(intent);
            }
        });

       ReadyVideoButton.setOnClickListener(new View.OnClickListener() {
           @Override
           public void onClick(View v) {
               // Launch ReadyVideosActivity
               Intent intent = new Intent(MainActivity.this, ReadyVideosActivity.class);
               startActivity(intent);
           }
       });

        StartButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Launch ReadyVideosActivity
                Intent intent = new Intent(MainActivity.this, CameraFeedActivity.class);
                startActivity(intent);
            }
        });

    }
}
