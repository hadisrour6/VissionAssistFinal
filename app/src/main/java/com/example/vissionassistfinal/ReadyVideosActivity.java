package com.example.vissionassistfinal;

import android.content.Intent;
import android.content.res.AssetManager;
import android.net.Uri;
import android.os.Bundle;
import android.widget.Button;
import android.widget.ListView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class ReadyVideosActivity extends AppCompatActivity {

    private static final String ASSET_SUBFOLDER = "processed_videos";

    private ListView listView;
    private Button backButton;
    private VideoListAdapter adapter;
    private final List<String> assetVideoNames = new ArrayList<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.ready_videos_activity);

        listView = findViewById(R.id.videoListView);
        backButton = findViewById(R.id.backButton);

        // 1) Load any .mp4 files from assets/processed_videos/
        loadVideoFilesFromAssets();

        // 2) Create the VideoListAdapter with the list of asset filenames
        adapter = new VideoListAdapter(this, R.layout.video_list_item, assetVideoNames);
        listView.setAdapter(adapter);

        // 3) On item click => play the video
        listView.setOnItemClickListener((parent, view, position, id) -> {
            String assetName = assetVideoNames.get(position);
            playVideoFromAssets(assetName);
        });

        // 4) Back button
        backButton.setOnClickListener(v -> finish());
    }

    /**
     * Scans assets/processed_videos/ for .mp4 files
     */
    private void loadVideoFilesFromAssets() {
        assetVideoNames.clear();
        AssetManager assetManager = getAssets();
        try {
            String[] filesInFolder = assetManager.list(ASSET_SUBFOLDER);
            if (filesInFolder != null) {
                for (String fileName : filesInFolder) {
                    if (fileName.toLowerCase().endsWith(".mp4")) {
                        assetVideoNames.add(fileName);
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        if (assetVideoNames.isEmpty()) {
            Toast.makeText(this, "No .mp4 files found in assets/" + ASSET_SUBFOLDER, Toast.LENGTH_SHORT).show();
        }
    }

    /**
     * Copy from assets -> cacheDir, then play via FileProvider
     */
    private void playVideoFromAssets(String assetName) {
        File outFile = new File(getCacheDir(), assetName);
        if (!outFile.exists()) {
            try {
                copyAssetToCache(assetName, outFile);
            } catch (IOException e) {
                e.printStackTrace();
                Toast.makeText(this, "Failed to copy: " + assetName, Toast.LENGTH_SHORT).show();
                return;
            }
        }

        // Build a content Uri with FileProvider
        Uri contentUri = FileProvider.getUriForFile(
                this,
                getPackageName() + ".fileprovider",
                outFile
        );

        // Launch external player
        Intent intent = new Intent(Intent.ACTION_VIEW);
        intent.setDataAndType(contentUri, "video/mp4");
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        startActivity(intent);
    }

    /**
     * Copies assets/processed_videos/<assetName> -> cache/<assetName>
     */
    private void copyAssetToCache(String assetName, File outFile) throws IOException {
        AssetManager assetManager = getAssets();
        String fullPath = ASSET_SUBFOLDER + "/" + assetName;

        try (InputStream in = assetManager.open(fullPath);
             FileOutputStream out = new FileOutputStream(outFile)) {
            byte[] buffer = new byte[8192];
            int bytesRead;
            while ((bytesRead = in.read(buffer)) != -1) {
                out.write(buffer, 0, bytesRead);
            }
        }
    }
}
