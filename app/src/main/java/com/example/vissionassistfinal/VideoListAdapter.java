package com.example.vissionassistfinal;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.media.MediaMetadataRetriever;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.core.content.ContextCompat;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;

public class VideoListAdapter extends ArrayAdapter<String> {

    private final int resourceLayout;
    private final Context mContext;
    // Folder in assets where your videos reside
    private static final String ASSET_SUBFOLDER = "processed_videos";

    public VideoListAdapter(Context context, int resource, List<String> items) {
        super(context, resource, items);
        this.resourceLayout = resource;
        this.mContext = context;
    }

    @NonNull
    @Override
    public View getView(int position, View convertView, @NonNull ViewGroup parent) {
        View view = convertView;
        if (view == null) {
            LayoutInflater inflater = LayoutInflater.from(mContext);
            view = inflater.inflate(resourceLayout, parent, false);
        }

        String assetName = getItem(position); // e.g. "university_crosswalk.mp4"
        if (assetName != null) {
            ImageView imageView = view.findViewById(R.id.videoThumbnail);
            TextView textView = view.findViewById(R.id.videoTitle);

            // Display the filename
            textView.setText(assetName);

            // Generate or retrieve thumbnail
            Bitmap thumbnail = null;
            try {
                thumbnail = getVideoThumbnail(assetName);
            } catch (IOException e) {
                e.printStackTrace();
            }

            if (thumbnail != null) {
                imageView.setImageBitmap(thumbnail);
            } else {
                // fallback icon
                imageView.setImageDrawable(ContextCompat.getDrawable(mContext, android.R.drawable.ic_media_play));
            }
        }

        return view;
    }

    /**
     * Copies the asset to cache (if needed), retrieves a frame using MediaMetadataRetriever.
     */
    private Bitmap getVideoThumbnail(String assetName) throws IOException {
        File cacheFile = new File(mContext.getCacheDir(), assetName);
        if (!cacheFile.exists()) {
            copyAssetToCache(assetName, cacheFile);
        }

        MediaMetadataRetriever retriever = new MediaMetadataRetriever();
        try {
            retriever.setDataSource(cacheFile.getAbsolutePath());
            // Grab a frame at ~1 second into the video (in microseconds)
            return retriever.getFrameAtTime(1_000_000);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        } finally {
            retriever.release();
        }
    }

    /**
     * Copies e.g. assets/processed_videos/university_crosswalk.mp4 -> cacheFile
     */
    private void copyAssetToCache(String assetName, File outFile) throws IOException {
        AssetManager assetManager = mContext.getAssets();
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
