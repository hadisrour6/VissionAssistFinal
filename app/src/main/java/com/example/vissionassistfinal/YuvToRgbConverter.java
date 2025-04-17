package com.example.vissionassistfinal;

import android.graphics.Bitmap;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import androidx.camera.core.ImageProxy;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;

/**
 * Converts ImageProxy (YUV) data to an ARGB Bitmap.
 */
public class YuvToRgbConverter {

    public YuvToRgbConverter() {
        // no-op
    }

    public Bitmap toBitmap(ImageProxy imageProxy) {
        ByteBuffer yBuffer = imageProxy.getPlanes()[0].getBuffer(); // Y
        ByteBuffer uBuffer = imageProxy.getPlanes()[1].getBuffer(); // U
        ByteBuffer vBuffer = imageProxy.getPlanes()[2].getBuffer(); // V

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        // Combine YUV data into a single NV21 buffer
        byte[] nv21 = new byte[ySize + uSize + vSize];

        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        int width = imageProxy.getWidth();
        int height = imageProxy.getHeight();

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, width, height, null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, width, height), 100, out);
        byte[] imageBytes = out.toByteArray();

        return android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }
}
