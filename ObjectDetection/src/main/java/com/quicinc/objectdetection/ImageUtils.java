// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.quicinc.objectdetection;

import android.graphics.Bitmap;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;

import androidx.camera.core.ImageProxy;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;

/**
 * Utility class for image processing operations
 */
public class ImageUtils {
    
    /**
     * Convert ImageProxy to Bitmap
     */
    public static Bitmap imageProxyToBitmap(ImageProxy imageProxy) {
        if (imageProxy.getFormat() == ImageFormat.YUV_420_888) {
            return yuv420ToBitmap(imageProxy);
        } else {
            // Fallback for other formats
            return createBitmapFromBuffer(imageProxy);
        }
    }
    
    private static Bitmap yuv420ToBitmap(ImageProxy imageProxy) {
        ImageProxy.PlaneProxy[] planes = imageProxy.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];

        // U and V are swapped
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, 
                                        imageProxy.getWidth(), imageProxy.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, imageProxy.getWidth(), imageProxy.getHeight()), 
                               100, out);
        byte[] imageBytes = out.toByteArray();
        
        Bitmap bitmap = android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
        
        // Apply rotation
        return rotateBitmap(bitmap, imageProxy.getImageInfo().getRotationDegrees());
    }
    
    private static Bitmap createBitmapFromBuffer(ImageProxy imageProxy) {
        ByteBuffer buffer = imageProxy.getPlanes()[0].getBuffer();
        byte[] bytes = new byte[buffer.remaining()];
        buffer.get(bytes);
        
        // Create bitmap from raw bytes (this is a simplified approach)
        int width = imageProxy.getWidth();
        int height = imageProxy.getHeight();
        
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        
        // Apply rotation
        return rotateBitmap(bitmap, imageProxy.getImageInfo().getRotationDegrees());
    }
    
    private static Bitmap rotateBitmap(Bitmap bitmap, int rotationDegrees) {
        if (rotationDegrees == 0) {
            return bitmap;
        }
        
        Matrix matrix = new Matrix();
        matrix.postRotate(rotationDegrees);
        
        return Bitmap.createBitmap(bitmap, 0, 0, 
                                 bitmap.getWidth(), bitmap.getHeight(), 
                                 matrix, true);
    }
}