// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.quicinc.objectdetection;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.util.Pair;

import com.quicinc.ImageProcessing;
import com.quicinc.tflite.AIHubDefaults;
import com.quicinc.tflite.TFLiteHelpers;

import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Object Detection using YOLOv8 model
 */
public class ObjectDetection {
    private static final String TAG = "ObjectDetection";

    // Model configuration
    private static final int INPUT_SIZE = 640;
    private static final float CONFIDENCE_THRESHOLD = 0.1f; // 进一步降低置信度阈值用于调试
    private static final float IOU_THRESHOLD = 0.45f;
    private static final int MAX_DETECTIONS = 100;
    private static final int paddingValue = 0;
    private static final boolean USE_ASPECT_RATIO_CORRECTION = false; // 设为false使用简单缩放

    private Interpreter interpreter;
    private Map<TFLiteHelpers.DelegateType, Delegate> delegateStore;
    private List<String> labels;
    private ByteBuffer inputBuffer;
    private float[][][] outputBuffer; // [1][8400][84] for YOLOv8

    // Model input/output dimensions
    private int[] inputShape;
    private int[] outputShape;

    /**
     * Create an Object Detector from the given model.
     * Uses default compute units: NPU, GPU, CPU.
     * Ignores compute units that fail to load.
     *
     * @param context    App context.
     * @param modelPath  Model path to load.
     * @param labelsPath Labels path to load.
     * @throws IOException If the model can't be read from disk.
     */
    public ObjectDetection(Context context, String modelPath, String labelsPath)
            throws IOException, NoSuchAlgorithmException {
        this(context, modelPath, labelsPath, AIHubDefaults.delegatePriorityOrder);
    }

    /**
     * Create an Object Detector from the given model.
     * Ignores compute units that fail to load.
     *
     * @param context     App context.
     * @param modelPath   Model path to load.
     * @param labelsPath  Labels path to load.
     * @param delegatePriorityOrder Priority order of delegate sets to enable.
     * @throws IOException If the model can't be read from disk.
     */
    public ObjectDetection(Context context, String modelPath, String labelsPath,
                           TFLiteHelpers.DelegateType[][] delegatePriorityOrder)
            throws IOException, NoSuchAlgorithmException {
        // Load TF Lite model
        Pair<MappedByteBuffer, String> modelAndHash = TFLiteHelpers.loadModelFile(context.getAssets(), modelPath);
        Pair<Interpreter, Map<TFLiteHelpers.DelegateType, Delegate>> iResult = TFLiteHelpers.CreateInterpreterAndDelegatesFromOptions(
                modelAndHash.first,
                delegatePriorityOrder,
                AIHubDefaults.numCPUThreads,
                context.getApplicationInfo().nativeLibraryDir,
                context.getCacheDir().getAbsolutePath(),
                modelAndHash.second
        );
        interpreter = iResult.first;
        delegateStore = iResult.second;

        initializeModel(context, labelsPath);
    }

    /**
     * Create an Object Detector from the given model with custom Interpreter.Options.
     * This constructor is kept for backward compatibility.
     *
     * @param context    App context.
     * @param modelPath  Model path to load.
     * @param labelsPath Labels path to load.
     * @param options    Custom Interpreter.Options.
     * @throws IOException If the model can't be read from disk.
     */
    public ObjectDetection(Context context, String modelPath, String labelsPath,
                           Interpreter.Options options) throws IOException {
        // Load model
        ByteBuffer modelBuffer = FileUtil.loadMappedFile(context, modelPath);
        interpreter = new Interpreter(modelBuffer, options);
        delegateStore = null; // No delegate store when using custom options

        initializeModel(context, labelsPath);
    }

    private void initializeModel(Context context, String labelsPath) throws IOException {
        // Get input/output tensor info
        inputShape = interpreter.getInputTensor(0).shape();
        outputShape = interpreter.getOutputTensor(0).shape();

        // Validate model dimensions
        assert inputShape.length == 4 : "Input should be 4D: [batch, height, width, channels]";
        assert inputShape[1] == INPUT_SIZE && inputShape[2] == INPUT_SIZE :
                "Input size should be " + INPUT_SIZE + "x" + INPUT_SIZE;
        assert inputShape[3] == 3 : "Input should have 3 channels (RGB)";

        assert outputShape.length == 3 : "Output should be 3D: [batch, detections, features]";
        assert outputShape[2] >= 84 : "Output should have at least 84 features (4 bbox + 80 classes)";

        // Initialize buffers
        int inputSize = inputShape[1] * inputShape[2] * inputShape[3] * 4; // float32
        inputBuffer = ByteBuffer.allocateDirect(inputSize);
        inputBuffer.order(ByteOrder.nativeOrder());

        outputBuffer = new float[outputShape[0]][outputShape[1]][outputShape[2]];

        // Load labels with error handling
        try {
            labels = FileUtil.loadLabels(context, labelsPath);
            android.util.Log.d(TAG, "Loaded " + labels.size() + " labels from " + labelsPath);
            // 打印前几个标签用于调试
            for (int i = 0; i < Math.min(5, labels.size()); i++) {
                android.util.Log.d(TAG, "Label " + i + ": " + labels.get(i));
            }
        } catch (IOException e) {
            android.util.Log.e(TAG, "Failed to load labels from " + labelsPath, e);
            // 创建默认标签列表
            labels = new ArrayList<>();
            for (int i = 0; i < 80; i++) {
                labels.add("class_" + i);
            }
        }
    }

    public List<Detection> detect(Bitmap bitmap) {
        // Preprocess image
        Bitmap resizedBitmap = ImageProcessing.resizeAndPadMaintainAspectRatio(bitmap, INPUT_SIZE, INPUT_SIZE, paddingValue);
        fillInputBuffer(resizedBitmap);

        // Run inference
        long startTime = System.currentTimeMillis();
        interpreter.run(inputBuffer, outputBuffer);
        long inferenceTime = System.currentTimeMillis() - startTime;

        // Post-process results with scaling info
        List<Detection> detections = postProcess(outputBuffer[0],
                bitmap.getWidth(),
                bitmap.getHeight());

        android.util.Log.d(TAG, "Detected " + detections.size() + " objects in " + inferenceTime + "ms");
        return detections;
    }

    private void fillInputBuffer(Bitmap bitmap) {
        inputBuffer.rewind();

        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

        // Convert to normalized float values [0, 1]
        for (int pixel : pixels) {
            inputBuffer.putFloat(((pixel >> 16) & 0xFF) / 255.0f); // R
            inputBuffer.putFloat(((pixel >> 8) & 0xFF) / 255.0f);  // G
            inputBuffer.putFloat((pixel & 0xFF) / 255.0f);         // B
        }
    }

    private List<Detection> postProcess(float[][] output, int originalWidth, int originalHeight) {
        List<Detection> allDetections = new ArrayList<>();

        int finalWidth, finalHeight, offsetX, offsetY;
        float scaleX, scaleY;

        if (USE_ASPECT_RATIO_CORRECTION) {
            // Calculate scaling parameters for coordinate transformation
            // This matches the logic in ImageProcessing.resizeAndPadMaintainAspectRatio
            float ratioBitmap = (float) originalWidth / (float) originalHeight;
            float ratioMax = (float) INPUT_SIZE / (float) INPUT_SIZE; // 1.0 for square input

            finalWidth = INPUT_SIZE;
            finalHeight = INPUT_SIZE;

            if (ratioMax > ratioBitmap) {
                // ratioBitmap < 1.0, image is taller than wide
                finalWidth = (int) ((float) INPUT_SIZE * ratioBitmap);
                offsetX = (INPUT_SIZE - finalWidth) / 2;
                offsetY = 0;
            } else {
                // ratioBitmap >= 1.0, image is wider than tall or square
                finalHeight = (int) ((float) INPUT_SIZE / ratioBitmap);
                offsetX = 0;
                offsetY = (INPUT_SIZE - finalHeight) / 2;
            }

            scaleX = (float) originalWidth / (float) finalWidth;
            scaleY = (float) originalHeight / (float) finalHeight;
        } else {
            // Simple scaling without aspect ratio correction (original method)
            finalWidth = INPUT_SIZE;
            finalHeight = INPUT_SIZE;
            offsetX = 0;
            offsetY = 0;
            scaleX = (float) originalWidth / (float) INPUT_SIZE;
            scaleY = (float) originalHeight / (float) INPUT_SIZE;
        }

        android.util.Log.d(TAG, "PostProcess: mode=" + (USE_ASPECT_RATIO_CORRECTION ? "aspect_ratio" : "simple") +
                ", original=" + originalWidth + "x" + originalHeight +
                ", final=" + finalWidth + "x" + finalHeight +
                ", offset=(" + offsetX + "," + offsetY + ")" +
                ", scale=(" + scaleX + "," + scaleY + ")");

        int totalDetections = 0;
        int confidenceFiltered = 0;
        int boundsFiltered = 0;
        int invalidFiltered = 0;

        // Parse detections from output
        for (int i = 0; i < output.length; i++) {
            float[] detection = output[i];
            totalDetections++;

            // Extract bounding box (center_x, center_y, width, height)
            float centerX = detection[0];
            float centerY = detection[1];
            float width = detection[2];
            float height = detection[3];

            // Find class with highest confidence
            float maxConfidence = 0;
            int bestClass = -1;

            for (int j = 4; j < detection.length && j < 4 + labels.size(); j++) {
                if (detection[j] > maxConfidence) {
                    maxConfidence = detection[j];
                    bestClass = j - 4;
                }
            }

            // Filter by confidence threshold
            if (maxConfidence < CONFIDENCE_THRESHOLD) {
                confidenceFiltered++;
                continue;
            }

            float left, top, right, bottom;

            if (USE_ASPECT_RATIO_CORRECTION) {
                // Convert normalized coordinates to INPUT_SIZE pixel coordinates
                float centerXPixels = centerX * INPUT_SIZE;
                float centerYPixels = centerY * INPUT_SIZE;
                float widthPixels = width * INPUT_SIZE;
                float heightPixels = height * INPUT_SIZE;

                // Convert to corner coordinates in INPUT_SIZE space
                float x1 = centerXPixels - widthPixels / 2;
                float y1 = centerYPixels - heightPixels / 2;
                float x2 = centerXPixels + widthPixels / 2;
                float y2 = centerYPixels + heightPixels / 2;

                // Remove padding offset
                x1 -= offsetX;
                y1 -= offsetY;
                x2 -= offsetX;
                y2 -= offsetY;

                // Debug: log first few detections before scaling
                if (allDetections.size() < 3) {
                    android.util.Log.d(TAG, "Detection " + (allDetections.size() + 1) + ": " +
                            "raw center=(" + centerX + "," + centerY + ") size=(" + width + "," + height + ")");
                    android.util.Log.d(TAG, "  pixels center=(" + centerXPixels + "," + centerYPixels + ") size=(" + widthPixels + "," + heightPixels + ")");
                    android.util.Log.d(TAG, "  after offset removal: [" + x1 + "," + y1 + "," + x2 + "," + y2 + "]");
                    android.util.Log.d(TAG, "  offset=(" + offsetX + "," + offsetY + ") scale=(" + scaleX + "," + scaleY + ")");
                    android.util.Log.d(TAG, "  original size=" + originalWidth + "x" + originalHeight + ", final size=" + finalWidth + "x" + finalHeight);
                }

                // Check if coordinates are still valid after offset removal
                // Only skip if the box is completely outside the valid area or has invalid dimensions
                if (x2 <= x1 || y2 <= y1) {
                    if (allDetections.size() < 3) {
                        android.util.Log.w(TAG, "  Skipping detection: invalid box dimensions");
                    }
                    boundsFiltered++;
                    continue;
                }

                // Skip if box is completely outside the valid area
                if (x2 <= 0 || y2 <= 0 || x1 >= finalWidth || y1 >= finalHeight) {
                    if (allDetections.size() < 3) {
                        android.util.Log.w(TAG, "  Skipping detection: completely outside valid area");
                    }
                    boundsFiltered++;
                    continue;
                }

                // Scale to original image size
                left = x1 * scaleX;
                top = y1 * scaleY;
                right = x2 * scaleX;
                bottom = y2 * scaleY;
            } else {
                // Simple scaling (original method)
                // First convert normalized coordinates to INPUT_SIZE coordinates, then scale to original size
                float centerXPixels = centerX * INPUT_SIZE;
                float centerYPixels = centerY * INPUT_SIZE;
                float widthPixels = width * INPUT_SIZE;
                float heightPixels = height * INPUT_SIZE;

                left = (centerXPixels - widthPixels / 2) * scaleX;
                top = (centerYPixels - heightPixels / 2) * scaleY;
                right = (centerXPixels + widthPixels / 2) * scaleX;
                bottom = (centerYPixels + heightPixels / 2) * scaleY;

                // Debug: log first few detections
                if (allDetections.size() < 3) {
                    android.util.Log.d(TAG, "Detection " + (allDetections.size() + 1) + ": " +
                            "raw center=(" + centerX + "," + centerY + ") size=(" + width + "," + height + ")");
                    android.util.Log.d(TAG, "  pixels center=(" + centerXPixels + "," + centerYPixels + ") size=(" + widthPixels + "," + heightPixels + ")");
                    android.util.Log.d(TAG, "  simple scaled bbox=[" + left + "," + top + "," + right + "," + bottom + "]");
                    android.util.Log.d(TAG, "  scale=(" + scaleX + "," + scaleY + ")");
                }
            }

            // Debug: log final coordinates
            if (allDetections.size() < 3) {
                android.util.Log.d(TAG, "  final bbox=[" + left + "," + top + "," + right + "," + bottom + "]");
            }

            // Clamp to image bounds
            left = Math.max(0, Math.min(left, originalWidth));
            top = Math.max(0, Math.min(top, originalHeight));
            right = Math.max(0, Math.min(right, originalWidth));
            bottom = Math.max(0, Math.min(bottom, originalHeight));

            // Skip invalid boxes
            if (right <= left || bottom <= top) {
                if (allDetections.size() < 3) {
                    android.util.Log.w(TAG, "  Skipping detection: invalid box after clamping");
                }
                invalidFiltered++;
                continue;
            }

            RectF boundingBox = new RectF(left, top, right, bottom);
            String label = (bestClass >= 0 && bestClass < labels.size()) ? labels.get(bestClass) : "Unknown";

            allDetections.add(new Detection(boundingBox, label, maxConfidence, bestClass));
        }

        android.util.Log.d(TAG, "Detection filtering: total=" + totalDetections +
                ", confidence_filtered=" + confidenceFiltered +
                ", bounds_filtered=" + boundsFiltered +
                ", invalid_filtered=" + invalidFiltered +
                ", remaining=" + allDetections.size());

        // Log some statistics about the raw detections for debugging
        if (totalDetections > 0) {
            float maxConf = 0;
            float minConf = 1;
            int validConfCount = 0;

            for (int i = 0; i < Math.min(output.length, 100); i++) {
                float[] detection = output[i];

                // Find max confidence for this detection
                float detectionMaxConf = 0;
                for (int j = 4; j < detection.length && j < 4 + labels.size(); j++) {
                    detectionMaxConf = Math.max(detectionMaxConf, detection[j]);
                }

                if (detectionMaxConf > 0.01f) { // Only count non-trivial confidences
                    maxConf = Math.max(maxConf, detectionMaxConf);
                    minConf = Math.min(minConf, detectionMaxConf);
                    validConfCount++;
                }
            }

            android.util.Log.d(TAG, "Confidence stats: max=" + maxConf + ", min=" + minConf +
                    ", valid_count=" + validConfCount + ", threshold=" + CONFIDENCE_THRESHOLD);
        }

        // Apply Non-Maximum Suppression
        return applyNMS(allDetections);
    }

    private List<Detection> applyNMS(List<Detection> detections) {
        // Sort by confidence (descending)
        detections.sort((a, b) -> Float.compare(b.getConfidence(), a.getConfidence()));

        List<Detection> filteredDetections = new ArrayList<>();
        boolean[] suppressed = new boolean[detections.size()];

        for (int i = 0; i < detections.size() && filteredDetections.size() < MAX_DETECTIONS; i++) {
            if (suppressed[i]) continue;

            Detection detection = detections.get(i);
            filteredDetections.add(detection);

            // Suppress overlapping detections
            for (int j = i + 1; j < detections.size(); j++) {
                if (suppressed[j]) continue;

                Detection other = detections.get(j);
                if (detection.getClassId() == other.getClassId()) {
                    float iou = calculateIoU(detection.getBoundingBox(), other.getBoundingBox());
                    if (iou > IOU_THRESHOLD) {
                        suppressed[j] = true;
                    }
                }
            }
        }

        return filteredDetections;
    }

    private float calculateIoU(RectF box1, RectF box2) {
        float intersectionLeft = Math.max(box1.left, box2.left);
        float intersectionTop = Math.max(box1.top, box2.top);
        float intersectionRight = Math.min(box1.right, box2.right);
        float intersectionBottom = Math.min(box1.bottom, box2.bottom);

        if (intersectionLeft >= intersectionRight || intersectionTop >= intersectionBottom) {
            return 0.0f;
        }

        float intersectionArea = (intersectionRight - intersectionLeft) *
                (intersectionBottom - intersectionTop);

        float box1Area = (box1.right - box1.left) * (box1.bottom - box1.top);
        float box2Area = (box2.right - box2.left) * (box2.bottom - box2.top);

        float unionArea = box1Area + box2Area - intersectionArea;

        return intersectionArea / unionArea;
    }

    public void close() {
        if (interpreter != null) {
            interpreter.close();
            interpreter = null;
        }
    }
}