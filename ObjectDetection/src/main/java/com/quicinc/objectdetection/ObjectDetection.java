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
    private static final float CONFIDENCE_THRESHOLD = 0.5f;
    private static final float IOU_THRESHOLD = 0.45f;
    private static final int MAX_DETECTIONS = 100;
    private static final int paddingValue = 0;

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

        // Load labels
        labels = FileUtil.loadLabels(context, labelsPath);
    }

    public List<Detection> detect(Bitmap bitmap) {
        // Preprocess image
        Bitmap resizedBitmap = ImageProcessing.resizeAndPadMaintainAspectRatio(bitmap, INPUT_SIZE, INPUT_SIZE, paddingValue);
        fillInputBuffer(resizedBitmap);

        // Run inference
        long startTime = System.currentTimeMillis();
        interpreter.run(inputBuffer, outputBuffer);
        long inferenceTime = System.currentTimeMillis() - startTime;

        // Post-process results
        List<Detection> detections = postProcess(outputBuffer[0],
                bitmap.getWidth(),
                bitmap.getHeight());

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

        // Parse detections from output
        for (int i = 0; i < output.length; i++) {
            float[] detection = output[i];

            // Extract bounding box (center_x, center_y, width, height)
            float centerX = detection[0];
            float centerY = detection[1];
            float width = detection[2];
            float height = detection[3];

            // Find class with highest confidence
            float maxConfidence = 0;
            int bestClass = -1;

            for (int j = 4; j < detection.length; j++) {
                if (detection[j] > maxConfidence) {
                    maxConfidence = detection[j];
                    bestClass = j - 4;
                }
            }

            // Filter by confidence threshold
            if (maxConfidence < CONFIDENCE_THRESHOLD) {
                continue;
            }

            // Convert to corner coordinates and scale to original image size
            float left = (centerX - width / 2) * originalWidth / INPUT_SIZE;
            float top = (centerY - height / 2) * originalHeight / INPUT_SIZE;
            float right = (centerX + width / 2) * originalWidth / INPUT_SIZE;
            float bottom = (centerY + height / 2) * originalHeight / INPUT_SIZE;

            // Clamp to image bounds
            left = Math.max(0, Math.min(left, originalWidth));
            top = Math.max(0, Math.min(top, originalHeight));
            right = Math.max(0, Math.min(right, originalWidth));
            bottom = Math.max(0, Math.min(bottom, originalHeight));

            RectF boundingBox = new RectF(left, top, right, bottom);
            String label = bestClass < labels.size() ? labels.get(bestClass) : "Unknown";

            allDetections.add(new Detection(boundingBox, label, maxConfidence, bestClass));
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