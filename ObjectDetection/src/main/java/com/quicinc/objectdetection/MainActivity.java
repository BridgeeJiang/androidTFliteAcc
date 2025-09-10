// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.quicinc.objectdetection;

import android.Manifest;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.util.Size;
import android.widget.Button;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.TextView;
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

import com.google.common.util.concurrent.ListenableFuture;
import com.quicinc.tflite.AIHubDefaults;

import org.tensorflow.lite.Interpreter;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "ObjectDetection";
    private static final int REQUEST_CODE_PERMISSIONS = 10;
    private static final String[] REQUIRED_PERMISSIONS = {Manifest.permission.CAMERA};

    // UI Elements
    private PreviewView previewView;
    private com.quicinc.objectdetection.OverlayView overlayView;
    private Button cameraControlButton;
    private RadioGroup delegateSelectionGroup;
    private RadioButton cpuOnlyRadio;
    private RadioButton defaultDelegateRadio;
    private TextView inferenceTimeText;
    private TextView detectionResultsText;

    // Camera and ML
    private ProcessCameraProvider cameraProvider;
    private ObjectDetection objectDetection;
    private ExecutorService cameraExecutor;
    private ExecutorService inferenceExecutor;
    private boolean isCameraRunning = false;

    // Performance tracking
    private final DecimalFormat decimalFormat = new DecimalFormat("0.0");
    private Handler mainHandler;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main_activity);

        initializeViews();
        setupClickListeners();

        mainHandler = new Handler(Looper.getMainLooper());
        cameraExecutor = Executors.newSingleThreadExecutor();
        inferenceExecutor = Executors.newSingleThreadExecutor();

        if (allPermissionsGranted()) {
            initializeModel();
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS);
        }
    }

    private void initializeViews() {
        previewView = findViewById(R.id.previewView);
        overlayView = findViewById(R.id.overlayView);
        cameraControlButton = findViewById(R.id.cameraControlButton);
        delegateSelectionGroup = findViewById(R.id.delegateSelectionGroup);
        cpuOnlyRadio = findViewById(R.id.cpuOnlyRadio);
        defaultDelegateRadio = findViewById(R.id.defaultDelegateRadio);
        inferenceTimeText = findViewById(R.id.inferenceTimeText);
        detectionResultsText = findViewById(R.id.detectionResultsText);
    }

    private void setupClickListeners() {
        cameraControlButton.setOnClickListener(v -> {
            if (isCameraRunning) {
                stopCamera();
            } else {
                startCamera();
            }
        });

        delegateSelectionGroup.setOnCheckedChangeListener((group, checkedId) -> {
            if (objectDetection != null) {
                reinitializeModel();
            }
        });
    }

    private boolean isModelFileValid(String modelPath) {
        try {
            AssetFileDescriptor fileDescriptor = getAssets().openFd(modelPath);
            long fileSize = fileDescriptor.getLength();
            fileDescriptor.close();
            return fileSize > 1000; // Model should be at least 1KB
        } catch (IOException e) {
            return false;
        }
    }

    private void initializeModel() {
        inferenceExecutor.execute(() -> {
            try {
                String modelPath = getString(R.string.tfLiteModelAsset);
                String labelsPath = getString(R.string.tfLiteLabelsAsset);

                // Check if model file is valid before attempting to load
                if (!isModelFileValid(modelPath)) {
                    throw new RuntimeException("Model file '" + modelPath + "' is empty or missing. Please check assets/README.txt for instructions on downloading the YOLOv8 model.");
                }

                if (cpuOnlyRadio.isChecked()) {
                    // CPU only - use empty delegate priority order
                    objectDetection = new ObjectDetection(this, modelPath, labelsPath,
                            AIHubDefaults.delegatePriorityOrderForDelegates(new java.util.HashSet<>()));
                } else {
                    // Use hardware acceleration with default delegate priority order
                    objectDetection = new ObjectDetection(this, modelPath, labelsPath,
                            AIHubDefaults.delegatePriorityOrder);
                }

                mainHandler.post(() -> {
                    cameraControlButton.setEnabled(true);
                    Toast.makeText(this, "Model loaded successfully", Toast.LENGTH_SHORT).show();
                });

            } catch (IOException | java.security.NoSuchAlgorithmException | RuntimeException e) {
                Log.e(TAG, "Error initializing model", e);
                mainHandler.post(() -> {
                    String errorMessage;
                    if (e.getMessage() != null && e.getMessage().contains("ByteBuffer is not a valid TensorFlow Lite model")) {
                        errorMessage = "Model file is invalid or missing. Please check the assets/README.txt for instructions on downloading the YOLOv8 model.";
                    } else if (e.getMessage() != null && e.getMessage().contains("Unable to create an interpreter")) {
                        errorMessage = "Failed to load model. The model file may be corrupted or incompatible.";
                    } else {
                        errorMessage = "Error loading model: " + e.getMessage();
                    }
                    Toast.makeText(this, errorMessage, Toast.LENGTH_LONG).show();
                });
            }
        });
    }

    private void reinitializeModel() {
        cameraControlButton.setEnabled(false);
        if (isCameraRunning) {
            stopCamera();
        }

        if (objectDetection != null) {
            objectDetection.close();
            objectDetection = null;
        }

        initializeModel();
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                cameraProvider = cameraProviderFuture.get();
                bindCameraUseCases();

                isCameraRunning = true;
                cameraControlButton.setText(R.string.stop_camera);

            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "Error starting camera", e);
                Toast.makeText(this, "Error starting camera", Toast.LENGTH_SHORT).show();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void bindCameraUseCases() {
        // Preview use case
        Preview preview = new Preview.Builder().build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        // Image analysis use case
        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                .setTargetResolution(new Size(640, 640))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();

        imageAnalysis.setAnalyzer(cameraExecutor, this::analyzeImage);

        // Camera selector
        CameraSelector cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA;

        try {
            // Unbind all use cases before rebinding
            cameraProvider.unbindAll();

            // Bind use cases to camera
            cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);

        } catch (Exception e) {
            Log.e(TAG, "Use case binding failed", e);
        }
    }

    private void analyzeImage(@NonNull ImageProxy imageProxy) {
        if (objectDetection == null) {
            imageProxy.close();
            return;
        }

        // Convert ImageProxy to Bitmap
        Bitmap bitmap = ImageUtils.imageProxyToBitmap(imageProxy);

        // Run inference
        long startTime = System.currentTimeMillis();
        List<Detection> detections = objectDetection.detect(bitmap);
        long inferenceTime = System.currentTimeMillis() - startTime;

        // Update UI on main thread
        mainHandler.post(() -> {
            updateUI(detections, inferenceTime);
            overlayView.setDetections(detections);
        });

        imageProxy.close();
    }



    private void updateUI(List<Detection> detections, long inferenceTime) {
        // Update inference time
        inferenceTimeText.setText(decimalFormat.format(inferenceTime) + " ms");

        // Update detection results
        if (detections.isEmpty()) {
            detectionResultsText.setText("No detections");
        } else {
            StringBuilder results = new StringBuilder();
            for (int i = 0; i < Math.min(detections.size(), 3); i++) {
                if (i > 0) results.append("\n");
                results.append(detections.get(i).toString());
            }
            if (detections.size() > 3) {
                results.append("\n... and ").append(detections.size() - 3).append(" more");
            }
            detectionResultsText.setText(results.toString());
        }
    }

    private void stopCamera() {
        if (cameraProvider != null) {
            cameraProvider.unbindAll();
        }

        isCameraRunning = false;
        cameraControlButton.setText(R.string.start_camera);
        overlayView.setDetections(null);
        detectionResultsText.setText("No detections");
        inferenceTimeText.setText("-- ms");
    }

    private boolean allPermissionsGranted() {
        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission)
                    != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                initializeModel();
            } else {
                Toast.makeText(this, "Permissions not granted by the user.",
                        Toast.LENGTH_SHORT).show();
                finish();
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        if (objectDetection != null) {
            objectDetection.close();
        }

        if (cameraExecutor != null) {
            cameraExecutor.shutdown();
        }

        if (inferenceExecutor != null) {
            inferenceExecutor.shutdown();
        }
    }
}