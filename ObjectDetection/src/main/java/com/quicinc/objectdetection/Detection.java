// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.quicinc.objectdetection;

import android.graphics.RectF;

/**
 * Represents a single object detection result
 */
public class Detection {
    private final RectF boundingBox;
    private final String label;
    private final float confidence;
    private final int classId;

    public Detection(RectF boundingBox, String label, float confidence, int classId) {
        this.boundingBox = boundingBox;
        this.label = label;
        this.confidence = confidence;
        this.classId = classId;
    }

    public RectF getBoundingBox() {
        return boundingBox;
    }

    public String getLabel() {
        return label;
    }

    public float getConfidence() {
        return confidence;
    }

    public int getClassId() {
        return classId;
    }

    @Override
    public String toString() {
        return String.format("%s (%.2f%%)", label, confidence * 100);
    }
}