// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.quicinc.objectdetection;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import java.util.List;

/**
 * Custom view for drawing bounding boxes and labels over camera preview
 */
public class OverlayView extends View {
    private List<Detection> detections;
    private Paint boxPaint;
    private Paint textPaint;
    private Paint backgroundPaint;
    
    // Colors for different classes
    private final int[] colors = {
        Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW, Color.CYAN,
        Color.MAGENTA, Color.WHITE, Color.GRAY, Color.LTGRAY, Color.DKGRAY
    };

    public OverlayView(Context context) {
        super(context);
        init();
    }

    public OverlayView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    public OverlayView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        init();
    }

    private void init() {
        boxPaint = new Paint();
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(4f);
        boxPaint.setAntiAlias(true);

        textPaint = new Paint();
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(40f);
        textPaint.setAntiAlias(true);
        textPaint.setTextAlign(Paint.Align.LEFT);

        backgroundPaint = new Paint();
        backgroundPaint.setStyle(Paint.Style.FILL);
        backgroundPaint.setAntiAlias(true);
    }

    public void setDetections(List<Detection> detections) {
        this.detections = detections;
        invalidate(); // Trigger a redraw
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        
        if (detections == null || detections.isEmpty()) {
            return;
        }

        for (Detection detection : detections) {
            RectF box = detection.getBoundingBox();
            
            // Choose color based on class ID
            int color = colors[detection.getClassId() % colors.length];
            boxPaint.setColor(color);
            backgroundPaint.setColor(color);
            backgroundPaint.setAlpha(128); // Semi-transparent background

            // Draw bounding box
            canvas.drawRect(box, boxPaint);

            // Draw label background
            String label = detection.toString();
            float textWidth = textPaint.measureText(label);
            float textHeight = textPaint.getTextSize();
            
            RectF textBackground = new RectF(
                box.left,
                box.top - textHeight - 8,
                box.left + textWidth + 16,
                box.top
            );
            
            canvas.drawRect(textBackground, backgroundPaint);
            
            // Draw label text
            canvas.drawText(label, box.left + 8, box.top - 8, textPaint);
        }
    }
}