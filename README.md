# Final Project: Exploring Object Detection Capabilities from UAV Aerial Perspectives with Deep Learning Tools

## Introduction

Unmanned Aerial Vehicles (UAVs) are widely used in applications like traffic monitoring, surveillance, and disaster relief. This project aims to develop a YOLO-based object detection system optimized for UAV imagery, leveraging the **VisDrone Dataset** and integrating real-time detection and tracking capabilities.

---

## Workflow Overview

1. **Dataset Preparation**:
   - Used the **VisDrone Dataset** (10,209 images and 261,908 video frames).
   - Configured a custom dataset YAML for training and validation.

2. **Model Selection and Comparison**:
   - Compared YOLOv11n (lightweight) and YOLOv11x (high-capacity) models.

3. **Hyperparameter Optimization**:
   - Fine-tuned parameters like `epochs`, `batch size`, and `image resolution`.

4. **Real-Time Detection**:
   - Implemented real-time detection with object tracking and directional guidance.

5. **Evaluation**:
   - Validated with metrics like `val_loss`, `mAP50`, and confusion matrices.

---

## YOLO Model Comparison

| **Aspect**               | **YOLOv11n**                                                                 | **YOLOv11x**                                                                  |
|--------------------------|-----------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| **Model Scale**          | Small (~4M parameters), fast training.                                      | Large (~86M parameters), higher accuracy.                                   |
| **Depth and Width**      | Fewer layers, limited feature extraction.                                   | Deeper and wider feature maps.                                              |
| **Performance**          | Overfitting at 150 epochs; **box loss ≈ 1.3**, **cls loss ≈ 0.9** (Figure 1). | Stable at 50 epochs; **box loss ≈ 1.05**, **cls loss ≈ 0.7** (Figure 2).    |

**Figures**:
- **Figure 1**: Loss trends for YOLOv11n ([images/figure1.png](images/figure1.png)).
- **Figure 2**: Loss trends for YOLOv11x ([images/figure2.png](images/figure2.png)).

---

## Real-Time Detection and Tracking

### 1. **Basic Real-Time Detection**
- **Libraries Used**: `ultralytics` (YOLO), `OpenCV`.
- **Principle**:
  - Frame-by-frame detection with bounding boxes and class labels.
  - Fast and efficient for real-time applications.
- **Sample Output**:
  - **Figure 3**: Real-time detection ([images/figure3.png](images/figure3.png)).

### 2. **Object Tracking**
- **Enhancements**:
  - Used YOLO's tracking capability to assign persistent IDs.
  - Visualized object trajectories with historical coordinates.
- **Benefits**:
  - Useful for monitoring patterns and movements.
  - Lightweight implementation.
- **Sample Output**:
  - **Figure 4**: Object tracking with trajectories ([images/figure4.png](images/figure4.png)).

### 3. **Object Locking and Directional Guidance**
- **Features**:
  - Mouse interaction for object selection.
  - Directional guidance (e.g., "Left," "Up") based on relative position to the frame center.
- **Future Enhancements**:
  - Incorporate **DeepSORT** for better ID consistency.
- **Sample Output**:
  - **Figure 5**: Object locking and directional guidance ([images/figure5.png](images/figure5.png)).

---

## Evaluation

### Validation Metrics
- **YOLOv11n**: Higher losses, overfitting at 150 epochs.
- **YOLOv11x**: Stable convergence, better mAP scores.

### Real-Time Testing
- Tested with toy cars simulating UAV scenarios.
- Consistent detection of objects like `car`, `van`, and `truck`.

**Sample Results**:
- **Figure 6**: Confusion matrix for YOLOv11x ([images/figure6.png](images/figure6.png)).

---

## Future Work

1. **Accuracy Enhancements**:
   - Explore advanced feature extraction methods (e.g., 1D CNNs).
   - Address dataset imbalance for underrepresented classes like `truck`.

2. **UAV Deployment**:
   - Integrate the system with UAV hardware for real-world testing.

3. **Advanced Tracking**:
   - Incorporate **DeepSORT** for robust multi-object tracking.

---

## Conclusion

This project successfully developed a YOLO-based detection system capable of real-time object detection and tracking. YOLOv11x demonstrated superior performance, providing a solid foundation for future UAV applications.

---

## References
- **VisDrone Dataset**: [VisDrone Official Site](https://github.com/VisDrone/VisDrone-Dataset)
- **Ultralytics YOLO**: [Ultralytics Official Site](https://github.com/ultralytics/ultralytics)
