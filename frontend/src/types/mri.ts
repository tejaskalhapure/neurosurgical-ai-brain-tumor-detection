/**
 * MRI Volumetric and Tumor Segmentation Type Definitions.
 */

export interface BoundingBox3D {
  x_min: number;
  x_max: number;
  y_min: number;
  y_max: number;
  z_min: number;
  z_max: number;
}

export interface VolumetricMetrics {
  estimated_volume_mm3: number;
  estimated_volume_cm3: number;
  voxel_count: number;
  max_slice_area_mm2?: number;
  max_slice_index?: number;
  valid_voxel_metadata: boolean;
}

export interface MRIUploadResult {
  scan_id: string;
  filename: string;
  file_size_bytes: number;
  status: string;
  format_detected: string;
}

export interface InferenceResult {
  scan_id: string;
  tumor_detected: boolean;
  detection_confidence: number;
  bounding_box_3d?: BoundingBox3D;
  volumetrics?: VolumetricMetrics;
  slices_with_tumor: number[];
  status: string;
  disclaimer: string;
}
