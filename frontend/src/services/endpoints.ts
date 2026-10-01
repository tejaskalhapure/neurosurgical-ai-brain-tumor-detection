/**
 * Standardized backend API endpoint definitions.
 */

import { apiFetch } from './apiClient';
import type { SystemHealthResponse } from '../types/api';
import type { MRIUploadResult, InferenceResult } from '../types/mri';

export const apiService = {
  checkHealth: () => apiFetch<SystemHealthResponse>('/health'),

  uploadMri: async (file: File): Promise<MRIUploadResult> => {
    const formData = new FormData();
    formData.append('file', file);
    return apiFetch<MRIUploadResult>('/mri/upload', {
      method: 'POST',
      body: formData,
    });
  },

  getInferenceStatus: () => apiFetch<{ device: string; model_initialized: boolean }>('/inference/status'),

  triggerInference: (scanId: string): Promise<InferenceResult> => {
    return apiFetch<InferenceResult>(`/inference/predict/${scanId}`, {
      method: 'POST',
    });
  },

  getReportSummary: (scanId: string) => apiFetch<{ report_id: string; download_url: string }>(`/reports/${scanId}/summary`),
};
