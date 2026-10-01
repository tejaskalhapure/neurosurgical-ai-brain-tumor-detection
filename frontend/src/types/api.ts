/**
 * Backend API response contracts and error models.
 */

export interface SystemHealthResponse {
  status: 'healthy' | 'degraded' | 'error';
  version: string;
  device: 'cpu' | 'cuda';
  model_loaded: boolean;
  disclaimer: string;
}

export interface ApiResponse<T> {
  data: T;
  message?: string;
  timestamp?: string;
}

export interface ApiError {
  detail: string;
  status_code: number;
}
