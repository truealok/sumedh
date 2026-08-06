const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8001";

interface ApiErrorShape {
  detail?: string;
  message?: string;
}

class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      ...(init?.headers ?? {}),
    },
    ...init,
  });

  const contentType = response.headers.get("content-type") ?? "";
  const payload = contentType.includes("application/json") ? await response.json() : await response.text();

  if (!response.ok) {
    const message = typeof payload === "string"
      ? payload
      : (payload as ApiErrorShape)?.detail ?? (payload as ApiErrorShape)?.message ?? "Request failed";
    throw new ApiError(message, response.status);
  }

  return payload as T;
}

export async function getHealth() {
  return request<{ status: string; model_loaded: boolean }>('/health');
}

export async function getModelInfo() {
  return request<{ model_name: string; input_size: number; number_of_classes: number; framework: string; version: string }>('/model-info');
}

export async function getClasses() {
  return request<{ classes: string[] }>('/classes');
}

export async function predictImage(file: File) {
  const formData = new FormData();
  formData.append("file", file);

  return request<{ status: string; disease: string; confidence: number; message: string }>('/predict', {
    method: "POST",
    body: formData,
  });
}

export { ApiError };
