import { useEffect, useState } from "react";
import { getHealth, getModelInfo } from "@/lib/api-client";

export function useApiStatus() {
  const [health, setHealth] = useState<{ status: string; model_loaded: boolean } | null>(null);
  const [modelInfo, setModelInfo] = useState<{ model_name: string; input_size: number; number_of_classes: number; framework: string; version: string } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let ignore = false;

    async function load() {
      try {
        const [healthResponse, modelResponse] = await Promise.all([getHealth(), getModelInfo()]);
        if (!ignore) {
          setHealth(healthResponse);
          setModelInfo(modelResponse);
          setError(null);
        }
      } catch (err) {
        if (!ignore) {
          setError(err instanceof Error ? err.message : "Unable to reach backend");
        }
      } finally {
        if (!ignore) {
          setLoading(false);
        }
      }
    }

    load();
    return () => {
      ignore = true;
    };
  }, []);

  return { health, modelInfo, loading, error };
}
