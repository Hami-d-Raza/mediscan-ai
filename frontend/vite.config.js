import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      "/analyze-image":      { target: "http://localhost:8000", changeOrigin: true },
      "/analyze-brain-mri":  { target: "http://localhost:8000", changeOrigin: true },
      "/analyze-report":     { target: "http://localhost:8000", changeOrigin: true },
      "/health":             { target: "http://localhost:8000", changeOrigin: true },
      "/auth":               { target: "http://localhost:8000", changeOrigin: true },
    },
  },
});
