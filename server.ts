import express from "express";
import path from "path";
import dotenv from "dotenv";
import { processVisionDetect, processPathologySynthesis } from "./src/server/geminiVisionService";

dotenv.config();

const app = express();
const PORT = 3000;

// Body parser with serverless runtime compatibility
app.use((req, res, next) => {
  if (req.body && typeof req.body === "object" && Object.keys(req.body).length > 0) {
    return next();
  }
  return express.json({ limit: "50mb" })(req, res, next);
});

// Health check endpoint
app.get(["/api/health", "/health", "/api", "/api/index", "/index"], (req, res) => {
  res.json({
    status: "ok",
    hasApiKey: !!process.env.GEMINI_API_KEY,
    appName: "Vision Mortis Protocol One",
    timestamp: new Date().toISOString(),
  });
});

// XGBoost + TreeSHAP 208-Feature ML Model Proxy Endpoint
app.post(["/api/ml-predict", "/ml-predict"], async (req, res) => {
  try {
    const { modelUrl, caseData } = req.body;
    const targetUrl = modelUrl || "https://few-parents-return.loca.lt/predict";

    console.log(`[ML Proxy] Forwarding 208-feature inference to: ${targetUrl}`);

    const response = await fetch(targetUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Bypass-Tunnel-Reminder": "true", // Required for localtunnel to bypass interstitial page
      },
      body: JSON.stringify(caseData),
    });

    if (!response.ok) {
      const errText = await response.text();
      return res.status(response.status).json({
        success: false,
        error: `Python ML Server responded with HTTP ${response.status}: ${errText}`,
      });
    }

    const data = await response.json();
    return res.json({
      success: true,
      data,
    });
  } catch (error: any) {
    console.error("[ML Proxy Error]:", error);
    return res.status(500).json({
      success: false,
      error: error.message || "Failed to reach Python XGBoost model endpoint",
    });
  }
});

// AI Multimodal Forensic Analysis & Pathology Synthesis Handler
export async function handlePathologySynthesis(req: express.Request, res: express.Response) {
  try {
    const result = await processPathologySynthesis(req.body);
    return res.status(result.success ? 200 : 500).json(result);
  } catch (error: any) {
    console.error("AI Analysis error:", error);
    return res.status(500).json({
      success: false,
      error: error.message || "Failed to synthesize forensic pathology",
    });
  }
}

// Map endpoints with aliases to ensure full compatibility with and without /api prefix
app.post(["/api/ai-analyze", "/ai-analyze"], handlePathologySynthesis);
app.post(["/api/synthesize-pathology", "/synthesize-pathology", "/api/synthesize-report", "/synthesize-report"], handlePathologySynthesis);

// Computer Vision Image Analysis for Forensic Indicators
export async function handleVisionDetect(req: express.Request, res: express.Response) {
  try {
    const result = await processVisionDetect(req.body);
    if (!result.success && result.error?.includes("Missing image data")) {
      return res.status(400).json(result);
    }
    return res.status(result.success ? 200 : 500).json(result);
  } catch (error: any) {
    console.error("Vision detection error:", error);
    return res.status(500).json({
      success: false,
      error: error.message || "Failed to analyze image with forensic vision AI",
    });
  }
}

app.post(["/api/vision-detect", "/vision-detect"], handleVisionDetect);

// Vite middleware setup
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const { createServer: createViteServer } = await import("vite");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Vision Mortis server running on http://0.0.0.0:${PORT}`);
  });
}

// Only launch standalone HTTP server when executed directly as main script and not in a serverless environment
const isMainScript = Boolean(
  process.argv[1] &&
  (process.argv[1].endsWith("server.ts") ||
   process.argv[1].endsWith("server.cjs") ||
   process.argv[1].endsWith("server.js"))
);

if (isMainScript && !process.env.VERCEL && !process.env.AWS_LAMBDA_FUNCTION_NAME) {
  startServer();
}

export default app;
