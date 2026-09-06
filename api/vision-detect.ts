import { processVisionDetect } from "../src/server/geminiVisionService";

export const config = {
  api: {
    bodyParser: {
      sizeLimit: "4mb",
    },
  },
};

export default async function handler(req: any, res: any) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  try {
    const result = await processVisionDetect(req.body);
    if (!result.success && result.error?.includes("Missing image data")) {
      return res.status(400).json(result);
    }
    return res.status(200).json(result);
  } catch (error: any) {
    console.error("[Vercel Vision Detect Error]:", error);
    return res.status(200).json({
      success: false,
      error: error?.message || "Failed to analyze image with forensic vision AI",
    });
  }
}
