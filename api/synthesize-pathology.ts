import { processPathologySynthesis } from "../src/server/geminiVisionService";

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
    const result = await processPathologySynthesis(req.body);
    return res.status(200).json(result);
  } catch (error: any) {
    console.error("[Vercel Synthesize Error]:", error);
    return res.status(200).json({
      success: false,
      error: error?.message || "Failed to synthesize forensic pathology",
    });
  }
}
