import { processPathologySynthesis } from "../src/server/geminiVisionService";

export const config = {
  api: {
    bodyParser: {
      sizeLimit: "10mb",
    },
  },
};

export default async function handler(req: any, res: any) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const result = await processPathologySynthesis(req.body);
  if (!result.success) {
    return res.status(500).json(result);
  }
  return res.status(200).json(result);
}
