export default function handler(req: any, res: any) {
  res.setHeader("Cache-Control", "no-store, no-cache, must-revalidate");
  return res.status(200).json({
    status: "ok",
    hasApiKey: !!process.env.GEMINI_API_KEY,
    appName: "Vision Mortis Protocol One",
    timestamp: new Date().toISOString(),
  });
}
