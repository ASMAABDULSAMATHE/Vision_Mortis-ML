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

  try {
    const { modelUrl, caseData } = req.body || {};
    const targetUrl = modelUrl || "https://few-parents-return.loca.lt/predict";

    const response = await fetch(targetUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Bypass-Tunnel-Reminder": "true",
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
    return res.status(200).json({
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
}
