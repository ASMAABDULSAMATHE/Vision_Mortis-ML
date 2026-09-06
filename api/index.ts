import app from "../server";

export const config = {
  api: {
    bodyParser: {
      sizeLimit: "50mb",
    },
  },
};

export default function handler(req: any, res: any) {
  const matchedPath = (req.headers && (req.headers["x-matched-path"] || req.headers["x-invoke-path"])) as string | undefined;
  if (matchedPath && req.url && (req.url === "/api/index" || req.url === "/index" || req.url.startsWith("/api/index?"))) {
    req.url = matchedPath;
  }
  return app(req, res);
}
