import { handleVisionDetect } from "../server";

export const config = {
  api: {
    bodyParser: {
      sizeLimit: "10mb",
    },
  },
};

export default async function handler(req: any, res: any) {
  return handleVisionDetect(req, res);
}
