import fs from "fs/promises";
import path from "path";

const targetArg = process.argv[2] || "public/images/proj1/results";
const dir = path.resolve(process.cwd(), targetArg);
const outFile = path.join(dir, "index.json");

async function main() {
  try {
    const entries = await fs.readdir(dir);
    const images = entries
      .filter((f) => /\.(jpe?g|png|webp|gif|avif|svg)$/i.test(f))
      .sort();
    await fs.writeFile(outFile, JSON.stringify(images, null, 2), "utf8");
    console.log(`Wrote ${outFile} (${images.length} items)`);
  } catch (err) {
    console.error("Error generating manifest:", err.message);
    process.exit(1);
  }
}

main();