// Pilot currently keeps socket handles alive after stdin EOF. Own this client's lifetime.
import { spawn } from "node:child_process";
const target = process.argv[2];
if (!target) throw new Error("Pilot MCP script argument required");
const child = spawn(process.execPath, [target], { stdio: ["pipe", "pipe", "inherit"] });
process.stdin.pipe(child.stdin);
child.stdout.pipe(process.stdout);
child.stdin.on("error", () => {});
const stop = () => { child.stdin.end(); child.kill("SIGTERM"); };
process.stdin.on("end", stop);
process.on("SIGTERM", stop);
process.on("SIGINT", stop);
child.on("error", error => { process.stderr.write(String(error)); process.exit(1); });
child.on("exit", code => process.exit(code ?? 0));
