import { describe, expect, it } from "bun:test";

const STATIC_IMPORT_FIXTURE = `${import.meta.dir}/fixtures/oauth-barrel-import.ts`;

describe("OAuth barrel imports", () => {
	// This is a cold subprocess import of the package root. Under the package's
	// parallel=8 CI load it can exceed Bun's 5s unit-test default.
	it("loads with the Anthropic provider and auth storage while preserving public exports", async () => {
		const child = Bun.spawn([process.execPath, STATIC_IMPORT_FIXTURE], {
			cwd: import.meta.dir,
			stdout: "pipe",
			stderr: "pipe",
		});
		try {
			const [exitCode, stderr] = await Promise.all([
				child.exited,
				new Response(child.stderr).text(),
				new Response(child.stdout).text(),
			]);

			expect(exitCode, stderr).toBe(0);
		} finally {
			if (child.exitCode === null) {
				child.kill("SIGKILL");
				await child.exited;
			}
		}
	}, 60_000);
});
