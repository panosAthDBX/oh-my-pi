import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { resetSettingsForTest, Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { createMemoryRuntimeContext, resolveMemoryBackend } from "@oh-my-pi/pi-coding-agent/memory-backend";
import { removeWithRetries } from "@oh-my-pi/pi-utils";

describe("resolveMemoryBackend", () => {
	let tempDir: string;
	let agentDir: string;
	beforeEach(async () => {
		resetSettingsForTest();
		tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "omp-memory-resolve-"));
		agentDir = path.join(tempDir, "agent");
		await fs.mkdir(agentDir, { recursive: true });
	});

	afterEach(async () => {
		resetSettingsForTest();
		await removeWithRetries(tempDir);
	});

	it("returns the hindsight backend when memory.backend is hindsight, regardless of legacy memories.enabled", async () => {
		const a = Settings.isolated({ "memory.backend": "hindsight", "memories.enabled": false });
		const b = Settings.isolated({ "memory.backend": "hindsight", "memories.enabled": true });
		expect((await resolveMemoryBackend(a)).id).toBe("hindsight");
		expect((await resolveMemoryBackend(b)).id).toBe("hindsight");
	});

	it("returns the Supermemory backend only when memory.backend explicitly selects it", async () => {
		const selected = Settings.isolated({ "memory.backend": "supermemory" });
		const defaultSettings = Settings.isolated();
		expect((await resolveMemoryBackend(selected)).id).toBe("supermemory");
		expect((await resolveMemoryBackend(defaultSettings)).id).toBe("off");
	});

	it("exposes inactive status when no session is available", async () => {
		const memory = createMemoryRuntimeContext({ agentDir, cwd: "/tmp/project" });

		await expect(memory.status()).resolves.toMatchObject({
			backend: "off",
			active: false,
			writable: false,
			searchable: false,
		});
	});

	it("resolves status, search, and save from settings when the runtime context has no captured backend method", async () => {
		const settings = Settings.isolated({ "memory.backend": "local" });
		const memory = createMemoryRuntimeContext({
			agentDir,
			cwd: "/tmp/project",
			session: { settings } as never,
		});

		await expect(memory.status()).resolves.toMatchObject({
			backend: "local",
			active: true,
			writable: true,
			searchable: false,
		});
		await expect(memory.search("project preference")).resolves.toMatchObject({
			backend: "local",
			count: 0,
		});
		await expect(memory.save("project preference")).resolves.toMatchObject({
			backend: "local",
			stored: 1,
		});
	});
});
