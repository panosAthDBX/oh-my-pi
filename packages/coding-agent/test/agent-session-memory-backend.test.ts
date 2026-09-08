import { afterEach, beforeEach, describe, expect, it, vi } from "bun:test";
import * as path from "node:path";
import { type } from "@oh-my-pi/omptype";
import { Agent, type AgentTool } from "@oh-my-pi/pi-agent-core";
import { createMockModel } from "@oh-my-pi/pi-ai/providers/mock";
import { buildModel } from "@oh-my-pi/pi-catalog/build";
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import * as memoryBackend from "@oh-my-pi/pi-coding-agent/memory-backend/resolve";
import type { MemoryBackend } from "@oh-my-pi/pi-coding-agent/memory-backend/types";
import { getMnemopiSessionState } from "@oh-my-pi/pi-coding-agent/mnemopi/state";
import { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import type { AuthStorage } from "@oh-my-pi/pi-coding-agent/session/auth-storage";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { resetMemoryForTests } from "@oh-my-pi/pi-mnemopi";
import { TempDir } from "@oh-my-pi/pi-utils";
import { createInMemoryAuthStorage } from "./helpers/agent-session-setup";

function createTool(name: string): AgentTool {
	return {
		name,
		label: name,
		description: `${name} memory tool`,
		parameters: type({}),
		async execute() {
			return { content: [{ type: "text", text: name }] };
		},
	};
}

describe("AgentSession memory backend lifecycle", () => {
	let authStorage: AuthStorage;
	let session: AgentSession | undefined;
	let settings: Settings;
	let tempDir: TempDir;

	beforeEach(() => {
		tempDir = TempDir.createSync("@memory-backend-lifecycle-");
		authStorage = createInMemoryAuthStorage();
		authStorage.setRuntimeApiKey("anthropic", "test-key");
		authStorage.setRuntimeApiKey("openai", "test-key");
		settings = Settings.isolated({
			"compaction.enabled": false,
			"memory.backend": "off",
			"mnemopi.noEmbeddings": true,
			"mnemopi.llmMode": "none",
		});
	});

	afterEach(async () => {
		await session?.dispose();
		session = undefined;
		resetMemoryForTests();
		vi.restoreAllMocks();
		authStorage.close();
		tempDir.removeSync();
	});

	function createSession(createMemoryTools: () => Promise<AgentTool[]>): AgentSession {
		const model = buildModel({
			id: "mock",
			name: "mock",
			api: "openai-responses",
			provider: "openai",
			baseUrl: "https://example.invalid",
			reasoning: false,
			input: ["text"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 8192,
			maxTokens: 2048,
		});
		const read = createTool("read");
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: { model, systemPrompt: ["initial"], tools: [read] },
			streamFn: createMockModel({ responses: [{ content: ["ok"] }] }).stream,
		});
		const toolRegistry = new Map<string, AgentTool>([[read.name, read]]);
		session = new AgentSession({
			agent,
			sessionManager: SessionManager.create(tempDir.path(), tempDir.join("sessions")),
			settings,
			modelRegistry: new ModelRegistry(authStorage, path.join(tempDir.path(), "models.yml")),
			memoryAgentDir: tempDir.path(),
			memoryTaskDepth: 0,
			createMemoryTools,
			toolRegistry,
			builtInToolNames: [read.name],
			rebuildSystemPrompt: async toolNames => ({
				systemPrompt: [`backend:${settings.get("memory.backend")};tools:${toolNames.sort().join(",")}`],
			}),
		});
		return session;
	}

	// Backend activation opens SQLite-backed state and rebuilds the prompt twice;
	// the complete lifecycle can exceed Bun's 5s default under concurrent CI chunks.
	it("switches runtime state, memory tools, and prompt in one apply", async () => {
		const current = createSession(async () =>
			settings.get("memory.backend") === "mnemopi" ? [createTool("retain"), createTool("memory_edit")] : [],
		);

		settings.override("memory.backend", "mnemopi");
		await current.applyMemoryBackend();

		expect(getMnemopiSessionState(current)).toBeDefined();
		expect(current.getActiveToolNames()).toEqual(expect.arrayContaining(["read", "retain", "memory_edit"]));
		expect(current.systemPrompt).toEqual(["backend:mnemopi;tools:memory_edit,read,retain"]);

		settings.override("memory.backend", "off");
		await current.applyMemoryBackend();

		expect(getMnemopiSessionState(current)).toBeUndefined();
		expect(current.getActiveToolNames()).toEqual(["read"]);
		expect(current.getAllToolNames()).toEqual(["read"]);
		expect(current.systemPrompt).toEqual(["backend:off;tools:read"]);
	});
	it("applies a deferred backend change when a new session starts", async () => {
		const current = createSession(async () =>
			settings.get("memory.backend") === "mnemopi" ? [createTool("retain"), createTool("memory_edit")] : [],
		);
		settings.override("memory.backend", "mnemopi");
		await current.applyMemoryBackend();
		expect(getMnemopiSessionState(current)).toBeDefined();

		settings.override("memory.backend", "off");
		await current.newSession();

		expect(current.getMemoryBackend()?.id).toBe("off");
		expect(getMnemopiSessionState(current)).toBeUndefined();
		expect(current.getActiveToolNames()).toEqual(["read"]);
	});

	it("applies a deferred backend change when a fork starts", async () => {
		const current = createSession(async () =>
			settings.get("memory.backend") === "mnemopi" ? [createTool("retain"), createTool("memory_edit")] : [],
		);
		settings.override("memory.backend", "mnemopi");
		await current.applyMemoryBackend();
		await current.sendUserMessage("first");

		settings.override("memory.backend", "off");
		expect(await current.fork()).toBe(true);

		expect(current.getMemoryBackend()?.id).toBe("off");
		expect(getMnemopiSessionState(current)).toBeUndefined();
		expect(current.getActiveToolNames()).toEqual(["read"]);
	});

	it("applies a deferred backend change when another session resumes", async () => {
		const current = createSession(async () =>
			settings.get("memory.backend") === "mnemopi" ? [createTool("retain"), createTool("memory_edit")] : [],
		);
		settings.override("memory.backend", "mnemopi");
		await current.applyMemoryBackend();
		await current.sendUserMessage("first");
		const firstSessionFile = current.sessionManager.getSessionFile();
		if (!firstSessionFile) throw new Error("Expected a persisted session file");
		await current.newSession();

		settings.override("memory.backend", "off");
		await current.switchSession(firstSessionFile);

		expect(current.getMemoryBackend()?.id).toBe("off");
		expect(getMnemopiSessionState(current)).toBeUndefined();
		expect(current.getActiveToolNames()).toEqual(["read"]);
	});

	async function expectBackendLifecycleOrdering(
		prepareTransition: (current: AgentSession) => Promise<() => Promise<unknown>>,
	): Promise<void> {
		const events: string[] = [];
		const previousBackend: MemoryBackend = {
			id: "mnemopi",
			async start({ session: activeSession }) {
				events.push(`start:${activeSession.sessionId}`);
			},
			async disposeSession(activeSession) {
				events.push(`dispose:${activeSession.sessionId}`);
			},
			async buildDeveloperInstructions() {
				return undefined;
			},
			async clear() {},
			async enqueue() {},
		};
		const selectedBackend: MemoryBackend = {
			id: "off",
			async start({ session: activeSession }) {
				events.push(`start:${activeSession.sessionId}`);
			},
			async buildDeveloperInstructions() {
				return undefined;
			},
			async clear() {},
			async enqueue() {},
		};
		vi.spyOn(memoryBackend, "resolveMemoryBackend").mockImplementation(async currentSettings =>
			currentSettings.get("memory.backend") === "mnemopi" ? previousBackend : selectedBackend,
		);
		const current = createSession(async () => []);
		settings.override("memory.backend", "mnemopi");
		await current.applyMemoryBackend();
		const runTransition = await prepareTransition(current);
		const previousSessionId = current.sessionId;
		events.length = 0;

		settings.override("memory.backend", "off");
		await runTransition();

		expect(current.sessionId).not.toBe(previousSessionId);
		expect(events).toEqual([`dispose:${previousSessionId}`, `start:${current.sessionId}`]);
	}

	it("disposes a deferred backend before replacing a new-session transcript", async () => {
		await expectBackendLifecycleOrdering(async current => async () => current.newSession());
	});

	it("disposes a deferred backend before replacing a fork transcript", async () => {
		await expectBackendLifecycleOrdering(async current => {
			await current.sendUserMessage("persist before fork");
			return async () => current.fork();
		});
	});

	it("disposes a deferred backend before replacing a resumed transcript", async () => {
		await expectBackendLifecycleOrdering(async current => {
			const targetManager = SessionManager.create(tempDir.path(), tempDir.join("target-sessions"));
			await targetManager.flush();
			const targetSessionFile = targetManager.getSessionFile();
			await targetManager.close();
			if (!targetSessionFile) throw new Error("Expected a target session file");
			return async () => current.switchSession(targetSessionFile);
		});
	});

	it("restores the active backend when a new-session transition fails", async () => {
		const current = createSession(async () =>
			current.getMemoryBackend()?.id === "mnemopi" ? [createTool("retain"), createTool("memory_edit")] : [],
		);
		settings.override("memory.backend", "mnemopi");
		await current.applyMemoryBackend();
		const previousSessionId = current.sessionId;
		vi.spyOn(current.sessionManager, "newSession").mockRejectedValue(new Error("session transition failed"));

		settings.override("memory.backend", "off");
		await expect(current.newSession()).rejects.toThrow("session transition failed");

		expect(current.sessionId).toBe(previousSessionId);
		expect(current.getMemoryBackend()?.id).toBe("mnemopi");
		expect(getMnemopiSessionState(current)).toBeDefined();
		expect(current.getActiveToolNames()).toEqual(expect.arrayContaining(["read", "retain", "memory_edit"]));
	});

	it("flushes the active backend before an unchanged-backend new session replaces its transcript", async () => {
		const events: string[] = [];
		const backend: MemoryBackend = {
			id: "mnemopi",
			async start() {},
			async beforeTranscriptReplace(activeSession) {
				events.push(`flush:${activeSession.sessionId}`);
			},
			async buildDeveloperInstructions() {
				return undefined;
			},
			async clear() {},
			async enqueue() {},
		};
		vi.spyOn(memoryBackend, "resolveMemoryBackend").mockResolvedValue(backend);
		const current = createSession(async () => []);
		settings.override("memory.backend", "mnemopi");
		await current.applyMemoryBackend();
		const previousSessionId = current.sessionId;

		await current.newSession();

		expect(current.sessionId).not.toBe(previousSessionId);
		expect(events).toEqual([`flush:${previousSessionId}`]);
	});

	it("flushes and resets transcript-scoped memory when navigating the session tree", async () => {
		const events: string[] = [];
		const backend: MemoryBackend = {
			id: "mnemopi",
			async start() {},
			async beforeTranscriptReplace(activeSession) {
				events.push(`flush:${activeSession.sessionId}`);
			},
			resetSession(activeSession) {
				events.push(`reset:${activeSession.sessionId}`);
				return true;
			},
			async buildDeveloperInstructions() {
				return undefined;
			},
			async clear() {},
			async enqueue() {},
		};
		vi.spyOn(memoryBackend, "resolveMemoryBackend").mockResolvedValue(backend);
		const current = createSession(async () => []);
		settings.override("memory.backend", "mnemopi");
		await current.applyMemoryBackend();
		await current.sendUserMessage("first branch turn");
		await current.sendUserMessage("second branch turn");
		const firstAssistant = current.sessionManager
			.getBranch()
			.find(entry => entry.type === "message" && entry.message.role === "assistant");
		if (!firstAssistant) throw new Error("Expected an assistant entry to navigate to");
		events.length = 0;

		const result = await current.navigateTree(firstAssistant.id);

		expect(result.cancelled).toBe(false);
		expect(events).toEqual([`flush:${current.sessionId}`, `reset:${current.sessionId}`]);
	});

	it("restores agent event delivery when transcript preparation fails", async () => {
		const backend: MemoryBackend = {
			id: "mnemopi",
			async start() {},
			async beforeTranscriptReplace() {
				throw new Error("retention failed");
			},
			async buildDeveloperInstructions() {
				return undefined;
			},
			async clear() {},
			async enqueue() {},
		};
		vi.spyOn(memoryBackend, "resolveMemoryBackend").mockResolvedValue(backend);
		const current = createSession(async () => []);
		settings.override("memory.backend", "mnemopi");
		await current.applyMemoryBackend();
		const previousSessionId = current.sessionId;
		const eventTypes: string[] = [];
		current.subscribe(event => eventTypes.push(event.type));

		await expect(current.newSession()).rejects.toThrow("retention failed");
		expect(current.sessionId).toBe(previousSessionId);
		await current.sendUserMessage("still connected");

		expect(eventTypes).toContain("agent_end");
	});

	it("reconciles the current mode when transcript preparation aborts a session switch", async () => {
		const backend: MemoryBackend = {
			id: "mnemopi",
			async start() {},
			async beforeTranscriptReplace() {
				throw new Error("retention failed");
			},
			async buildDeveloperInstructions() {
				return undefined;
			},
			async clear() {},
			async enqueue() {},
		};
		vi.spyOn(memoryBackend, "resolveMemoryBackend").mockResolvedValue(backend);
		const current = createSession(async () => []);
		settings.override("memory.backend", "mnemopi");
		await current.applyMemoryBackend();
		const previousSessionId = current.sessionId;
		const targetManager = SessionManager.create(tempDir.path(), tempDir.join("failed-switch-target"));
		await targetManager.flush();
		const targetSessionFile = targetManager.getSessionFile();
		await targetManager.close();
		if (!targetSessionFile) throw new Error("Expected a target session file");
		const events: string[] = [];
		current.setSessionBeforeSwitchReconciler(async () => {
			events.push("quiesce");
		});
		current.setSessionSwitchReconciler(async () => {
			events.push("reconcile");
		});

		await expect(current.switchSession(targetSessionFile)).rejects.toThrow("retention failed");

		expect(current.sessionId).toBe(previousSessionId);
		expect(events).toEqual(["quiesce", "reconcile"]);
	}, 15_000);

	it("cancels a displaced local startup generation", async () => {
		const current = createSession(async () => []);
		const localStartup = current.beginLocalMemoryStartup();

		await current.applyMemoryBackend();

		expect(localStartup.aborted).toBe(true);
	});

	it("serializes concurrent backend applies", async () => {
		const firstStarted = Promise.withResolvers<void>();
		const releaseFirst = Promise.withResolvers<void>();
		let calls = 0;
		let running = 0;
		let maxRunning = 0;
		const current = createSession(async () => {
			calls++;
			running++;
			maxRunning = Math.max(maxRunning, running);
			if (calls === 1) {
				firstStarted.resolve();
				await releaseFirst.promise;
			}
			running--;
			return [];
		});

		const first = current.applyMemoryBackend();
		await firstStarted.promise;
		const second = current.applyMemoryBackend();
		await Promise.resolve();
		expect(calls).toBe(1);
		releaseFirst.resolve();
		await Promise.all([first, second]);

		expect(maxRunning).toBe(1);
		expect(calls).toBe(2);
	});
});
