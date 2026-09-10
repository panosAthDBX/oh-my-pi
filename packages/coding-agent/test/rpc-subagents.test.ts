import { afterEach, describe, expect, test } from "bun:test";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import type { ImageContent } from "@oh-my-pi/pi-ai";
import { RpcClient } from "@oh-my-pi/pi-coding-agent/modes/rpc/rpc-client";
import { RpcFrameDecoder } from "@oh-my-pi/pi-coding-agent/modes/rpc/rpc-frame";
import {
	handleRpcSessionChange,
	type RpcSessionChangeCommand,
	type RpcSessionChangeResult,
	type RpcSessionChangeSession,
} from "@oh-my-pi/pi-coding-agent/modes/rpc/rpc-mode";
import { RpcSubagentRegistry, readRpcSubagentTranscript } from "@oh-my-pi/pi-coding-agent/modes/rpc/rpc-subagents";
import type { RpcSubagentFrame } from "@oh-my-pi/pi-coding-agent/modes/rpc/rpc-types";
import type { AgentSessionEvent } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import {
	type AgentProgress,
	type SubagentEventPayload,
	type SubagentLifecyclePayload,
	type SubagentProgressPayload,
	TASK_SUBAGENT_EVENT_CHANNEL,
	TASK_SUBAGENT_LIFECYCLE_CHANNEL,
	TASK_SUBAGENT_PROGRESS_CHANNEL,
} from "@oh-my-pi/pi-coding-agent/task";
import { EventBus } from "@oh-my-pi/pi-coding-agent/utils/event-bus";
import { isRecord, ptree, readJsonl, removeSyncWithRetries, withTimeout } from "@oh-my-pi/pi-utils";

const tempPaths: string[] = [];

afterEach(() => {
	for (const tempPath of tempPaths.splice(0)) {
		removeSyncWithRetries(tempPath);
	}
});

function createProgress(overrides: Partial<AgentProgress> = {}): AgentProgress {
	return {
		index: 0,
		id: "SubagentA",
		agent: "task",
		agentSource: "bundled",
		status: "running",
		task: "Do work",
		assignment: "Implement work",
		description: "Worker",
		recentTools: [],
		recentOutput: [],
		toolCount: 0,
		requests: 0,
		tokens: 0,
		cost: 0,
		durationMs: 0,
		...overrides,
	};
}

function createRegistryWithSnapshot(): RpcSubagentRegistry {
	const eventBus = new EventBus();
	const registry = new RpcSubagentRegistry(eventBus, () => {});
	eventBus.emit(TASK_SUBAGENT_LIFECYCLE_CHANNEL, {
		id: "SubagentA",
		index: 0,
		agent: "task",
		agentSource: "bundled",
		status: "started",
		sessionFile: "/tmp/subagent.jsonl",
	} satisfies SubagentLifecyclePayload);
	expect(registry.getSubagents()).toHaveLength(1);
	return registry;
}

type SessionChangeStubOptions = {
	newSession?: boolean;
	switchSession?: boolean;
	branch?: { selectedText: string; selectedImages: ImageContent[]; cancelled: boolean };
};

function createSessionChangeSession(options: SessionChangeStubOptions): RpcSessionChangeSession {
	return {
		newSession: async (_options?: unknown) => options.newSession ?? true,
		switchSession: async (_sessionPath: string) => options.switchSession ?? true,
		branch: async (_entryId: string) =>
			options.branch ?? { selectedText: "branched text", selectedImages: [], cancelled: false },
	};
}

describe("RPC subagent registry", () => {
	test("defaults subagent frame emission to off while tracking snapshots", () => {
		const eventBus = new EventBus();
		const frames: RpcSubagentFrame[] = [];
		const registry = new RpcSubagentRegistry(eventBus, frame => frames.push(frame));
		const lifecycle: SubagentLifecyclePayload = {
			id: "SubagentA",
			index: 0,
			agent: "task",
			agentSource: "bundled",
			description: "Worker",
			status: "started",
			sessionFile: "/tmp/subagent.jsonl",
			parentToolCallId: "toolu_parent",
		};
		const progressPayload: SubagentProgressPayload = {
			index: 0,
			agent: "task",
			agentSource: "bundled",
			task: "Do work",
			assignment: "Implement work",
			parentToolCallId: "toolu_parent",
			sessionFile: "/tmp/subagent.jsonl",
			progress: createProgress(),
		};
		const eventPayload: SubagentEventPayload = {
			id: "SubagentA",
			event: { type: "agent_start" },
		};

		expect(registry.getSubscriptionLevel()).toBe("off");
		eventBus.emit(TASK_SUBAGENT_LIFECYCLE_CHANNEL, lifecycle);
		eventBus.emit(TASK_SUBAGENT_PROGRESS_CHANNEL, progressPayload);
		eventBus.emit(TASK_SUBAGENT_EVENT_CHANNEL, eventPayload);

		expect(frames).toHaveLength(0);
		expect(registry.getSubagents()).toMatchObject([
			{
				id: "SubagentA",
				status: "running",
				sessionFile: "/tmp/subagent.jsonl",
			},
		]);
		registry.dispose();
	});

	test("emits progress frames after explicit progress subscription and snapshots tracked subagents", () => {
		const eventBus = new EventBus();
		const frames: RpcSubagentFrame[] = [];
		const registry = new RpcSubagentRegistry(eventBus, frame => frames.push(frame));
		registry.setSubscriptionLevel("progress");
		const lifecycle: SubagentLifecyclePayload = {
			id: "SubagentA",
			index: 0,
			agent: "task",
			agentSource: "bundled",
			description: "Worker",
			status: "started",
			sessionFile: "/tmp/subagent.jsonl",
			parentToolCallId: "toolu_parent",
		};
		const progressPayload: SubagentProgressPayload = {
			index: 0,
			agent: "task",
			agentSource: "bundled",
			task: "Do work",
			assignment: "Implement work",
			parentToolCallId: "toolu_parent",
			sessionFile: "/tmp/subagent.jsonl",
			progress: createProgress(),
		};

		eventBus.emit(TASK_SUBAGENT_LIFECYCLE_CHANNEL, lifecycle);
		eventBus.emit(TASK_SUBAGENT_PROGRESS_CHANNEL, progressPayload);

		expect(frames.map(frame => frame.type)).toEqual(["subagent_lifecycle", "subagent_progress"]);
		expect(registry.getSubagents()).toMatchObject([
			{
				id: "SubagentA",
				status: "running",
				task: "Do work",
				assignment: "Implement work",
				sessionFile: "/tmp/subagent.jsonl",
				parentToolCallId: "toolu_parent",
			},
		]);

		registry.dispose();
	});

	test("clears stale snapshots when the active RPC session changes", () => {
		const eventBus = new EventBus();
		const registry = new RpcSubagentRegistry(eventBus, () => {});
		eventBus.emit(TASK_SUBAGENT_LIFECYCLE_CHANNEL, {
			id: "SubagentA",
			index: 0,
			agent: "task",
			agentSource: "bundled",
			status: "started",
			sessionFile: "/tmp/subagent.jsonl",
		} satisfies SubagentLifecyclePayload);

		expect(registry.getSubagents()).toHaveLength(1);
		registry.clear();

		expect(registry.getSubagents()).toHaveLength(0);
		registry.dispose();
	});

	test("clears stale snapshots after successful RPC session changes", async () => {
		const cases: Array<{
			command: RpcSessionChangeCommand;
			session: RpcSessionChangeSession;
			expected: RpcSessionChangeResult;
		}> = [
			{
				command: { type: "new_session", parentSession: "/tmp/parent.jsonl" },
				session: createSessionChangeSession({ newSession: true }),
				expected: { type: "new_session", data: { cancelled: false } },
			},
			{
				command: { type: "switch_session", sessionPath: "/tmp/next.jsonl" },
				session: createSessionChangeSession({ switchSession: true }),
				expected: { type: "switch_session", data: { cancelled: false } },
			},
			{
				command: { type: "branch", entryId: "entry-1" },
				session: createSessionChangeSession({
					branch: { selectedText: "Branch text", selectedImages: [], cancelled: false },
				}),
				expected: { type: "branch", data: { text: "Branch text", cancelled: false } },
			},
		];

		for (const testCase of cases) {
			const registry = createRegistryWithSnapshot();
			try {
				const result = await handleRpcSessionChange(testCase.session, testCase.command, registry);

				expect(result).toEqual(testCase.expected);
				expect(registry.getSubagents()).toHaveLength(0);
				expect(() => registry.resolveSessionFile({ subagentId: "SubagentA" })).toThrow(
					/Unknown subagent or session file unavailable/,
				);
			} finally {
				registry.dispose();
			}
		}
	});

	test("keeps stale snapshots when RPC session changes are cancelled", async () => {
		const cases: Array<{
			command: RpcSessionChangeCommand;
			session: RpcSessionChangeSession;
			expected: RpcSessionChangeResult;
		}> = [
			{
				command: { type: "new_session", parentSession: "/tmp/parent.jsonl" },
				session: createSessionChangeSession({ newSession: false }),
				expected: { type: "new_session", data: { cancelled: true } },
			},
			{
				command: { type: "switch_session", sessionPath: "/tmp/next.jsonl" },
				session: createSessionChangeSession({ switchSession: false }),
				expected: { type: "switch_session", data: { cancelled: true } },
			},
			{
				command: { type: "branch", entryId: "entry-1" },
				session: createSessionChangeSession({ branch: { selectedText: "", selectedImages: [], cancelled: true } }),
				expected: { type: "branch", data: { text: "", cancelled: true } },
			},
		];

		for (const testCase of cases) {
			const registry = createRegistryWithSnapshot();
			try {
				const result = await handleRpcSessionChange(testCase.session, testCase.command, registry);

				expect(result).toEqual(testCase.expected);
				expect(registry.getSubagents()).toMatchObject([{ id: "SubagentA" }]);
				expect(registry.resolveSessionFile({ subagentId: "SubagentA" })).toBe("/tmp/subagent.jsonl");
			} finally {
				registry.dispose();
			}
		}
	});

	test("prunes terminal lifecycle snapshots while retaining transcript selectors", () => {
		const eventBus = new EventBus();
		const registry = new RpcSubagentRegistry(eventBus, () => {});
		const sessionFile = "/tmp/subagent.jsonl";
		eventBus.emit(TASK_SUBAGENT_LIFECYCLE_CHANNEL, {
			id: "SubagentA",
			index: 0,
			agent: "task",
			agentSource: "bundled",
			status: "started",
			sessionFile,
		} satisfies SubagentLifecyclePayload);

		expect(registry.getSubagents()).toHaveLength(1);
		eventBus.emit(TASK_SUBAGENT_LIFECYCLE_CHANNEL, {
			id: "SubagentA",
			index: 0,
			agent: "task",
			agentSource: "bundled",
			status: "completed",
			sessionFile,
		} satisfies SubagentLifecyclePayload);

		expect(registry.getSubagents()).toHaveLength(0);
		expect(registry.resolveSessionFile({ subagentId: "SubagentA" })).toBe(sessionFile);
		expect(registry.resolveSessionFile({ sessionFile })).toBe(sessionFile);
		registry.dispose();
	});

	test("gates raw subagent events behind the events subscription level", () => {
		const eventBus = new EventBus();
		const frames: RpcSubagentFrame[] = [];
		const registry = new RpcSubagentRegistry(eventBus, frame => frames.push(frame));
		const eventPayload: SubagentEventPayload = {
			id: "SubagentA",
			event: { type: "agent_start" },
		};

		eventBus.emit(TASK_SUBAGENT_EVENT_CHANNEL, eventPayload);
		expect(frames).toHaveLength(0);

		registry.setSubscriptionLevel("events");
		eventBus.emit(TASK_SUBAGENT_EVENT_CHANNEL, eventPayload);

		expect(frames).toHaveLength(1);
		expect(frames[0]).toEqual({ type: "subagent_event", payload: eventPayload });
		registry.dispose();
	});
});

describe("readRpcSubagentTranscript", () => {
	test("returns complete JSONL entries and byte cursor", async () => {
		const dir = fs.mkdtempSync(path.join(os.tmpdir(), "omp-rpc-subagent-transcript-"));
		tempPaths.push(dir);
		const sessionFile = path.join(dir, "session.jsonl");
		const headerLine = `${JSON.stringify({ type: "session", id: "s1", timestamp: "2026-06-09T00:00:00.000Z", cwd: dir })}\n`;
		const messageLine = `${JSON.stringify({
			type: "message",
			id: "m1",
			parentId: null,
			timestamp: "2026-06-09T00:00:00.000Z",
			message: { role: "user", content: [{ type: "text", text: "hello" }] },
		})}\n`;
		await Bun.write(sessionFile, `${headerLine}${messageLine}{"type":"message"`);

		const result = await readRpcSubagentTranscript(sessionFile);

		expect(result.entries).toHaveLength(2);
		expect(result.messages).toHaveLength(1);
		expect(result.nextByte).toBe(Buffer.byteLength(`${headerLine}${messageLine}`, "utf8"));
		expect(result.reset).toBe(false);
	});

	test("returns empty cursor result for missing transcript files", async () => {
		const dir = fs.mkdtempSync(path.join(os.tmpdir(), "omp-rpc-subagent-transcript-missing-"));
		tempPaths.push(dir);
		const sessionFile = path.join(dir, "missing.jsonl");

		const result = await readRpcSubagentTranscript(sessionFile, 42);

		expect(result).toEqual({
			sessionFile,
			fromByte: 42,
			nextByte: 42,
			reset: false,
			entries: [],
			messages: [],
		});
	});
});

describe("RpcClient subagent frames", () => {
	test("dispatches subagent frames and session-specific events", async () => {
		const scriptPath = path.join(os.tmpdir(), `omp-rpc-subagent-client-${Date.now()}.js`);
		tempPaths.push(scriptPath);
		await Bun.write(
			scriptPath,
			`
let buffer = "";
function write(frame) {
	process.stdout.write(JSON.stringify(frame) + "\\n");
}
const progress = {
	index: 0,
	id: "SubagentA",
	agent: "task",
	agentSource: "bundled",
	status: "running",
	task: "Do work",
	assignment: "Implement work",
	recentTools: [],
	recentOutput: [],
	toolCount: 0,
	tokens: 0,
	cost: 0,
	durationMs: 0
};
write({ type: "ready" });
process.stdin.on("data", chunk => {
	buffer += chunk.toString("utf8");
	let index = buffer.indexOf("\\n");
	while (index !== -1) {
		const line = buffer.slice(0, index).trim();
		buffer = buffer.slice(index + 1);
		if (line) handle(JSON.parse(line));
		index = buffer.indexOf("\\n");
	}
});
function handle(frame) {
	if (frame.type === "set_subagent_subscription") {
		write({ id: frame.id, type: "response", command: "set_subagent_subscription", success: true, data: { level: frame.level } });
		return;
	}
	if (frame.type === "get_subagents") {
		write({ id: frame.id, type: "response", command: "get_subagents", success: true, data: { subagents: [{ id: "SubagentA", index: 0, agent: "task", agentSource: "bundled", status: "running", lastUpdate: 1 }] } });
		return;
	}
	if (frame.type === "get_subagent_messages") {
		write({ id: frame.id, type: "response", command: "get_subagent_messages", success: true, data: { sessionFile: frame.sessionFile || "/tmp/subagent.jsonl", fromByte: frame.fromByte || 0, nextByte: 0, reset: false, entries: [], messages: [] } });
		return;
	}
	if (frame.type === "prompt") {
		write({ id: frame.id, type: "response", command: "prompt", success: true });
		write({ type: "notice", level: "info", message: "subagent test" });
		write({ type: "todo_projection_changed", projections: [] });
		write({ type: "subagent_lifecycle", payload: { id: "SubagentA", index: 0, agent: "task", agentSource: "bundled", status: "started", sessionFile: "/tmp/subagent.jsonl" } });
		write({ type: "subagent_progress", payload: { index: 0, agent: "task", agentSource: "bundled", task: "Do work", assignment: "Implement work", sessionFile: "/tmp/subagent.jsonl", progress } });
		write({ type: "subagent_event", payload: { id: "SubagentA", event: { type: "agent_start" } } });
		write({ type: "agent_end", messages: [] });
	}
}
`,
		);

		using client = new RpcClient({ cliPath: scriptPath });
		const lifecycleIds: string[] = [];
		const progressTasks: string[] = [];
		const rawEventTypes: string[] = [];
		const sessionEventTypes: string[] = [];
		client.onSubagentLifecycle(payload => lifecycleIds.push(payload.id));
		client.onSubagentProgress(payload => progressTasks.push(payload.task));
		client.onSubagentEvent(payload => rawEventTypes.push(payload.event.type));
		client.onSessionEvent(event => sessionEventTypes.push(event.type));

		await client.start();
		await expect(client.setSubagentSubscription("events")).resolves.toBe("events");
		await client.promptAndWait("Trigger subagent frames");
		expect(await client.getSubagents()).toHaveLength(1);
		expect(await client.getSubagentMessages({ sessionFile: "/tmp/subagent.jsonl" })).toMatchObject({
			sessionFile: "/tmp/subagent.jsonl",
		});

		expect(lifecycleIds).toEqual(["SubagentA"]);
		expect(progressTasks).toEqual(["Do work"]);
		expect(rawEventTypes).toEqual(["agent_start"]);
		expect(sessionEventTypes).toContain("notice");
		expect(sessionEventTypes).toContain("todo_projection_changed");
	});

	test("delivers startup projections to a passive v1 host and cleans up the expired grace", async () => {
		const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "omp-rpc-passive-v1-projection-"));
		tempPaths.push(tempDir);
		const extensionPath = path.join(tempDir, "startup-projection.ts");
		await Bun.write(
			extensionPath,
			`
export default function (pi) {
	pi.on("session_start", () => {
		pi.setTodoProjection("passive-v1", [{
			id: "startup-phase",
			name: "Startup",
			tasks: [{ id: "startup-task", content: "ready", status: "in_progress" }]
		}]);
	});
}
`,
		);

		const child = ptree.spawn(
			[
				"bun",
				path.join(import.meta.dir, "..", "src", "cli.ts"),
				"--mode",
				"rpc",
				"--provider",
				"anthropic",
				"--model",
				"claude-sonnet-4-5",
				"--extension",
				extensionPath,
			],
			{
				cwd: path.join(import.meta.dir, ".."),
				env: { ...Bun.env, PI_CODING_AGENT_DIR: path.join(tempDir, "agent"), PI_NO_TITLE: "1" },
				stdin: "pipe",
			},
		);
		try {
			const frames = await withTimeout(
				(async () => {
					const received: object[] = [];
					let projectionSeen = false;
					for await (const frame of readJsonl(child.stdout)) {
						if (!isRecord(frame)) continue;
						received.push(frame);
						if (frame.type === "todo_projection_changed" && !projectionSeen) {
							projectionSeen = true;
							child.stdin.write(
								`${JSON.stringify({ type: "negotiate_protocol", protocolVersion: 2, id: "late-v2" })}\n`,
							);
							await child.stdin.flush();
						}
						if (frame.type === "response" && frame.id === "late-v2") return received;
					}
					throw new Error(`RPC v1 output closed before grace cleanup probe: ${child.peekStderr()}`);
				})(),
				10_000,
				"passive RPC v1 startup projection timed out",
			);

			expect(frames[0]).toMatchObject({ type: "ready", protocolVersion: 1 });
			const availableCommandsIndex = frames.findIndex(
				frame => isRecord(frame) && frame.type === "available_commands_update",
			);
			const projectionIndexes = frames.flatMap((frame, index) =>
				isRecord(frame) && frame.type === "todo_projection_changed" ? [index] : [],
			);
			const responseIndex = frames.findIndex(
				frame => isRecord(frame) && frame.type === "response" && frame.id === "late-v2",
			);
			expect(availableCommandsIndex).toBeGreaterThan(0);
			expect(projectionIndexes).toHaveLength(1);
			expect(projectionIndexes[0]!).toBeGreaterThan(availableCommandsIndex);
			expect(responseIndex).toBeGreaterThan(projectionIndexes[0]!);
			expect(frames[projectionIndexes[0]!]).toMatchObject({
				type: "todo_projection_changed",
				projections: [{ namespace: "passive-v1" }],
			});
			expect(frames[responseIndex]).toMatchObject({
				type: "response",
				command: "negotiate_protocol",
				success: true,
			});
		} finally {
			child.stdin.end();
			child.kill();
			await child.exited.catch(() => {});
		}
	});

	test("orders v2 negotiation before the deferred startup projection", async () => {
		const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "omp-rpc-v2-projection-order-"));
		tempPaths.push(tempDir);
		const extensionPath = path.join(tempDir, "startup-projection.ts");
		await Bun.write(
			extensionPath,
			`
export default function (pi) {
	pi.on("session_start", () => {
		pi.setTodoProjection("v2-order", [{
			id: "startup-phase",
			name: "Startup",
			tasks: [{ id: "startup-task", content: "ready", status: "in_progress" }]
		}]);
	});
}
`,
		);

		const child = ptree.spawn(
			[
				"bun",
				path.join(import.meta.dir, "..", "src", "cli.ts"),
				"--mode",
				"rpc",
				"--provider",
				"anthropic",
				"--model",
				"claude-sonnet-4-5",
				"--extension",
				extensionPath,
			],
			{
				cwd: path.join(import.meta.dir, ".."),
				env: { ...Bun.env, PI_CODING_AGENT_DIR: path.join(tempDir, "agent"), PI_NO_TITLE: "1" },
				stdin: "pipe",
			},
		);
		try {
			child.stdin.write(`${JSON.stringify({ type: "negotiate_protocol", protocolVersion: 2, id: "v2" })}\n`);
			await child.stdin.flush();
			const frames = await withTimeout(
				(async () => {
					const received: object[] = [];
					for await (const frame of readJsonl(child.stdout)) {
						if (!isRecord(frame)) continue;
						received.push(frame);
						if (frame.type === "todo_projection_changed") return received;
					}
					throw new Error(`RPC output closed before negotiated startup projection: ${child.peekStderr()}`);
				})(),
				10_000,
				"negotiated RPC startup projection timed out",
			);

			const responseIndex = frames.findIndex(
				frame => isRecord(frame) && frame.type === "response" && frame.id === "v2",
			);
			const projectionIndex = frames.findIndex(frame => isRecord(frame) && frame.type === "todo_projection_changed");
			expect(frames[0]).toMatchObject({ type: "ready", protocolVersion: 1 });
			expect(responseIndex).toBeGreaterThan(0);
			expect(projectionIndex).toBeGreaterThan(responseIndex);
			expect(frames[responseIndex]).toMatchObject({
				type: "response",
				command: "negotiate_protocol",
				success: true,
				data: { protocolVersion: 2 },
			});
		} finally {
			child.stdin.end();
			child.kill();
			await child.exited.catch(() => {});
		}
	});

	test("negotiates v2 after ready before writing an oversized startup projection", async () => {
		const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "omp-rpc-ready-negotiate-projection-"));
		tempPaths.push(tempDir);
		const extensionPath = path.join(tempDir, "startup-projection.ts");
		await Bun.write(
			extensionPath,
			`
export default function (pi) {
	pi.on("session_start", () => {
		pi.setTodoProjection("ready-then-v2", [{
			id: "startup-phase",
			name: "Startup",
			tasks: [{ id: "startup-task", content: "x".repeat(1024 * 1024 + 4096), status: "in_progress" }]
		}]);
	});
}
`,
		);

		const child = ptree.spawn(
			[
				"bun",
				path.join(import.meta.dir, "..", "src", "cli.ts"),
				"--mode",
				"rpc",
				"--provider",
				"anthropic",
				"--model",
				"claude-sonnet-4-5",
				"--extension",
				extensionPath,
			],
			{
				cwd: path.join(import.meta.dir, ".."),
				env: { ...Bun.env, PI_CODING_AGENT_DIR: path.join(tempDir, "agent"), PI_NO_TITLE: "1" },
				stdin: "pipe",
			},
		);
		try {
			const frames = await withTimeout(
				(async () => {
					const received: object[] = [];
					const decoder = new RpcFrameDecoder();
					let negotiationSent = false;
					for await (const rawFrame of readJsonl(child.stdout)) {
						if (!isRecord(rawFrame)) continue;
						received.push(rawFrame);
						if (!negotiationSent && rawFrame.type === "ready") {
							negotiationSent = true;
							child.stdin.write(
								`${JSON.stringify({ type: "negotiate_protocol", protocolVersion: 2, id: "ready-v2" })}\n`,
							);
							await child.stdin.flush();
						}
						const frame = decoder.push(rawFrame);
						if (isRecord(frame) && frame.type === "todo_projection_changed") {
							expect(negotiationSent).toBe(true);
							return { received, projection: frame };
						}
						if (rawFrame.type === "rpc_frame_error" && rawFrame.originalType === "todo_projection_changed") {
							return { received, projection: undefined };
						}
					}
					throw new Error(`RPC output closed before ready-then-negotiate projection: ${child.peekStderr()}`);
				})(),
				10_000,
				"ready-then-negotiate RPC startup projection timed out",
			);

			const responseIndex = frames.received.findIndex(
				frame => isRecord(frame) && frame.type === "response" && frame.id === "ready-v2",
			);
			const firstProjectionTransportIndex = frames.received.findIndex(
				frame =>
					isRecord(frame) &&
					(frame.type === "rpc_chunk" ||
						frame.type === "todo_projection_changed" ||
						(frame.type === "rpc_frame_error" && frame.originalType === "todo_projection_changed")),
			);
			expect(responseIndex).toBeGreaterThan(0);
			expect(firstProjectionTransportIndex).toBeGreaterThan(responseIndex);
			expect(
				frames.received.some(
					frame =>
						isRecord(frame) &&
						frame.type === "rpc_frame_error" &&
						frame.originalType === "todo_projection_changed",
				),
			).toBe(false);
			expect(frames.projection).toMatchObject({
				type: "todo_projection_changed",
				projections: [{ namespace: "ready-then-v2" }],
			});
		} finally {
			child.stdin.end();
			child.kill();
			await child.exited.catch(() => {});
		}
	});

	test("delivers oversized startup projection snapshots after v2 negotiation", async () => {
		const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "omp-rpc-startup-projection-"));
		tempPaths.push(tempDir);
		const extensionPath = path.join(tempDir, "startup-projection.ts");
		await Bun.write(
			extensionPath,
			`
export default function (pi) {
	pi.on("session_start", () => {
		pi.setTodoProjection("rpc-startup", [{
			id: "startup-phase",
			name: "Startup",
			tasks: [{ id: "startup-task", content: "x".repeat(1024 * 1024 + 4096), status: "in_progress" }]
		}]);
	});
}
`,
		);

		const { promise, resolve } =
			Promise.withResolvers<Extract<AgentSessionEvent, { type: "todo_projection_changed" }>>();
		using client = new RpcClient({
			cliPath: path.join(import.meta.dir, "..", "src", "cli.ts"),
			cwd: path.join(import.meta.dir, ".."),
			env: { PI_CODING_AGENT_DIR: path.join(tempDir, "agent"), PI_NO_TITLE: "1" },
			provider: "anthropic",
			model: "claude-sonnet-4-5",
			args: ["--extension", extensionPath],
		});
		client.onSessionEvent(event => {
			if (event.type === "todo_projection_changed") resolve(event);
		});

		await client.start();
		const event = await withTimeout(promise, 10_000, "oversized startup projection event never reached RpcClient");
		const projection = event.projections[0];
		expect(projection?.namespace).toBe("rpc-startup");
		expect(projection?.phases[0]?.tasks[0]).toMatchObject({
			id: "startup-task",
			status: "in_progress",
		});
		expect(projection?.phases[0]?.tasks[0]?.content).toHaveLength(1024 * 1024 + 4096);
	});

	test("reports an oversized startup projection cleanly to a v1 client", async () => {
		const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "omp-rpc-v1-startup-projection-"));
		tempPaths.push(tempDir);
		const extensionPath = path.join(tempDir, "startup-projection.ts");
		await Bun.write(
			extensionPath,
			`
export default function (pi) {
	pi.on("session_start", () => {
		pi.setTodoProjection("rpc-startup", [{
			id: "startup-phase",
			name: "Startup",
			tasks: [{ id: "startup-task", content: "x".repeat(1024 * 1024 + 4096), status: "in_progress" }]
		}]);
	});
}
`,
		);

		const child = ptree.spawn(
			[
				"bun",
				path.join(import.meta.dir, "..", "src", "cli.ts"),
				"--mode",
				"rpc",
				"--provider",
				"anthropic",
				"--model",
				"claude-sonnet-4-5",
				"--extension",
				extensionPath,
			],
			{
				cwd: path.join(import.meta.dir, ".."),
				env: { ...Bun.env, PI_CODING_AGENT_DIR: path.join(tempDir, "agent"), PI_NO_TITLE: "1" },
				stdin: "pipe",
			},
		);
		try {
			child.stdin.write(`${JSON.stringify({ type: "get_state", id: "v1-probe" })}\n`);
			await child.stdin.flush();
			const frames = await withTimeout(
				(async () => {
					const received: object[] = [];
					for await (const frame of readJsonl(child.stdout)) {
						if (!isRecord(frame)) continue;
						received.push(frame);
						if (frame.type === "response" && frame.id === "v1-probe") return received;
					}
					throw new Error(`RPC v1 output closed before probe response: ${child.peekStderr()}`);
				})(),
				10_000,
				"RPC v1 startup projection probe timed out",
			);

			expect(frames).toContainEqual({
				type: "rpc_frame_error",
				originalType: "todo_projection_changed",
				error: "RPC frame exceeded the transport limit",
			});
			expect(frames.some(frame => isRecord(frame) && frame.type === "todo_projection_changed")).toBe(false);
		} finally {
			child.stdin.end();
			child.kill();
			await child.exited.catch(() => {});
		}
	});
});
