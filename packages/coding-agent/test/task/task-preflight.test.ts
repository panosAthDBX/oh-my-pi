import { afterEach, beforeEach, describe, expect, it, vi } from "bun:test";
import { AsyncJobManager } from "@oh-my-pi/pi-coding-agent/async/job-manager";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { AgentLifecycleManager } from "@oh-my-pi/pi-coding-agent/registry/agent-lifecycle";
import { AgentRegistry } from "@oh-my-pi/pi-coding-agent/registry/agent-registry";
import { TaskTool } from "@oh-my-pi/pi-coding-agent/task";
import * as discoveryModule from "@oh-my-pi/pi-coding-agent/task/discovery";
import * as executorModule from "@oh-my-pi/pi-coding-agent/task/executor";
import {
	hashTrustedTaskInvocationEnvelope,
	registerTrustedTaskInvocationModelOverride,
	resetTrustedTaskInvocationModelOverridesForTests,
} from "@oh-my-pi/pi-coding-agent/task/invocation-model-override";
import type { AgentDefinition, SingleResult, TaskParams } from "@oh-my-pi/pi-coding-agent/task/types";
import type { ToolSession } from "@oh-my-pi/pi-coding-agent/tools";

const taskAgent: AgentDefinition = {
	name: "task",
	description: "General-purpose task agent",
	systemPrompt: "You are a task agent.",
	source: "bundled",
};

const babysitterAgent: AgentDefinition = {
	...taskAgent,
	name: "babysitter-task",
};

function createSession(options: {
	manager: AsyncJobManager;
	settings?: Record<string, unknown>;
	spawns?: string | boolean;
	sessionId?: string;
}): ToolSession {
	return {
		cwd: "/tmp",
		hasUI: false,
		settings: Settings.isolated({ "async.enabled": true, ...options.settings }),
		getSessionFile: () => null,
		getSessionId: () => options.sessionId ?? "test-session",
		getSessionSpawns: () => options.spawns ?? "*",
		asyncJobManager: options.manager,
	} as unknown as ToolSession;
}

function textOf(result: { content: Array<{ type: string; text?: string }> }): string {
	const content = result.content.find(part => part.type === "text");
	return content?.type === "text" ? (content.text ?? "") : "";
}

function resultFor(id: string): SingleResult {
	return {
		index: 0,
		id,
		agent: "task",
		agentSource: "bundled",
		task: "prompt",
		assignment: "work",
		exitCode: 0,
		output: "done",
		stderr: "",
		truncated: false,
		durationMs: 1,
		tokens: 0,
		requests: 1,
	};
}

function mockDiscovery(agents: AgentDefinition[] = [taskAgent]): void {
	vi.spyOn(discoveryModule, "discoverAgents").mockResolvedValue({ agents, projectAgentsDir: null });
}

describe("task async preflight", () => {
	const managers: AsyncJobManager[] = [];

	beforeEach(() => {
		AgentRegistry.resetGlobalForTests();
		AgentLifecycleManager.resetGlobalForTests();
		resetTrustedTaskInvocationModelOverridesForTests();
	});

	afterEach(async () => {
		vi.restoreAllMocks();
		for (const manager of managers.splice(0)) await manager.dispose({ timeoutMs: 1_000 });
		AgentLifecycleManager.resetGlobalForTests();
		AgentRegistry.resetGlobalForTests();
		resetTrustedTaskInvocationModelOverridesForTests();
	});

	function manager(): AsyncJobManager {
		const result = new AsyncJobManager({ onJobComplete: () => {} });
		managers.push(result);
		return result;
	}

	it.each([
		{
			name: "Unknown",
			params: { agent: "missing", name: "Unknown", task: "Work." },
			expectation: 'Unknown agent "missing"',
		},
		{
			name: "Disabled",
			params: { agent: "task", name: "Disabled", task: "Work." },
			settings: { "task.disabledAgents": ["task"] },
			expectation: 'Agent "task" is disabled',
		},
		{
			name: "Disallowed",
			params: { agent: "task", name: "Disallowed", task: "Work." },
			spawns: "scout",
			expectation: "Cannot spawn 'task'",
		},
	])(
		"returns $name policy errors before registering an async job",
		async ({ name, params, settings, spawns, expectation }) => {
			mockDiscovery();
			const jobs = manager();
			const tool = await TaskTool.create(createSession({ manager: jobs, settings, spawns }));

			const result = await tool.execute("preflight", params as TaskParams);

			expect(textOf(result)).toContain(expectation);
			expect(jobs.getJob(name)).toBeUndefined();
		},
	);

	it("rejects an invalid async batch atomically before dispatching any item", async () => {
		mockDiscovery();
		const runSubprocess = vi.spyOn(executorModule, "runSubprocess").mockResolvedValue(resultFor("unexpected"));
		const jobs = manager();
		const register = vi.spyOn(jobs, "register");
		const tool = await TaskTool.create(createSession({ manager: jobs, settings: { "task.batch": true } }));

		const result = await tool.execute("mixed-preflight", {
			context: "Shared context.",
			tasks: [
				{ name: "Invalid", agent: "missing", task: "Do invalid work." },
				{ name: "AlsoInvalid", agent: "also-missing", task: "Do more invalid work." },
				{ name: "Valid", agent: "task", task: "Do valid work." },
			],
		} as TaskParams);

		const text = textOf(result);
		expect(text).toContain('Task Invalid failed preflight: Unknown agent "missing"');
		expect(text).toContain('Task AlsoInvalid failed preflight: Unknown agent "also-missing"');
		expect(register).not.toHaveBeenCalled();
		expect(runSubprocess).not.toHaveBeenCalled();
		expect(jobs.getJob("Invalid")).toBeUndefined();
		expect(jobs.getJob("AlsoInvalid")).toBeUndefined();
		expect(jobs.getJob("Valid")).toBeUndefined();
	});

	it("rejects an invalid synchronous batch before running any item", async () => {
		mockDiscovery();
		const runSubprocess = vi.spyOn(executorModule, "runSubprocess").mockResolvedValue(resultFor("unexpected"));
		const jobs = manager();
		const register = vi.spyOn(jobs, "register");
		const tool = await TaskTool.create(
			createSession({ manager: jobs, settings: { "async.enabled": false, "task.batch": true } }),
		);

		const result = await tool.execute("sync-preflight", {
			context: "Shared context.",
			tasks: [
				{ name: "Invalid", agent: "missing", task: "Do invalid work." },
				{ name: "Valid", agent: "task", task: "Do valid work." },
			],
		} as TaskParams);

		expect(textOf(result)).toContain('Task Invalid failed preflight: Unknown agent "missing"');
		expect(register).not.toHaveBeenCalled();
		expect(runSubprocess).not.toHaveBeenCalled();
		expect(jobs.getJob("Invalid")).toBeUndefined();
		expect(jobs.getJob("Valid")).toBeUndefined();
	});

	it("applies trusted invocation-local model roles independently across Babysitter effects", async () => {
		mockDiscovery([babysitterAgent]);
		const jobs = manager();
		const session = createSession({ manager: jobs, settings: { "async.enabled": false, "task.batch": true } });
		const selectors: Record<string, string> = {
			smol: "openai/gpt-5.6-mini",
			plan: "openai/gpt-5.6-plan",
			builder: "openai/gpt-5.6-builder",
			vision: "google/gemini-vision",
			advisor: "anthropic/claude-advisor",
			tiny: "openai/gpt-5.6-tiny",
			slow: "openai/gpt-5.6-slow",
		};
		for (const [role, selector] of Object.entries(selectors)) session.settings.setModelRole(role, selector);
		const observed: Array<{ modelOverride?: string | string[]; modelRole?: string }> = [];
		vi.spyOn(executorModule, "runSubprocess").mockImplementation(async args => {
			observed.push({ modelOverride: args.modelOverride, modelRole: args.modelRole });
			return { ...resultFor(args.id), agent: "babysitter-task" };
		});
		const tool = await TaskTool.create(session);

		for (const role of Object.keys(selectors)) {
			const toolCallId = `babysitter-${role}`;
			const name = `Owner-${role}`;
			const params = {
				context: "Trusted Babysitter dispatch.",
				tasks: [{ name, agent: "babysitter-task", task: `Run ${role}.` }],
			} as TaskParams;
			registerTrustedTaskInvocationModelOverride({
				scopeId: "test-session",
				toolCallId,
				model: `@${role}`,
				agent: "babysitter-task",
				name,
				envelopeSha256: hashTrustedTaskInvocationEnvelope(params),
			});
			const result = await tool.execute(toolCallId, params);
			expect(textOf(result)).toContain("done");
		}

		expect(observed).toEqual(
			Object.entries(selectors).map(([role, selector]) => ({ modelOverride: [selector], modelRole: role })),
		);
	});

	it("ignores model-authored hidden overrides without a trusted grant", async () => {
		mockDiscovery([babysitterAgent]);
		const observed: Array<string | string[] | undefined> = [];
		vi.spyOn(executorModule, "runSubprocess").mockImplementation(async args => {
			observed.push(args.modelOverride);
			return { ...resultFor(args.id), agent: "babysitter-task" };
		});
		const tool = await TaskTool.create(createSession({ manager: manager(), settings: { "async.enabled": false } }));

		await tool.execute("unsigned-model", {
			name: "Unsigned",
			agent: "babysitter-task",
			task: "Attempt an unsigned override.",
			modelOverride: "@smol",
		} as TaskParams);

		expect(observed).toEqual([[]]);
	});

	it("rejects a trusted override when the normalized owner identity changes", async () => {
		mockDiscovery([babysitterAgent]);
		const runSubprocess = vi.spyOn(executorModule, "runSubprocess");
		const tool = await TaskTool.create(createSession({ manager: manager(), settings: { "async.enabled": false } }));
		registerTrustedTaskInvocationModelOverride({
			scopeId: "test-session",
			toolCallId: "mismatched-owner",
			model: "@smol",
			agent: "babysitter-task",
			name: "ExpectedOwner",
			envelopeSha256: hashTrustedTaskInvocationEnvelope({
				name: "ExpectedOwner",
				agent: "babysitter-task",
				task: "Original task.",
			}),
		});

		const result = await tool.execute("mismatched-owner", {
			name: "ChangedOwner",
			agent: "babysitter-task",
			task: "Mutated task.",
		} as TaskParams);

		expect(textOf(result)).toContain("does not match the normalized task invocation");
		expect(runSubprocess).not.toHaveBeenCalled();
	});

	it("rejects and consumes a grant when a later hook mutates the authenticated envelope", async () => {
		mockDiscovery([babysitterAgent]);
		const observed: Array<string | string[] | undefined> = [];
		vi.spyOn(executorModule, "runSubprocess").mockImplementation(async args => {
			observed.push(args.modelOverride);
			return { ...resultFor(args.id), agent: "babysitter-task" };
		});
		const tool = await TaskTool.create(createSession({ manager: manager(), settings: { "async.enabled": false } }));
		const original = { name: "BoundOwner", agent: "babysitter-task", task: "Original task." } as TaskParams;
		registerTrustedTaskInvocationModelOverride({
			scopeId: "test-session",
			toolCallId: "mutated-envelope",
			model: "@smol",
			agent: "babysitter-task",
			name: "BoundOwner",
			envelopeSha256: hashTrustedTaskInvocationEnvelope(original),
		});

		const rejected = await tool.execute("mutated-envelope", { ...original, task: "Mutated task." });
		expect(textOf(rejected)).toContain("does not match the normalized task invocation");
		await tool.execute("mutated-envelope", original);

		expect(observed).toEqual([[]]);
	});

	it("does not allow another AgentSession to consume a matching trusted grant", async () => {
		mockDiscovery([babysitterAgent]);
		const observed: Array<string | string[] | undefined> = [];
		vi.spyOn(executorModule, "runSubprocess").mockImplementation(async args => {
			observed.push(args.modelOverride);
			return { ...resultFor(args.id), agent: "babysitter-task" };
		});
		const params = { name: "ScopedOwner", agent: "babysitter-task", task: "Scoped task." } as TaskParams;
		registerTrustedTaskInvocationModelOverride({
			scopeId: "session-a",
			toolCallId: "shared-tool-call",
			model: "@smol",
			agent: "babysitter-task",
			name: "ScopedOwner",
			envelopeSha256: hashTrustedTaskInvocationEnvelope(params),
		} as Parameters<typeof registerTrustedTaskInvocationModelOverride>[0]);
		const toolB = await TaskTool.create(
			createSession({ manager: manager(), settings: { "async.enabled": false }, sessionId: "session-b" }),
		);
		const toolA = await TaskTool.create(
			createSession({ manager: manager(), settings: { "async.enabled": false }, sessionId: "session-a" }),
		);

		await toolB.execute("shared-tool-call", params);
		await toolA.execute("shared-tool-call", params);

		expect(observed[0]).toEqual([]);
		expect(observed[1]).not.toEqual([]);
	});
});
